import os
import logging
import math
import time
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain_core.retrievers import BaseRetriever
import chromadb
import streamlit as st
from typing import List, Dict, Optional, Any, Union, Set
import re
import json
from langchain.retrievers.ensemble import EnsembleRetriever
import traceback
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Retrieve values from .env
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")
CHROMA_SERVER_HOST = st.secrets.chroma.server
CHROMA_SERVER_PORT = int(st.secrets.chroma.port)
CLASS_FOLDERS = os.getenv("CLASS_FOLDERS", "data/class_folders")
OPENAI_API_KEY = st.secrets.openai.api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_with_ocr(pdf_path):
    """Extracts text from scanned PDFs using OCR."""
    logging.info(f"Using OCR to extract text from {pdf_path}...")
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    text = "\n".join(pytesseract.image_to_string(image) for image in images)
    return text

def semantic_chunking(text: str) -> List[str]:
    """
    Split text into semantic chunks based on headers, paragraphs and sections
    rather than arbitrary character counts.
    """
    # Check if text is empty or None
    if not text or not text.strip():
        return []
    
    # Extract headers to use as context for chunks
    header_pattern = r'(?:\n\s*#{1,6}\s+(.+))|(?:\n\s*([A-Z][A-Za-z\s]+)\n\s*[-=]+\s*\n)'
    headers = re.findall(header_pattern, text)
    current_header = ""
    
    # Clean headers list - each match returns a tuple with groups
    headers = [h[0] if h[0] else h[1] for h in headers if h[0] or h[1]]
    
    # Split by headers (various markdown or document formats)
    split_pattern = r'(?:\n\s*#{1,6}\s+.+)|(?:\n\s*[A-Z][A-Za-z\s]+\n\s*[-=]+\s*\n)'
    sections = re.split(split_pattern, text)
    
    # Further split large sections by paragraphs if needed
    chunks = []
    
    for i, section in enumerate(sections):
        # Set current header if available
        if i-1 < len(headers) and i > 0:
            current_header = headers[i-1]
            
        if not section.strip():
            continue
            
        # Add header to the section for context preservation
        if current_header and i > 0:
            section_with_header = f"{current_header}\n\n{section}"
        else:
            section_with_header = section
            
        if len(section_with_header) < 2000:
            # Only add non-empty sections
            chunks.append(section_with_header.strip())
        else:
            # Split large sections by paragraphs
            paragraphs = re.split(r'\n{2,}', section)
            current_chunk = current_header + "\n\n" if current_header else ""
            
            for para in paragraphs:
                if not para.strip():
                    continue
                    
                # If adding this paragraph would exceed limit, store current chunk and start new one
                if len(current_chunk) + len(para) < 1500:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    # Start new chunk with header for context
                    current_chunk = (current_header + "\n\n" if current_header else "") + para + "\n\n"
            
            if current_chunk:  # Add the last chunk
                chunks.append(current_chunk.strip())
    
    # Handle case where no chunks were created
    if not chunks and text.strip():
        # Fall back to simple paragraph splitting
        paragraphs = re.split(r'\n{2,}', text)
        current_chunk = ""
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            if len(current_chunk) + len(para) < 1500:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
    return chunks

def semantic_chunking_v2(text: str) -> List[str]:
    """
    Splits text into semantic chunks based on headers, paragraphs, and sections,
    aiming for better accuracy and context preservation.
    """
    if not text or not text.strip():
        return []

    # Improved header extraction (captures the header text more directly)
    header_pattern = r'(?:\n\s*(#{1,6})\s+(.+))|(?:\n\s*([A-Z][A-Za-z\s]+)\n\s*[-=]+\s*\n)'
    header_matches = list(re.finditer(header_pattern, text))

    chunks = []
    start_index = 0
    current_header = ""

    for i, match in enumerate(header_matches):
        header_level = len(match.group(1)) if match.group(1) else None
        header_text = match.group(2) if match.group(2) else match.group(3)
        header_start = match.start()

        # Process the content between the previous header (or start) and the current header
        content = text[start_index:header_start].strip()
        if content:
            # Prepend the previous header for context
            chunk_content = f"{current_header}\n\n{content}" if current_header else content
            if len(chunk_content) < 2000:
                chunks.append(chunk_content.strip())
            else:
                # Split large content by paragraphs
                paragraphs = re.split(r'\n{2,}', content)
                current_paragraph_chunk = f"{current_header}\n\n" if current_header else ""
                for para in paragraphs:
                    if not para.strip():
                        continue
                    if len(current_paragraph_chunk) + len(para) < 1500:
                        current_paragraph_chunk += para + "\n\n"
                    else:
                        if current_paragraph_chunk.strip():
                            chunks.append(current_paragraph_chunk.strip())
                        current_paragraph_chunk = (f"{current_header}\n\n" if current_header else "") + para + "\n\n"
                if current_paragraph_chunk.strip():
                    chunks.append(current_paragraph_chunk.strip())

        # Update the current header and start index for the next iteration
        current_header = header_text.strip()
        start_index = match.end()

    # Process any remaining content after the last header
    remaining_content = text[start_index:].strip()
    if remaining_content:
        chunk_content = f"{current_header}\n\n{remaining_content}" if current_header else remaining_content
        if len(chunk_content) < 2000:
            chunks.append(chunk_content.strip())
        else:
            paragraphs = re.split(r'\n{2,}', remaining_content)
            current_paragraph_chunk = f"{current_header}\n\n" if current_header else ""
            for para in paragraphs:
                if not para.strip():
                    continue
                if len(current_paragraph_chunk) + len(para) < 1500:
                    current_paragraph_chunk += para + "\n\n"
                else:
                    if current_paragraph_chunk.strip():
                        chunks.append(current_paragraph_chunk.strip())
                    current_paragraph_chunk = (f"{current_header}\n\n" if current_header else "") + para + "\n\n"
            if current_paragraph_chunk.strip():
                chunks.append(current_paragraph_chunk.strip())

    # Fallback for text with no headers (process by paragraphs)
    if not chunks and text.strip():
        paragraphs = re.split(r'\n{2,}', text)
        current_chunk = ""
        for para in paragraphs:
            if not para.strip():
                continue
            if len(current_chunk) + len(para) < 1500:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()] # Ensure no empty chunks

def _connect_chroma(host: str, port: int) -> chromadb.HttpClient:
    """Connects to the ChromaDB server."""
    return chromadb.HttpClient(host=host, port=port)

def _get_openai_embeddings(api_key: str, model_name: str = "text-embedding-3-small", dimensions: int = 384):
    """Initializes OpenAI embeddings."""
    return OpenAIEmbeddings(
        openai_api_key=api_key,
        model=model_name,
        dimensions=dimensions
    )

def _add_or_update_collection(
    chroma_client: chromadb.HttpClient,
    collection_name: str,
    all_text_chunks: List[str],
    metadata_list: List[dict],
    openai_api_key: str
):
    """Adds or updates documents in a ChromaDB collection."""
    try:
        # Get the list of collection names directly.
        collections = chroma_client.list_collections()
        logging.info(f"Existing collections: {collections}")
        
        # Check if the target collection name exists.
        collection_exists = collection_name in collections

        if collection_exists:
            # Retrieve the collection using get_collection.
            collection = chroma_client.get_collection(name=collection_name)
            existing_count = collection.count()
            logging.info(f"Collection '{collection_name}' exists with {existing_count} documents.")

            if existing_count == 0:
                collection.add(
                    documents=all_text_chunks,
                    metadatas=metadata_list,
                    ids=[f"{metadata['source']}_p{metadata['page']}_chunk_{metadata['chunk_index']}" for metadata in metadata_list]
                )
                logging.info(f"Added {len(all_text_chunks)} chunks to empty collection '{collection_name}'.")
            else:
                embeddings = _get_openai_embeddings(openai_api_key)
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name=collection_name,
                    embedding_function=embeddings
                )
                added_count = 0
                for i, (chunk, metadata) in enumerate(zip(all_text_chunks, metadata_list)):
                    doc_id = f"{metadata['source']}_p{metadata['page']}_chunk_{metadata['chunk_index']}"
                    try:
                        vectorstore.add_texts(
                            texts=[chunk],
                            metadatas=[metadata],
                            ids=[doc_id]
                        )
                        added_count += 1
                        if (i + 1) % 100 == 0 or i + 1 == len(all_text_chunks):
                            logging.info(f"Added {i + 1}/{len(all_text_chunks)} chunks to collection '{collection_name}'.")
                    except Exception as e:
                        logging.error(f"Error adding chunk {i + 1}/{len(all_text_chunks)}: {e}")
                logging.info(f"Updated collection '{collection_name}' with {added_count} additional chunks.")

        else:
            embeddings = _get_openai_embeddings(openai_api_key)
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"dimension": embeddings.embedding_ctx_length, "model": embeddings.model}
            )
            collection.add(
                documents=all_text_chunks,
                metadatas=metadata_list,
                ids=[f"{metadata['source']}_p{metadata['page']}_chunk_{metadata['chunk_index']}" for metadata in metadata_list]
            )
            logging.info(f"Created new collection '{collection_name}' with {len(all_text_chunks)} chunks.")

    except Exception as e:
        logging.error(f"Error creating/updating collection '{collection_name}': {e}")
        return None

    return collection_name

def process_pdf(pdf_path: str) -> List[str]:
    """Processes a single PDF file to extract text and chunk it."""
    logging.info(f"Processing PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_chunks = []
    metadata_list = []
    pdf_file = os.path.basename(pdf_path)

    for page_num, page in enumerate(pages):
        extracted_text = page.page_content

        if not extracted_text.strip():
            logging.info(f"Trying OCR for page {page_num + 1} of {pdf_file}")
            page_images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1, poppler_path=POPPLER_PATH)
            extracted_text = pytesseract.image_to_string(page_images[0])

        if not extracted_text.strip():
            logging.warning(f"No text extracted from page {page_num + 1} of {pdf_file}")
            continue

        # Try semantic chunking first
        semantic_chunks = semantic_chunking(extracted_text)
        if semantic_chunks:
            chunks_to_use = semantic_chunks
            chunk_type = "semantic"
        else:
            # Fall back to traditional chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks_to_use = text_splitter.split_text(extracted_text)
            chunk_type = "recursive"

        for idx, chunk in enumerate(chunks_to_use):
            text_chunks.append(chunk)
            metadata_list.append({
                "source": pdf_file,
                "page": page_num + 1,
                "chunk_index": idx,
                "document_id": f"{pdf_file}_p{page_num + 1}",
                "chunk_type": chunk_type
            })
    return text_chunks, metadata_list

def process_all_pdfs(class_folder: str, collection_name: str):
    """
    Processes all PDFs in a class folder and creates or updates a ChromaDB collection.
    Uses semantic chunking for better context preservation, with a fallback to recursive chunking.

    Args:
        class_folder: The folder containing PDFs to process.
        collection_name: The name of the collection to use in ChromaDB.

    Returns:
        The name of the ChromaDB collection if processing is successful, otherwise None.
    """
    pdf_files = [f for f in os.listdir(class_folder) if f.endswith(".pdf")]

    if not pdf_files:
        logging.warning(f"⚠️ No PDFs found in {class_folder}. Please upload documents.")
        return None

    logging.info(f"Processing {len(pdf_files)} PDFs from {class_folder}...")

    all_text_chunks = []
    all_metadata = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(class_folder, pdf_file)
        chunks, metadata = process_pdf(pdf_path)
        all_text_chunks.extend(chunks)
        all_metadata.extend(metadata)

    if not all_text_chunks:
        logging.error("No valid text chunks found in PDFs.")
        return None

    logging.info(f"Extracted a total of {len(all_text_chunks)} chunks from all PDFs.")

    # Connect to ChromaDB
    try:
        chroma_client = _connect_chroma(CHROMA_SERVER_HOST, CHROMA_SERVER_PORT)
    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {e}")
        return None

    # Add or update the ChromaDB collection
    return _add_or_update_collection(
        chroma_client,
        collection_name,
        all_text_chunks,
        all_metadata,
        OPENAI_API_KEY
    )

def rewrite_query(question: str, llm) -> Dict[str, Any]:
    """
    Rewrite the original query to improve retrieval performance by:
    1. Creating a more specific search query.
    2. Identifying key entities and concepts.
    3. Generating alternative phrasings to catch different terminology.
    
    Args:
        question: The original user question.
        llm: The language model to use for rewriting.
    
    Returns:
        A dictionary containing:
            - "rewritten_query": A refined search query.
            - "alternative_queries": A list of alternative phrasings.
            - "key_entities": A list of key entities and concepts.
    """
    rewrite_prompt = (
        "Given a user question, help me optimize it for a retrieval system.\n\n"
        "Original question: {question}\n\n"
        "Please provide:\n"
        "1. A rewritten, more specific search query that would help find relevant information\n"
        "2. 2-3 alternative phrasings that might use different terminology\n"
        "3. A list of key entities and concepts to look for in the documents\n\n"
        "Format your response as a JSON object with keys 'rewritten_query', 'alternative_queries', and 'key_entities'.\n"
        "JSON Example:\n"
        '{"rewritten_query": "specific query here", "alternative_queries": ["alternative 1", "alternative 2"], "key_entities": ["entity1", "entity2"]}'
    )

    try:
        # Format prompt and invoke the LLM
        formatted_prompt = rewrite_prompt.format(question=question)
        response = llm.invoke(formatted_prompt)
        logging.info(f"Query rewriting response: {response.content}")
        content = response.content

        # Attempt to extract JSON from the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed_response = json.loads(json_str)
                required_keys = ['rewritten_query', 'alternative_queries', 'key_entities']
                if not all(key in parsed_response for key in required_keys):
                    raise ValueError("Missing required keys in response")
                return parsed_response
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON format: {json_str}")

        # Fallback extraction if JSON extraction fails
        logging.warning("Could not extract valid JSON from response. Using fallback extraction method.")
        fallback = {
            "rewritten_query": question,
            "alternative_queries": [question],
            "key_entities": []
        }

        # Attempt manual extraction for each section
        fallback["rewritten_query"] = extract_rewritten_query(content) or fallback["rewritten_query"]
        fallback["alternative_queries"] = extract_alternative_queries(content) or fallback["alternative_queries"]
        fallback["key_entities"] = extract_key_entities(content) or fallback["key_entities"]

        return fallback

    except Exception as e:
        logging.warning(f"Error in query rewriting: {e}")
        # Return a basic fallback if an exception occurs
        return {
            "rewritten_query": question,
            "alternative_queries": [question],
            "key_entities": []
        }

def extract_rewritten_query(text: str) -> str:
    """Extract the rewritten query from text using a regex pattern."""
    match = re.search(r'rewritten\s+query:?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def extract_alternative_queries(text: str) -> List[str]:
    """Extract alternative queries from text using regex."""
    alt_section = re.search(r'alternative.+?phrasing.+?:(.+?)(?:\n\d|\n[A-Za-z]|$)', text, re.IGNORECASE | re.DOTALL)
    if alt_section:
        alternatives = re.findall(r'[-*•]?\s*(.+?)(?:\n|$)', alt_section.group(1), re.DOTALL)
        return [alt.strip() for alt in alternatives if alt.strip()]
    return []

def extract_key_entities(text: str) -> List[str]:
    """Extract key entities from text using regex."""
    entity_section = re.search(r'key entities.+?:(.+?)(?:\n\d|\n[A-Za-z]|$)', text, re.IGNORECASE | re.DOTALL)
    if entity_section:
        entities = re.findall(r'[-*•]?\s*(.+?)(?:\n|$)', entity_section.group(1), re.DOTALL)
        return [entity.strip() for entity in entities if entity.strip()]
    return []

class HybridRetriever(BaseRetriever):
    """
    A hybrid retrieval approach that combines semantic search with keyword search,
    query rewriting to improve results, and Maximal Marginal Relevance (MMR) for diversity.
    Implements the BaseRetriever interface for LangChain compatibility.
    """
    
    def __init__(self, vectorstore, llm, search_kwargs: Dict[str, Any] = None,
                 use_mmr: bool = True, mmr_diversity: float = 0.25):
        super().__init__()
        self._vectorstore = vectorstore
        self._llm = llm
        self._search_kwargs = search_kwargs or {"k": 5}
        self._use_mmr = use_mmr
        self._mmr_diversity = mmr_diversity
        self._retrieval_cache: Dict[str, List[Any]] = {}
    
    @property
    def vectorstore(self):
        """Expose the underlying vectorstore."""
        return self._vectorstore
    
    @property
    def llm(self):
        """Expose the underlying llm."""
        return self._llm
    
    @property
    def search_kwargs(self):
        """Expose the search kwargs."""
        return self._search_kwargs
    
    @property
    def use_mmr(self):
        """Expose the MMR usage flag."""
        return self._use_mmr
    
    @property
    def mmr_diversity(self):
        """Expose the MMR diversity parameter."""
        return self._mmr_diversity
    
    @property
    def retrieval_cache(self):
        """Expose the retrieval cache."""
        return self._retrieval_cache

    def get_relevant_documents(self, query: str) -> List[Any]:
        cache_key = query.strip().lower()
        if cache_key in self.retrieval_cache:
            logging.info(f"Cache hit for query: {cache_key[:30]}...")
            return self.retrieval_cache[cache_key]
        
        try:
            # Step 1: Rewrite the query for better retrieval
            query_variations = rewrite_query(query, self.llm)
            rewritten_query = query_variations.get("rewritten_query", query)
            
            # Step 2: Retrieve using the main rewritten query
            main_results = self._retrieve_main_results(rewritten_query)
            
            # Step 3: Process key entities for additional context
            entity_results = self._retrieve_entity_results(query_variations.get("key_entities", []))
            
            # Step 4: Retrieve using alternative phrasings
            alt_results = self._retrieve_alternative_results(query_variations.get("alternative_queries", []),
                                                             rewritten_query)
            
            # Step 5: Combine and deduplicate results
            all_results = main_results + alt_results + entity_results
            unique_results = self._deduplicate_results(all_results)
            
            # Limit final results to desired k
            k = self.search_kwargs.get("k", 5)
            final_results = unique_results[:k]
            
            # Cache results
            self.retrieval_cache[cache_key] = final_results
            return final_results
        
        except Exception as e:
            logging.error(f"Error in hybrid retrieval: {e}")
            return self._fallback_retrieval(query)
    
    def _retrieve_main_results(self, rewritten_query: str) -> List[Any]:
        """Retrieve documents using the main rewritten query, optionally with MMR."""
        k = self.search_kwargs.get("k", 5)
        if self.use_mmr:
            try:
                return self.vectorstore.max_marginal_relevance_search(
                    rewritten_query,
                    k=k,
                    fetch_k=k * 2,  # Fetch extra docs for diversity
                    lambda_mult=self.mmr_diversity
                )
            except Exception as mmr_err:
                logging.warning(f"MMR retrieval failed, falling back to standard search: {mmr_err}")
        # Fallback to standard similarity search if MMR is disabled or fails
        return self.vectorstore.similarity_search(rewritten_query, **self.search_kwargs)
    
    def _retrieve_entity_results(self, key_entities: List[str]) -> List[Any]:
        """Retrieve additional documents based on key entities."""
        results = []
        for entity in key_entities[:3]:  # Limit to top 3 entities
            if entity and len(entity) > 3:
                try:
                    docs = self.vectorstore.similarity_search(entity, k=2)
                    results.extend(docs)
                except Exception as e:
                    logging.warning(f"Error retrieving docs for entity '{entity}': {e}")
        return results
    
    def _retrieve_alternative_results(self, alternative_queries: List[str], rewritten_query: str) -> List[Any]:
        """Retrieve documents using alternative phrasings, excluding the main query."""
        results = []
        for alt_query in alternative_queries[:2]:  # Limit to top 2 alternatives
            if alt_query and alt_query.lower() != rewritten_query.lower():
                try:
                    docs = self.vectorstore.similarity_search(alt_query, k=3)
                    results.extend(docs)
                except Exception as e:
                    logging.warning(f"Error retrieving docs for alternative query '{alt_query}': {e}")
        return results

    def _deduplicate_results(self, documents: List[Any]) -> List[Any]:
        """Deduplicate documents based on a unique ID from document metadata."""
        seen_ids = set()
        unique_docs = []
        for doc in documents:
            try:
                # Construct a unique identifier based on metadata attributes
                meta = doc.metadata
                doc_id = f"{meta.get('source', '')}_p{meta.get('page', '')}_chunk_{meta.get('chunk_index', '')}"
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)
            except Exception as dedup_err:
                logging.warning(f"Error during deduplication: {dedup_err}")
                if doc not in unique_docs:
                    unique_docs.append(doc)
        return unique_docs

    def _fallback_retrieval(self, query: str) -> List[Any]:
        """Fallback retrieval using basic similarity search."""
        try:
            return self.vectorstore.similarity_search(query, k=self.search_kwargs.get("k", 5))
        except Exception as fallback_err:
            logging.error(f"Fallback retrieval also failed: {fallback_err}")
            return []  # Return an empty list as a last resort

    def clear_cache(self) -> None:
        """Clear the retrieval cache."""
        self.retrieval_cache = {}

class QAAgent:
    """Enhanced QA agent with improved context handling and query analysis."""
    
    def __init__(self, qa_chain, llm):
        self.qa_chain = qa_chain
        self.llm = llm
        self.question_cache: Dict[str, Dict[str, Any]] = {}
        self.last_error = None
        self.stats = {
            "questions_asked": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "cache_hits": 0
        }

    def ask_question(self, question: str) -> str:
        """
        Process a user question and return an answer with source citations.
        Implements caching, error recovery, and robust source handling.
        """
        # Update statistics
        self.stats["questions_asked"] += 1
        cache_key = question.strip().lower()
        
        # Return cached answer if available
        if cache_key in self.question_cache:
            self.stats["cache_hits"] += 1
            return self.question_cache[cache_key]["formatted_answer"]
        
        try:
            start_time = time.time()
            
            # Analyze the question to improve retrieval
            query_analysis = self.analyze_query(question)
            
            # Use retry logic for qa_chain invocation
            max_retries = 2
            retry_count = 0
            response = None
            
            while retry_count <= max_retries:
                try:
                    # We should format both analysis, rewritten query and question into on string that is injected
                    combined_question = f"###Analysis:\n{query_analysis}\n\n###Question:\n{question}"
                    response = self.qa_chain.invoke({
                        "query": combined_question,
                    })
                    break  # Success: exit loop
                except Exception as chain_error:
                    retry_count += 1
                    logging.warning(f"Chain invocation failed (attempt {retry_count}): {chain_error}")
                    if retry_count > max_retries:
                        raise chain_error
                    time.sleep(1.0 * retry_count)  # Exponential backoff
            
            # Extract the answer and source documents
            answer = response.get("result", "").strip()
            if not answer:
                answer = "I couldn't find a specific answer to your question in the provided documents."
                
            source_documents = response.get("source_documents", [])
            sources_by_doc: Dict[str, Set[Union[int, str]]] = {}

            # Process and group source metadata
            for doc in source_documents:
                try:
                    metadata = getattr(doc, "metadata", {}) or {}
                    source = metadata.get("source", "Unknown")
                    page = metadata.get("page", "")
                    if source not in sources_by_doc:
                        sources_by_doc[source] = set()
                    if page:
                        sources_by_doc[source].add(page)
                except Exception as metadata_err:
                    logging.warning(f"Error processing document metadata: {metadata_err}")

            # Format source citations if available
            source_info = ""
            if sources_by_doc:
                source_info = "\n\nSource Documents:\n"
                for source, pages in sources_by_doc.items():
                    if pages:
                        page_list = ", ".join([f"p.{page}" for page in sorted(pages)])
                        source_info += f"- {source} ({page_list})\n"
                    else:
                        source_info += f"- {source}\n"

            # Append a note if no specific source was found
            if not source_documents or not sources_by_doc:
                if "couldn't find" not in answer.lower() and "no information" not in answer.lower():
                    answer += ("\n\nNote: I couldn't find specific source documents that directly "
                               "answer your question.")
            
            # Combine answer with source information
            formatted_answer = f"{answer}{source_info}"
            
            # Cache the complete response
            processing_time = time.time() - start_time
            self.question_cache[cache_key] = {
                "raw_response": response,
                "answer": answer,
                "source_documents": source_documents,
                "formatted_answer": formatted_answer,
                "timestamp": time.time(),
                "processing_time": processing_time
            }
            
            self.stats["successful_responses"] += 1
            return formatted_answer

        except Exception as e:
            self.last_error = str(e)
            self.stats["failed_responses"] += 1
            logging.error(f"An error occurred while retrieving the answer: {e}")
            return ("Error: Something went wrong while processing your question. "
                    f"Please try again later. Technical details: {str(e)[:100]}...")

    def analyze_query(self, query: str) -> str:
        """
        Analyzes the query to determine key concepts and the expected answer type.
        """
        analysis_prompt = (
            "Analyze this question to help guide document retrieval:\n\n"
            "Question: {question}\n\n"
            "What type of information is the user looking for? (e.g., definition, procedure, comparison, etc.)\n"
            "What key concepts or terms should the retrieval system focus on?\n"
            "What would a complete answer to this question include?\n\n"
            "Provide a brief analysis:"
        )
    
        try:
            analysis_response = self.llm.invoke(analysis_prompt.format(question=query))
            return analysis_response.content
        except Exception as e:
            logging.warning(f"Error in query analysis: {e}")
            return "Unable to analyze query. Proceeding with standard retrieval."

def get_answer_from_pdfs(collection_names: List[str]) -> Optional[QAAgent]:
    """
    Connects to ChromaDB collections and initializes an enhanced QA agent.

    Args:
        collection_names (list): A list of ChromaDB collection names to use

    Returns:
        QAAgent instance if successful, otherwise None.
    """
    try:
        # Validate collection names
        if not collection_names or not isinstance(collection_names, list):
            logging.error("Invalid collection names provided")
            return None

        # Initialize collection metrics dictionary for diagnostics
        collection_metrics: Dict[str, Any] = {}

        # Connect to ChromaDB with retry logic
        chroma_client = chromadb.HttpClient(
                host=CHROMA_SERVER_HOST,
                port=CHROMA_SERVER_PORT,
                settings=chromadb.Settings(anonymized_telemetry=False)
            )

        # Initialize embeddings and language model
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small",
            dimensions=384,  # Must match existing collection dimensions
            retry_min_seconds=1,
            retry_max_seconds=20,
            max_retries=5
        )
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=OPENAI_API_KEY,
            max_retries=3,
            request_timeout=60  # Increased timeout for complex reasoning
        )

        # Define the enhanced prompt template with query analysis
        prompt_template = (
            "You are a highly accurate AI assistant that provides precise answers using ONLY the provided PDF documents.\n\n"
            "### Retrieved Document Context:\n{context}\n\n"
            "{question}\n\n"
            "### Instructions:\n"
            "- Answer ONLY based on the information in the provided document context\n"
            "- If the retrieved information is incomplete, clearly state what's missing\n"
            "- If the information is not found in the provided documents, say so directly and suggest checking other sources\n"
            "- Cite specific sources for your information (e.g., document name and page number)\n"
            "- Organize your answer logically with clear structure\n"
            "- Keep your answer concise and focused on the question\n"
            "- Do not hallucinate or make up information not present in the documents\n"
            "- If you need to make inferences, clearly indicate they are inferences\n\n"
            "### Answer:"
        )
        PROMPT = PromptTemplate.from_template(prompt_template)

        # Load collections and build retrievers
        if len(collection_names) == 1:
            final_retriever = _load_single_collection(
                collection_names[0], chroma_client, embeddings, llm
            )
            if final_retriever is None:
                return None
        else:
            final_retriever = _load_multiple_collections(
                collection_names, chroma_client, embeddings, llm
            )
            if final_retriever is None:
                return None


        # Initialize QA chain using the final retriever and enhanced prompt template
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Using simple stuffing strategy
            retriever=final_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT,}
        )

        # Create the QA Agent and add diagnostics
        qa_agent = QAAgent(qa_chain, llm)
        qa_agent.collection_metrics = collection_metrics

        return qa_agent

    except Exception as e:
        logging.error(f"An error occurred while initializing the QA agent: {e}")
        logging.error(f"Detailed error: {traceback.format_exc()}")
        return None


def _load_single_collection(collection_name: str, chroma_client, embeddings, llm) -> Optional[Any]:
    """
    Load a single Chroma collection and create a compressed hybrid retriever.
    """
    try:
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        retriever = HybridRetriever(
            vectorstore=vectorstore,
            llm=llm,
            search_kwargs={"k": 6}  # Retrieve more documents for better coverage
        )
        compressor = LLMChainExtractor.from_llm(llm)
        return ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=compressor
        )
    except Exception as e:
        logging.error(f"Error loading collection {collection_name}: {e}")
        return None


def _load_multiple_collections(collection_names: List[str], chroma_client, embeddings, llm) -> Optional[Any]:
    """
    Load multiple collections and build an ensemble retriever with dynamic weights.
    """
    valid_retrievers = []
    for collection_name in collection_names:
        try:
            vs = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embeddings
            )
            retriever = HybridRetriever(
                vectorstore=vs,
                llm=llm,
                search_kwargs={"k": 4}  # Smaller k for each collection
            )
            valid_retrievers.append(retriever)
            logging.info(f"Successfully loaded collection: {collection_name}")
        except Exception as e:
            logging.warning(f"Could not load collection {collection_name}: {e}")

    if not valid_retrievers:
        logging.error("No valid collections found")
        return None

    # If only one valid retriever exists, use it directly.
    if len(valid_retrievers) == 1:
        logging.info("Only one valid collection found, using it directly")
        base_retriever = valid_retrievers[0]
    else:
        weights = []
        for retriever in valid_retrievers:
            try:
                sample = retriever.get_relevant_documents("sample query")
                weights.append(0.5 + (len(sample) / 10))  # Base weight plus bonus for available content
            except Exception:
                weights.append(0.5)  # Default weight in case of error

        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(valid_retrievers)] * len(valid_retrievers)

        base_retriever = EnsembleRetriever(
            retrievers=valid_retrievers,
            weights=weights
        )
        logging.info(f"Created weighted ensemble retriever with {len(valid_retrievers)} collections")

    # Add contextual compression to filter retrieved documents
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )
