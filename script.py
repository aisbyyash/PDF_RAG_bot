from langchain_chroma import Chroma
import chromadb
import streamlit as st


CHROMA_SERVER_HOST = st.secrets.chroma.server
CHROMA_SERVER_PORT = int(st.secrets.chroma.port)

def get_collection_name(class_name, role):
    """Generate collection name based on class and role"""
    if class_name == "General":
        return f"general_{role.lower()}"
    else:
        class_number = class_name.split()[1]
        return f"class_{class_number}_{role.lower()}"

def generate_all_collection_names():
    """Generate all possible collection names"""
    # All possible classes
    classes = ["General"] + [f"Class {i}" for i in range(1, 11)]
    
    # All possible roles
    roles = ["teacher", "student", "principal"]
    
    # Generate all combinations
    collections = []
    for class_name in classes:
        for role in roles:
            collection_name = get_collection_name(class_name, role)
            collections.append(collection_name)
    
    return collections

def main():
    """Main function"""
    chroma_client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_PORT)
    collections = chroma_client.list_collections()
    print(collections)
    for collection in collections:
        chroma_client.delete_collection(collection)
    print("All collections deleted.")
    collections = chroma_client.list_collections()
    print(collections)

if __name__ == "__main__":
    main()