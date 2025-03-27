# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import json
import pandas as pd
import logging
import time
from datetime import datetime
from main import process_all_pdfs, get_answer_from_pdfs  # Import functions from main.py

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths to data files
PASSWORDS_FILE = "data/passwords.json"
LOGS_FILE = "data/access_logs.csv"
CLASS_FOLDERS = "classes"

# Streamlit UI Configuration
st.set_page_config(page_title="School PDF Q&A System", layout="wide")

# Helper function to generate consistent collection names
def get_collection_name(class_name, role):
    if class_name == "General":
        return f"general_{role.lower()}"
    else:
        class_number = class_name.split()[1]
        return f"class_{class_number}_{role.lower()}"

# Helper function to get all PDFs in a folder
def get_pdfs_in_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

# Helper to get PDF file size and modification date
def get_file_info(file_path):
    if not os.path.exists(file_path):
        return {"size": "N/A", "modified": "N/A"}
    
    size_bytes = os.path.getsize(file_path)
    if size_bytes < 1024:
        size_str = f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes/1024:.1f} KB"
    else:
        size_str = f"{size_bytes/(1024*1024):.1f} MB"
    
    mod_time = os.path.getmtime(file_path)
    mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
    
    return {"size": size_str, "modified": mod_date}

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Admin", "User"])

# ----------------------------------------
# Admin Page
# ----------------------------------------
if page == "Admin":
    st.title("📊 Admin Dashboard - School PDF Q&A System")

    # Admin Login System
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = True

    # ✅ Admin Successfully Logged In - Show Dashboard Features
    st.sidebar.header("Admin Actions")
    admin_action = st.sidebar.selectbox(
        "Select Action",
        ["Dashboard Overview", "Manage PDFs", "Delete PDFs", "Access Logs"]
    )
    
    # Initialize class selection in session state if not present
    if "selected_class" not in st.session_state:
        st.session_state.selected_class = "General"
    
    # Class selector in sidebar that persists across all admin sections
    st.sidebar.subheader("Select Class")
    selected_class = st.sidebar.selectbox(
        "Choose a class:",
        ["General"] + [f"Class {i}" for i in range(1, 11)],
        index=(["General"] + [f"Class {i}" for i in range(1, 11)]).index(st.session_state.selected_class)
    )
    
    # Update session state
    st.session_state.selected_class = selected_class
    
    # Define paths for Teacher/Student folders
    class_folder = os.path.join(CLASS_FOLDERS, f"class_{selected_class.split()[1]}") if "Class" in selected_class else os.path.join(CLASS_FOLDERS, "general")
    teacher_folder = os.path.join(class_folder, "Teacher")
    student_folder = os.path.join(class_folder, "Student")
    
    # Ensure folders exist
    os.makedirs(teacher_folder, exist_ok=True)
    os.makedirs(student_folder, exist_ok=True)
    
    # Get PDF lists
    teacher_pdfs = get_pdfs_in_folder(teacher_folder)
    student_pdfs = get_pdfs_in_folder(student_folder)
    
    # -----------------------------
    # DASHBOARD OVERVIEW SECTION
    # -----------------------------
    if admin_action == "Dashboard Overview":
        st.header("📊 Dashboard Overview")
        
        # Summary Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_classes = 11  # 10 classes + General
            st.metric("Total Classes", total_classes)
        
        with col2:
            # Count total PDFs across all classes
            total_pdfs = 0
            for class_name in ["General"] + [f"Class {i}" for i in range(1, 11)]:
                class_path = os.path.join(CLASS_FOLDERS, f"class_{class_name.split()[1]}") if "Class" in class_name else os.path.join(CLASS_FOLDERS, "general")
                teacher_path = os.path.join(class_path, "Teacher")
                student_path = os.path.join(class_path, "Student")
                
                if os.path.exists(teacher_path):
                    total_pdfs += len(get_pdfs_in_folder(teacher_path))
                if os.path.exists(student_path):
                    total_pdfs += len(get_pdfs_in_folder(student_path))
            
            st.metric("Total PDFs", total_pdfs)
        
        with col3:
            # Current class PDFs
            current_class_pdfs = len(teacher_pdfs) + len(student_pdfs)
            st.metric(f"{selected_class} PDFs", current_class_pdfs)

        # Add this after the Dashboard Overview metrics and before the file listing

        # Handle PDF viewer from dashboard
        if hasattr(st.session_state, 'selected_pdf') and hasattr(st.session_state, 'selected_folder'):
            selected_pdf = st.session_state.selected_pdf
            selected_folder = st.session_state.selected_folder
            
            # Determine the folder path based on the selected folder
            pdf_folder_path = teacher_folder if selected_folder == "Teacher" else student_folder
            pdf_path = os.path.join(pdf_folder_path, selected_pdf)
            file_info = get_file_info(pdf_path)
            
            # Use a cleaner title layout with back button directly in the header
            col1, col2 = st.columns([5, 1])
            with col1:
                st.title(f"📄 {selected_pdf}")
                st.caption(f"{selected_folder} Materials • {selected_class}")
            with col2:
                if st.button("Back to Files", type="primary", use_container_width=True):
                    del st.session_state.selected_pdf
                    del st.session_state.selected_folder
                    st.rerun()
            
            # Horizontal line for visual separation
            st.markdown("---")
            
            # Create tabs for different views of the PDF
            tabs = st.tabs(["📋 Overview", "📝 Content Preview", "🔍 Details"])
            
            # Tab 1: Overview - Quick summary and main actions
            with tabs[0]:
                # Two columns - left for info, right for actions
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Clean, card-style info display
                    st.markdown("""
                    <style>
                    .info-card {
                        background-color: #f8f9fa;
                        border-radius: 0.5rem;
                        padding: 1.5rem;
                        margin-bottom: 1rem;
                        border-left: 5px solid #4361ee;
                    }
                    .info-item {
                        margin-bottom: 0.8rem;
                    }
                    .info-label {
                        font-weight: 600;
                        color: #495057;
                    }
                    .info-value {
                        color: #212529;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-item">
                            <span class="info-label">File:</span>
                            <span class="info-value">{selected_pdf}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Collection:</span>
                            <span class="info-value">{get_collection_name(selected_class, selected_folder)}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Size:</span>
                            <span class="info-value">{file_info['size']}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Last Modified:</span>
                            <span class="info-value">{file_info['modified']}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Location:</span>
                            <span class="info-value">{pdf_folder_path}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Action buttons in a visually appealing card
                    st.markdown("""
                    <style>
                    .action-card {
                        background-color: #f8f9fa;
                        border-radius: 0.5rem;
                        padding: 1.5rem;
                        margin-bottom: 1rem;
                        border-left: 5px solid #4cc9f0;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="action-card">
                        <h4>Available Actions</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Read file for download button
                    with open(pdf_path, "rb") as file:
                        pdf_contents = file.read()
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_contents,
                            file_name=selected_pdf,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    st.button("🔄 Reprocess PDF", 
                            use_container_width=True,
                            on_click=lambda: st.session_state.update({'reprocess_pdf': True}))
                    
                    st.button("🗑️ Delete PDF", 
                            use_container_width=True,
                            type="secondary",
                            on_click=lambda: st.session_state.update({
                                'delete_pdf': selected_pdf,
                                'delete_folder': selected_folder,
                                'selected_pdf': None,
                                'selected_folder': None
                            }))
                        
                    # Handle reprocessing if button was clicked
                    if hasattr(st.session_state, 'reprocess_pdf') and st.session_state.reprocess_pdf:
                        with st.spinner(f"Reprocessing {selected_pdf}..."):
                            role = "teacher" if selected_folder == "Teacher" else "student"
                            collection_name = get_collection_name(selected_class, role)
                            # Reprocess just this folder with this single PDF
                            process_result = process_all_pdfs(pdf_folder_path, collection_name)
                            
                            if process_result:
                                st.success(f"✅ Successfully reprocessed {selected_pdf}")
                            else:
                                st.error(f"❌ Failed to reprocess {selected_pdf}")
                            
                            # Clear the flag
                            del st.session_state.reprocess_pdf
            
            # Tab 2: Content Preview
            with tabs[1]:
                try:
                    from PyPDF2 import PdfReader
                    
                    # Create a reader object
                    reader = PdfReader(pdf_path)
                    
                    # Get the number of pages
                    num_pages = len(reader.pages)
                    
                    if num_pages > 0:
                        # Let user select which page to view if multiple pages
                        if num_pages > 1:
                            page_num = st.select_slider(
                                "Select page to view:",
                                options=list(range(1, num_pages + 1)),
                                value=1
                            )
                        else:
                            page_num = 1
                        
                        # Display page info
                        st.caption(f"Viewing page {page_num} of {num_pages}")
                        
                        # Get page content
                        page = reader.pages[page_num - 1]  # Adjust for 0-based index
                        page_text = page.extract_text()
                        
                        # Create a cleaner text display
                        st.markdown("""
                        <style>
                        .pdf-content {
                            background-color: white;
                            border: 1px solid #dee2e6;
                            border-radius: 0.5rem;
                            padding: 1.5rem;
                            font-family: monospace;
                            white-space: pre-wrap;
                            overflow-x: auto;
                            line-height: 1.5;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Fix for the problematic line with backslash in f-string
                        # First create the HTML content variable separately
                        html_content = page_text.replace('\n', '<br>')

                        # Then use it in the f-string
                        st.markdown(f"""
                        <div class="pdf-content">
                        {html_content}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Navigation buttons for multi-page PDFs
                        if num_pages > 1:
                            col1, col2 = st.columns(2)
                            with col1:
                                if page_num > 1:
                                    if st.button("◀️ Previous Page", use_container_width=True):
                                        st.session_state.current_page = page_num - 1
                                        st.rerun()
                            with col2:
                                if page_num < num_pages:
                                    if st.button("Next Page ▶️", use_container_width=True):
                                        st.session_state.current_page = page_num + 1
                                        st.rerun()
                    else:
                        st.info("This PDF does not contain any pages.")
                        
                except Exception as e:
                    st.warning(f"Unable to preview PDF content.")
                    st.info("You can download the PDF to view it in your preferred PDF reader.")
                    st.expander("Technical Error Details").write(str(e))
            
            # Tab 3: Details - Technical information about the PDF
            with tabs[2]:
                try:
                    # Display more detailed info about the PDF
                    import fitz  # PyMuPDF
                    
                    st.subheader("PDF Technical Information")
                    
                    try:
                        # Try to open with PyMuPDF for more detailed info
                        doc = fitz.open(pdf_path)
                        
                        # Two columns for metadata
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Document Properties")
                            properties = {
                                "Page Count": doc.page_count,
                                "Form Fields": len(doc.get_form_text_fields()),
                                "File Size": file_info['size'],
                                "Format": f"PDF {doc.pdf_version}",
                                "Has Annotations": "Yes" if doc.has_annots() else "No",
                            }
                            
                            for prop, value in properties.items():
                                st.markdown(f"**{prop}:** {value}")
                        
                        with col2:
                            st.markdown("### Metadata")
                            metadata = doc.metadata
                            if metadata:
                                for key, value in metadata.items():
                                    if value and str(value).strip():
                                        st.markdown(f"**{key}:** {value}")
                            else:
                                st.info("No metadata available for this PDF.")
                        
                        # Close the document
                        doc.close()
                        
                    except ImportError:
                        st.info("Install PyMuPDF for more detailed PDF information.")
                        st.code("pip install pymupdf", language="bash")
                    except Exception as e:
                        st.warning("Cannot extract detailed PDF information.")
                        st.expander("Error details").write(str(e))
                        
                except ImportError:
                    st.info("For detailed PDF analysis, install additional libraries:")
                    st.code("pip install pymupdf", language="bash")
            
            # Stop further rendering of dashboard content
            st.stop()
        
        # Class File Explorer
        st.subheader(f"📁 {selected_class} File Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Teacher Materials")
            if teacher_pdfs:
                # Create a visual card for each PDF
                for pdf in teacher_pdfs:
                    pdf_path = os.path.join(teacher_folder, pdf)
                    file_info = get_file_info(pdf_path)
                    
                    # Create a card-like container with border
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;">
                        <h4 style="margin-top:0">📘 {pdf}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(f"**Size:** {file_info['size']}")
                    st.write(f"**Last Modified:** {file_info['modified']}")
                    
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"View Info", key=f"view_{pdf}_teacher"):
                            st.session_state.selected_pdf = pdf
                            st.session_state.selected_folder = "Teacher"
                            st.rerun()
                    
                    with button_col2:
                        if st.button(f"Delete", key=f"delete_{pdf}_teacher"):
                            st.session_state.delete_pdf = pdf
                            st.session_state.delete_folder = "Teacher"
                            st.rerun()
                    
                    # Add a separator between files
                    st.markdown("---")
            else:
                st.info("No teacher PDFs uploaded yet.")
        
        with col2:
            st.subheader("Student Materials")
            if student_pdfs:
                # Create a visual card for each PDF
                for pdf in student_pdfs:
                    pdf_path = os.path.join(student_folder, pdf)
                    file_info = get_file_info(pdf_path)
                    
                    # Create a card-like container with border
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;">
                        <h4 style="margin-top:0">📗 {pdf}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(f"**Size:** {file_info['size']}")
                    st.write(f"**Last Modified:** {file_info['modified']}")
                    
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"View Info", key=f"view_{pdf}_student"):
                            st.session_state.selected_pdf = pdf
                            st.session_state.selected_folder = "Student"
                            st.rerun()
                    
                    with button_col2:
                        if st.button(f"Delete", key=f"delete_{pdf}_student"):
                            st.session_state.delete_pdf = pdf
                            st.session_state.delete_folder = "Student"
                            st.rerun()
                    
                    # Add a separator between files
                    st.markdown("---")
            else:
                st.info("No student PDFs uploaded yet.")
        
        # Handle file deletion from dashboard
        if hasattr(st.session_state, 'delete_pdf') and hasattr(st.session_state, 'delete_folder'):
            delete_pdf = st.session_state.delete_pdf
            delete_folder = st.session_state.delete_folder
            
            delete_folder_path = teacher_folder if delete_folder == "Teacher" else student_folder
            delete_path = os.path.join(delete_folder_path, delete_pdf)
            
            st.warning(f"Are you sure you want to delete {delete_pdf} from {delete_folder}?")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("Confirm Delete", type="primary"):
                    try:
                        os.remove(delete_path)
                        st.success(f"Successfully deleted {delete_pdf}")
                        # Clear the deletion state
                        del st.session_state.delete_pdf
                        del st.session_state.delete_folder
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {e}")
            
            with col2:
                if st.button("Cancel"):
                    # Clear the deletion state
                    del st.session_state.delete_pdf
                    del st.session_state.delete_folder
                    st.rerun()
    
    # -----------------------------
    # MANAGE PDFs SECTION
    # -----------------------------
    elif admin_action == "Manage PDFs":
        st.header("📂 Upload PDFs")
        
        # Force refresh of PDF lists when needed
        if "refresh_pdf_lists" not in st.session_state:
            st.session_state.refresh_pdf_lists = True
            
        # Only scan directories when needed to avoid performance issues
        if st.session_state.refresh_pdf_lists:
            teacher_pdfs = get_pdfs_in_folder(teacher_folder)
            student_pdfs = get_pdfs_in_folder(student_folder)
            st.session_state.refresh_pdf_lists = False
        
        # Display current content summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📘 Teacher Materials ({len(teacher_pdfs)} files)")
            if teacher_pdfs:
                with st.expander("View Teacher PDFs", expanded=True):
                    for pdf in teacher_pdfs:
                        pdf_path = os.path.join(teacher_folder, pdf)
                        file_info = get_file_info(pdf_path)
                        st.write(f"• **{pdf}** - {file_info['size']} - {file_info['modified']}")
            else:
                st.info("No teacher PDFs uploaded for this class.")
        
        with col2:
            st.subheader(f"📗 Student Materials ({len(student_pdfs)} files)")
            if student_pdfs:
                with st.expander("View Student PDFs", expanded=True):
                    for pdf in student_pdfs:
                        pdf_path = os.path.join(student_folder, pdf)
                        file_info = get_file_info(pdf_path)
                        st.write(f"• **{pdf}** - {file_info['size']} - {file_info['modified']}")
            else:
                st.info("No student PDFs uploaded for this class.")
        
        # Upload Section
        st.subheader("📤 Upload New PDFs")
        upload_destination = st.radio("Upload PDFs to:", ["Teacher", "Student", "Both"], horizontal=True)
        
        # Visual separator
        st.markdown("---")
        
        # Initialize session state variables for uploader
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = f"pdf_uploader_{int(time.time())}"
        if "processing_files" not in st.session_state:
            st.session_state.processing_files = False
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
        if "upload_status_messages" not in st.session_state:
            st.session_state.upload_status_messages = []
        
        # Reset the uploader completely
        def reset_uploader():
            st.session_state.uploader_key = f"pdf_uploader_{int(time.time())}"
            st.session_state.processing_files = False
            st.session_state.processed_files = set()
            st.session_state.upload_status_messages = []
            st.session_state.refresh_pdf_lists = True
        
        # Clear error messages when needed
        status_container = st.container()
        
        # File uploader with support for multiple files
        uploaded_pdfs = st.file_uploader("Choose PDFs to Upload", type=["pdf"], accept_multiple_files=True, key=st.session_state.uploader_key)
        
        # Skip previously processed files
        new_pdfs = [pdf for pdf in uploaded_pdfs if pdf.name not in st.session_state.processed_files]
        
        # Detect when new files are added and clear previous error messages
        if new_pdfs and not st.session_state.processing_files:
            st.session_state.upload_status_messages = []
            st.session_state.processing_files = True
        
        # Process new files if any are available
        if new_pdfs and st.session_state.processing_files:
            # Placeholders for progress and status
            overall_progress = st.progress(0, text="Preparing to upload files...")
            
            # Track success and failures
            successful_files = []
            failed_files = []
            skipped_files = []
            
            # Process each PDF
            for file_index, uploaded_pdf in enumerate(new_pdfs):
                # Update overall progress
                overall_file_progress = (file_index / len(new_pdfs)) * 100
                overall_progress.progress(int(overall_file_progress), text=f"Processing file {file_index+1} of {len(new_pdfs)}: {uploaded_pdf.name}")
                
                # Create a status container for this file
                file_status = st.empty()
                file_status.info(f"⏳ Processing {uploaded_pdf.name}...")
                
                # Read the file content once
                pdf_content = uploaded_pdf.read()
                
                destination_folders = []
                if upload_destination in ["Teacher", "Both"]:
                    destination_folders.append(teacher_folder)
                if upload_destination in ["Student", "Both"]:
                    destination_folders.append(student_folder)
                
                # Track temporary file paths for cleanup in case of failure
                temp_files = []
                file_success = True
                
                try:
                    for folder_index, folder in enumerate(destination_folders):
                        pdf_path = os.path.join(folder, uploaded_pdf.name)
                        role = "teacher" if folder == teacher_folder else "student"
                        
                        # Check if file already exists
                        if os.path.exists(pdf_path):
                            file_status.warning(f"⚠️ The file '{uploaded_pdf.name}' already exists in {role} folder. Skipping.")
                            if uploaded_pdf.name not in skipped_files:
                                skipped_files.append(uploaded_pdf.name)
                            continue
                        
                        # Write the content to the file
                        with open(pdf_path, "wb") as f:
                            f.write(pdf_content)
                        
                        # Add to list of temporary files for cleanup in case of failure
                        temp_files.append(pdf_path)
                        
                        # Process PDFs and store in ChromaDB collections
                        collection_name = get_collection_name(selected_class, role)
                        file_status.info(f"⏳ Indexing {uploaded_pdf.name} for {role}... (This may take a while)")
                        
                        process_result = process_all_pdfs(folder, collection_name)
                        
                        # If processing failed, raise an exception
                        if not process_result:
                            raise Exception(f"Failed to process {uploaded_pdf.name} for {role}")
                    
                    # All destinations processed successfully
                    if len(temp_files) > 0:  # At least one file was uploaded (not skipped)
                        file_status.success(f"✅ Successfully processed {uploaded_pdf.name}")
                        successful_files.append(uploaded_pdf.name)
                    elif uploaded_pdf.name not in skipped_files:
                        file_status.error(f"❌ No destinations were valid for {uploaded_pdf.name}")
                        failed_files.append(uploaded_pdf.name)
                        file_success = False
                    
                except Exception as e:
                    # Cleanup temporary files on failure
                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except Exception as cleanup_error:
                                st.error(f"Error removing temporary file {temp_file}: {cleanup_error}")
                    
                    # Show error message
                    error_message = str(e)
                    file_status.error(f"❌ Error processing {uploaded_pdf.name}: {error_message}")
                    failed_files.append(uploaded_pdf.name)
                    file_success = False
                
                # Mark file as processed regardless of success/failure
                st.session_state.processed_files.add(uploaded_pdf.name)
            
            # Update overall status
            overall_progress.progress(100, text="Processing complete")
            
            # Add summary to status messages
            if successful_files:
                st.session_state.upload_status_messages.append({
                    "type": "success",
                    "message": f"✅ Successfully processed {len(successful_files)} file(s): {', '.join(successful_files)}"
                })
            
            if failed_files:
                st.session_state.upload_status_messages.append({
                    "type": "error",
                    "message": f"❌ Failed to process {len(failed_files)} file(s): {', '.join(failed_files)}"
                })
            
            if skipped_files:
                st.session_state.upload_status_messages.append({
                    "type": "warning",
                    "message": f"⚠️ Skipped {len(skipped_files)} existing file(s): {', '.join(skipped_files)}"
                })
            
            # Reset processing flag
            st.session_state.processing_files = False
            
            # Mark that we need to refresh the PDF lists
            st.session_state.refresh_pdf_lists = True
            
            # Force rerun to update the UI
            st.rerun()
        
        # Display status messages
        with status_container:
            for msg in st.session_state.upload_status_messages:
                if msg["type"] == "success":
                    st.success(msg["message"])
                elif msg["type"] == "error":
                    st.error(msg["message"])
                elif msg["type"] == "warning":
                    st.warning(msg["message"])
                elif msg["type"] == "info":
                    st.info(msg["message"])
        
        # Add a button to reset the uploader if there are any processed files
        if st.session_state.processed_files:
            if st.button("Reset Uploader"):
                reset_uploader()
                st.rerun()
                
        # Force rerun to update the list of PDFs if they were changed
        if st.session_state.refresh_pdf_lists:
            st.rerun()
        
    # -----------------------------
    # DELETE PDFs SECTION
    # -----------------------------
    elif admin_action == "Delete PDFs":
        st.header("❌ Delete PDFs")
        
        # Create tabs for Teacher and Student materials
        tab1, tab2 = st.tabs(["Teacher Materials", "Student Materials"])
        
        with tab1:
            st.subheader(f"📘 {selected_class} - Teacher PDFs")
            delete_folder_path = teacher_folder
            pdf_files = get_pdfs_in_folder(delete_folder_path)
            
            if pdf_files:
                # Create a visual grid of files
                cols = st.columns(3)
                for i, pdf in enumerate(pdf_files):
                    pdf_path = os.path.join(delete_folder_path, pdf)
                    file_info = get_file_info(pdf_path)
                    
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style="border:1px solid #cccccc; padding:10px; border-radius:5px; margin-bottom:10px">
                            <h4>📘 {pdf}</h4>
                            <p><b>Size:</b> {file_info['size']}</p>
                            <p><b>Modified:</b> {file_info['modified']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"Delete {pdf}", key=f"del_teacher_{pdf}"):
                            st.session_state.confirm_delete = True
                            st.session_state.delete_file = pdf
                            st.session_state.delete_role = "Teacher"
                            st.rerun()
            else:
                st.info("No Teacher PDFs found for this class.")
        
        with tab2:
            st.subheader(f"📗 {selected_class} - Student PDFs")
            delete_folder_path = student_folder
            pdf_files = get_pdfs_in_folder(delete_folder_path)
            
            if pdf_files:
                # Create a visual grid of files
                cols = st.columns(3)
                for i, pdf in enumerate(pdf_files):
                    pdf_path = os.path.join(delete_folder_path, pdf)
                    file_info = get_file_info(pdf_path)
                    
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style="border:1px solid #cccccc; padding:10px; border-radius:5px; margin-bottom:10px">
                            <h4>📗 {pdf}</h4>
                            <p><b>Size:</b> {file_info['size']}</p>
                            <p><b>Modified:</b> {file_info['modified']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"Delete {pdf}", key=f"del_student_{pdf}"):
                            st.session_state.confirm_delete = True
                            st.session_state.delete_file = pdf
                            st.session_state.delete_role = "Student"
                            st.rerun()
            else:
                st.info("No Student PDFs found for this class.")
        
        # Handle confirmation dialog
        if hasattr(st.session_state, 'confirm_delete') and st.session_state.confirm_delete:
            delete_file = st.session_state.delete_file
            delete_role = st.session_state.delete_role
            
            delete_path = os.path.join(teacher_folder if delete_role == "Teacher" else student_folder, delete_file)
            
            st.warning(f"⚠️ Are you sure you want to delete **{delete_file}** from the **{delete_role}** collection?")
            st.markdown("This action cannot be undone.")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Yes, Delete", type="primary"):
                    try:
                        os.remove(delete_path)
                        st.success(f"Successfully deleted {delete_file}")
                        
                        # Clear the confirmation state
                        del st.session_state.confirm_delete
                        del st.session_state.delete_file
                        del st.session_state.delete_role
                        
                        # Small delay to show the success message
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {e}")
            
            with col2:
                if st.button("❌ Cancel"):
                    # Clear the confirmation state
                    del st.session_state.confirm_delete
                    del st.session_state.delete_file
                    del st.session_state.delete_role
                    st.rerun()
    
    # -----------------------------
    # ACCESS LOGS SECTION
    # -----------------------------
    elif admin_action == "Access Logs":
        st.header("📜 Access Logs")
        
        if os.path.exists(LOGS_FILE):
            try:
                logs_df = pd.read_csv(LOGS_FILE, on_bad_lines='skip')
                
                # Add filtering options
                st.subheader("Filter Logs")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'role' in logs_df.columns:
                        roles = ['All'] + list(logs_df['role'].unique())
                        selected_role = st.selectbox('Filter by Role:', roles)
                
                with col2:
                    if 'class' in logs_df.columns:
                        classes = ['All'] + list(logs_df['class'].unique())
                        selected_class_filter = st.selectbox('Filter by Class:', classes)
                
                # Apply filters
                filtered_df = logs_df.copy()
                if selected_role != 'All' and 'role' in logs_df.columns:
                    filtered_df = filtered_df[filtered_df['role'] == selected_role]
                
                if selected_class_filter != 'All' and 'class' in logs_df.columns:
                    filtered_df = filtered_df[filtered_df['class'] == selected_class_filter]
                
                # Display filtered data
                st.dataframe(filtered_df, use_container_width=True)
                
                # Add export options
                if st.button("Export Filtered Logs"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"access_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
            except pd.errors.ParserError:
                st.error("⚠️ Error reading access logs. The file may have formatting issues.")
            except pd.errors.EmptyDataError:
                st.info("No access logs found. The log file will be created when users access the system.")
        else:
            st.info("No access logs found. The log file will be created when users access the system.")

elif page == "User":
    # Apply better styling for the user interface
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #1E3A8A;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .login-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .login-button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .user-info {
        background-color: #F1F1F1;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Page title with better styling
    st.markdown('<h1 class="main-header">School PDF Q&A System</h1>', unsafe_allow_html=True)

    # User is automatically authenticated - no password check needed
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = True
        
    # Let user select their role and class without authentication
    if "current_role" not in st.session_state or "current_class" not in st.session_state:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Select Access Level</h2>', unsafe_allow_html=True)
        
        # Use columns for a better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p style="font-weight: 500;">Select Your Role:</p>', unsafe_allow_html=True)
            role = st.selectbox(
                "Choose your role in the school system",
                ["Principal", "Teacher", "Student"],
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown('<p style="font-weight: 500;">Select Your Class:</p>', unsafe_allow_html=True)
            class_options = [f"Class {i}" for i in range(1, 11)] + ["General"]
            selected_class = st.selectbox(
                "Choose a class or general access",
                class_options,
                label_visibility="collapsed"
            )
        
        # Store user selections in session state
        if st.button("Continue", use_container_width=True, type="primary"):
            st.session_state.current_role = role
            st.session_state.current_class = selected_class
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Skip the rest of the code if role and class are not set yet
        st.stop()
    
    # Display user info with option to change selections
    st.markdown('<div class="user-info">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Using system as:** {st.session_state.current_role} | **Class:** {st.session_state.current_class}")
    with col2:
        if st.button("Change Selection", type="primary"):
            # Clear role and class selections to allow reselection
            for key in ['current_role', 'current_class', 'qa_agent', 'chat_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 4: Start Chat Interface
    st.markdown('<h2 class="section-header">Chat with Your PDFs</h2>', unsafe_allow_html=True)
    
    # Reset QA agent if role or class changes
    if "current_role" not in st.session_state or st.session_state.current_role != st.session_state.get('current_role') or \
       "current_class" not in st.session_state or st.session_state.current_class != st.session_state.get('current_class'):
        if "qa_agent" in st.session_state:
            del st.session_state.qa_agent
        st.session_state.current_role = st.session_state.get('current_role')
        st.session_state.current_class = st.session_state.get('current_class')
    
    if "qa_agent" not in st.session_state:
        try:
            # Initialize collections list based on role
            collections_to_search = []
            
            # Principal-specific access setup
            if st.session_state.current_role == "Principal":
                with st.expander("🔑 Principal Access Options", expanded=False):
                    st.write("As a Principal, you have access to all school materials.")
                    
                    # UI for Principal to configure which collections to search
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Class-specific options
                        st.subheader("Class Materials Access")
                        search_selected_teacher = st.checkbox("Access Teacher materials for selected class", value=True)
                        search_selected_student = st.checkbox("Access Student materials for selected class", value=True)
                    
                    with col2:
                        # General options
                        st.subheader("General Materials Access")
                        search_general_teacher = st.checkbox("Access General Teacher materials", value=True)
                        search_general_student = st.checkbox("Access General Student materials", value=True)
                    
                    # Option to include all classes
                    st.subheader("All School Materials")
                    include_all_classes = st.checkbox("Include materials from ALL classes", value=False)
                    
                    if include_all_classes:
                        with st.expander("Select specific class materials to include"):
                            # Allow selecting specific classes and roles to include
                            for class_num in range(1, 11):
                                class_name = f"Class {class_num}"
                                if class_name != st.session_state.current_class:  # Skip already selected class
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        include_teacher = st.checkbox(f"{class_name} Teacher materials", value=True)
                                        if include_teacher:
                                            collections_to_search.append(get_collection_name(class_name, "Teacher"))
                                    with col2:
                                        include_student = st.checkbox(f"{class_name} Student materials", value=True)
                                        if include_student:
                                            collections_to_search.append(get_collection_name(class_name, "Student"))
                
                # Add selected class collections based on checkboxes
                if st.session_state.current_class != "General":
                    if search_selected_teacher:
                        collections_to_search.append(get_collection_name(st.session_state.current_class, "Teacher"))
                    if search_selected_student:
                        collections_to_search.append(get_collection_name(st.session_state.current_class, "Student"))
                
                # Add general collections based on checkboxes
                if search_general_teacher:
                    collections_to_search.append(get_collection_name("General", "Teacher"))
                if search_general_student:
                    collections_to_search.append(get_collection_name("General", "Student"))
                
                # Display selected collections in a cleaner way
                if collections_to_search:
                    st.success(f"Searching across {len(collections_to_search)} collections")
                    with st.expander("View collection details"):
                        st.code(", ".join(collections_to_search))
                
            else:
                # Regular user (Teacher or Student)
                collections_to_search.append(get_collection_name(st.session_state.current_class, st.session_state.current_role))
                
                # Option to also search general collection
                if st.session_state.current_class != "General":
                    search_general = st.checkbox("Also search General collection", value=True)
                    if search_general:
                        general_collection_name = get_collection_name("General", st.session_state.current_role)
                        collections_to_search.append(general_collection_name)
                        st.info(f"Searching in both {st.session_state.current_class} and General collections")
            
            # Initialize the QA agent with selected collections
            if collections_to_search:
                with st.spinner("Loading materials. This may take a moment..."):
                    st.session_state.qa_agent = get_answer_from_pdfs(collections_to_search)
                    
                    if st.session_state.qa_agent is None:
                        st.error("Error: No valid data found in the selected collections.")
                        st.stop()
            else:
                st.error("No collections selected for search. Please select at least one option.")
                st.stop()
                
        except Exception as e:
            st.error(f"An error occurred while initializing the QA agent: {e}")
            st.stop()

    # WhatsApp-like chat container with improved CSS
    # Add this CSS before the chat container
    st.markdown("""
    <style>
    .chat-container {
        background-color: #E5DDD5;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        max-height: 60vh;
        overflow-y: auto;
    }
    .message-container {
        display: flex;
        margin-bottom: 10px;
        width: 100%;
    }
    .message-container.user {
        justify-content: flex-end;
    }
    .message-container.bot {
        justify-content: flex-start;
    }
    .message-bubble {
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 75%;
        word-wrap: break-word;
        position: relative;
    }
    .user-message {
        background-color: #DCF8C6;
        border-top-right-radius: 0;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
    }
    .bot-message {
        background-color: #FFFFFF;
        border-top-left-radius: 0;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
    }
    .message-sender {
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 4px;
        color: #1E3A8A;
    }
    .message-content {
        font-size: 0.95rem;
        line-height: 1.4;
        color: #202020;
    }
    .chat-form {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a container for the chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    chat_container = st.container()

    with chat_container:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Show welcome message if chat is empty
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p style="font-size: 16px; margin-bottom: 8px;">👋 Welcome to the PDF Q&A System!</p>
                    <p style="font-size: 14px;">Ask questions about your course materials below.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Display chat messages with improved styling
        for entry in st.session_state.get("chat_history", []):
            if entry["type"] == "user":
                st.markdown(
                    f"""
                    <div class="message-container user">
                        <div class="message-bubble user-message">
                            <div class="message-sender">You</div>
                            <div class="message-content">{entry['content']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif entry["type"] == "bot":
                st.markdown(
                    f"""
                    <div class="message-container bot">
                        <div class="message-bubble bot-message">
                            <div class="message-sender">Assistant</div>
                            <div class="message-content">{entry['content']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Close the chat container
    st.markdown('</div>', unsafe_allow_html=True)

    # Input form for user questions with better styling
    with st.form("question_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "Ask a question about the PDFs:", 
                key="question",
                placeholder="Type your question here..."
            )
        with col2:
            submit_button = st.form_submit_button("Send", use_container_width=True)

    if submit_button:
        if not question.strip():
            st.warning("⚠️ Please enter a valid question before submitting.")
        else:
            # Add the user question to the chat history
            st.session_state.chat_history.append({"type": "user", "content": question})

            with st.spinner("Generating response..."):
                try:
                    # Call the ask_question method
                    response = st.session_state.qa_agent.ask_question(question)
                except Exception as e:
                    logging.error(f"Error while generating response: {e}")
                    response = "Error: Unable to process your question. Please try again later."

            # Add the bot's response to the chat history
            st.session_state.chat_history.append({"type": "bot", "content": response})
            st.rerun()  # Rerun to update chat history and UI