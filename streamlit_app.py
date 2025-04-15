import streamlit as st
import requests
import time
import os
from typing import List, Dict, Any
import json

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG - Interactive Q&A with Documents",
    page_icon="🔍",
    layout="wide"
)

# Set API URL - change if deployed elsewhere
API_URL = "http://localhost:8000"  # Assumes FastAPI running on same machine, port 8000

# Add a title and description
st.title("🔍 Multimodal RAG System")
st.subheader("Interactive Q&A with Files (PDF, Images)")

st.markdown("""
This app allows you to upload files (PDFs, images) and ask questions about their content. 
The system uses a multimodal retrieval-augmented generation (RAG) approach to provide relevant answers.
""")

# Initialize session state for tracking processed documents
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None
if 'answer_history' not in st.session_state:
    st.session_state.answer_history = []

# Sidebar for API key and configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input (password field for security)
    openai_api_key = st.text_input("OpenAI API Key", type="password", 
                                   help="Your OpenAI API key is required for processing")
    
    # Advanced options (expandable section)
    with st.expander("Advanced Options"):
        model_option = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o"],
            index=0,
            help="Select the model to use for answering questions"
        )
        
    st.divider()
    
    # About section
    st.markdown("### About")
    st.markdown("""
    This application demonstrates a multimodal retrieval-augmented generation (RAG) system.
    
    It processes text, tables, and images from uploaded files and allows you to ask questions
    about their content.
    
    Your files and API key are not stored permanently.
    """)

# Main layout - two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Files")
    
    # File upload - support for PDFs and common image formats
    uploaded_files = st.file_uploader(
        "Upload documents (PDF) and images (JPG, JPEG, PNG)",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one or more files to process"
    )
    
    # Process documents button (separate from question answering)
    process_button = st.button(
        "Process Documents", 
        disabled=not (uploaded_files and openai_api_key),
        type="primary"
    )
    
    # Show processing status and instruction
    if st.session_state.documents_processed:
        st.success("Documents processed! You can now ask questions.")
    elif st.session_state.processing_error:
        st.error(f"Error during processing: {st.session_state.processing_error}")
    elif not uploaded_files:
        st.info("Please upload at least one file to continue.")
    elif not openai_api_key:
        st.warning("Please enter your OpenAI API key.")
    
    # Question input (only show if documents are processed)
    if st.session_state.documents_processed:
        st.header("Ask a Question")
        question = st.text_input(
            "Enter your question about the uploaded content",
            help="Ask a question about the content in your uploaded files"
        )
        
        # Answer button
        answer_button = st.button(
            "Get Answer", 
            disabled=not question,
            type="primary"
        )
    
# Results column
with col2:
    # Process documents when the process button is clicked
    if process_button and uploaded_files and openai_api_key:
        st.header("Processing Documents")
        with st.spinner("Processing files... this may take several minutes for large documents"):
            try:
                # Prepare files for the request
                files = []
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    file_content = uploaded_file.read()
                    files.append(("files", (uploaded_file.name, file_content, uploaded_file.type)))
                
                # Prepare form data
                form_data = {
                    "api_key": openai_api_key,
                    "question": "Summarize the content of these documents briefly."  # Initial summarization question
                }
                
                start_time = time.time()
                  # Process documents endpoint - with increased timeout
                response = requests.post(
                    f"{API_URL}/process",  # Using existing /process endpoint
                    files=files,
                    data=form_data,
                    timeout=3600  # 1 hour timeout for large files
                )
                
                processing_time = time.time() - start_time
                
                # Check if request was successful
                if response.status_code == 200:
                    result = response.json()
                    # Store session ID for future queries
                    st.session_state.session_id = result.get("session_id", "temp-session")
                    st.session_state.documents_processed = True
                    st.session_state.processing_error = None
                    
                    # Display processing summary
                    st.success("Documents processed successfully!")
                    st.markdown("### Processing Summary")
                    st.markdown(f"- **Processing time:** {processing_time:.2f} seconds")
                    
                    # Display counts if available
                    if "source_count" in result:
                        st.markdown("### Document Stats")
                        st.markdown(f"- **PDF files:** {result['source_count'].get('pdf_files', 0)}")
                        st.markdown(f"- **Image files:** {result['source_count'].get('image_files', 0)}")
                        st.markdown(f"- **Text chunks:** {result['source_count'].get('text_chunks', 0)}")
                        st.markdown(f"- **Tables:** {result['source_count'].get('tables', 0)}")
                        st.markdown(f"- **Images:** {result['source_count'].get('images', 0)}")
                    
                    # Show initial summary
                    if "answer" in result:
                        st.markdown("### Document Summary")
                        st.markdown(result["answer"])
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    st.session_state.processing_error = f"HTTP {response.status_code}: {response.text}"
                    
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to backend at {API_URL}. Make sure the FastAPI server is running.")
                st.session_state.processing_error = "Connection error"
            except requests.exceptions.Timeout:
                st.error("Request timed out. Your documents may be too large or complex for processing.")
                st.session_state.processing_error = "Request timed out"
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.processing_error = str(e)
    
    # Handle question answering (only if documents are processed)
    if st.session_state.documents_processed and 'answer_button' in locals() and answer_button and question:
        st.header("Answer")
        with st.spinner("Generating answer..."):
            try:
                # Prepare request to query endpoint
                query_data = {
                    "api_key": openai_api_key,
                    "question": question
                }
                
                start_time = time.time()
                
                # Query the processed documents (using the same process endpoint)
                # We include the same files in the request to avoid changing the backend
                files = []
                for uploaded_file in uploaded_files:
                    # Reset the file read position
                    uploaded_file.seek(0)
                    file_content = uploaded_file.read()
                    files.append(("files", (uploaded_file.name, file_content, uploaded_file.type)))
                
                response = requests.post(
                    f"{API_URL}/process",
                    files=files,
                    data=query_data,
                    timeout=120  # 2 minutes should be enough for generating an answer
                )
                
                query_time = time.time() - start_time
                
                # Check if request was successful
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add to answer history
                    st.session_state.answer_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "time": query_time
                    })
                    
                    # Display the answer
                    st.markdown("#### Question:")
                    st.markdown(f"*{question}*")
                    st.markdown("#### Answer:")
                    st.markdown(result["answer"])
                    st.caption(f"Answer generated in {query_time:.2f} seconds")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to backend at {API_URL}. Make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Display answer history
    if st.session_state.answer_history:
        st.header("Previous Questions")
        for i, qa in enumerate(reversed(st.session_state.answer_history[:-1]), 1):
            with st.expander(f"Q: {qa['question']}"):
                st.markdown(qa["answer"])
                st.caption(f"Generated in {qa['time']:.2f} seconds")

# Reset button - in sidebar at bottom
with st.sidebar:
    st.divider()
    if st.button("Reset Session", type="secondary"):
        # Clear session state
        st.session_state.documents_processed = False
        st.session_state.session_id = None
        st.session_state.processing_error = None
        st.session_state.answer_history = []
        st.rerun()
        
# Footer
st.markdown("---")
st.caption("Multimodal RAG System | Built with Streamlit, FastAPI, LangChain, and OpenAI")