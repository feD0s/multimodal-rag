import streamlit as st
import requests
import io
import os
import time
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Multimodal RAG - Interactive Q&A with Documents",
    page_icon="🔍",
    layout="wide",
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

# Sidebar for API key and configuration
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    st.caption("Your API key is not stored and will be used only for this session.")
    
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    **Multimodal RAG System**  
    This app extracts and analyzes information from:
    * PDF documents (text and tables)
    * Images (visual content)
    
    It uses LangChain, ChromaDB, and OpenAI's multimodal models to provide accurate answers to your questions.
    """)

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    # File uploader
    st.subheader("1. Upload File(s)")
    uploaded_files = st.file_uploader(
        "Upload PDFs or images", 
        type=["pdf", "jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    # Display uploaded files
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s):")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.type})")
    
    # Question input
    st.subheader("2. Ask a Question")
    question = st.text_input("What would you like to know about the uploaded content?")
    
    # Process button
    process_button = st.button("Process and Answer", type="primary", disabled=not (uploaded_files and question and openai_api_key))

with col2:
    # Results area
    st.subheader("3. Results")
    
    if not uploaded_files:
        st.info("Please upload at least one file to begin.")
    elif not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not question:
        st.info("Please enter a question about the uploaded content.")
    
    # Handle form submission
    if process_button and uploaded_files and question and openai_api_key:
        with st.spinner("Processing files and generating answer..."):
            try:
                # Prepare the request to the backend
                files = []
                
                # Handle file uploads
                for uploaded_file in uploaded_files:
                    # Convert to bytes
                    bytes_data = uploaded_file.getvalue()
                    # Create a tuple for files parameter (field name, file data, filename)
                    files.append(
                        ("files", (uploaded_file.name, bytes_data, uploaded_file.type))
                    )
                
                # Include form data
                form_data = {
                    "api_key": openai_api_key,
                    "question": question
                }
                
                # Make request to API
                start_time = time.time()
                response = requests.post(
                    f"{API_URL}/process",
                    files=files,
                    data=form_data,
                    timeout=300  # 5 minutes timeout for large files
                )
                processing_time = time.time() - start_time
                
                # Check if request was successful
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display the answer
                    st.markdown("#### Answer:")
                    st.markdown(result["answer"])
                    
                    # Display processing metadata
                    st.caption(f"Processing time: {processing_time:.2f} seconds")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to backend at {API_URL}. Make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
# Footer
st.markdown("---")
st.caption("Multimodal RAG System | Built with Streamlit, FastAPI, LangChain, and OpenAI")
