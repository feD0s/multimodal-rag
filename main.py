import os
import uuid
import shutil
import time
import base64  # Added missing import
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import traceback

# Configure more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define custom exceptions for better error handling
class RAGProcessingError(Exception):
    """Base exception for RAG processing errors."""
    pass

class FileProcessingError(RAGProcessingError):
    """Exception raised when file processing fails."""
    pass

class SummarizationError(RAGProcessingError):
    """Exception raised when text or image summarization fails."""
    pass

class RetrieverError(RAGProcessingError):
    """Exception raised when retriever creation or querying fails."""
    pass

# Import functions from your rag_processing.py
try:
    from rag_processing import (
        extract_pdf_elements,
        categorize_elements,
        generate_text_summaries,
        generate_img_summaries,
        create_multi_vector_retriever,
        multi_modal_rag_chain,
        encode_image,  # Added for direct image handling
    )
    # Necessary imports
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
except ImportError as e:
    logger.critical(f"Critical import error from rag_processing: {e}")
    logger.critical(f"Traceback: {traceback.format_exc()}")
    raise ImportError(f"Failed to import required modules: {e}")

app = FastAPI(
    title="Multimodal RAG API",
    description="API for processing documents and images with RAG approach",
    version="1.0.0"
)

# --- Pydantic Models ---
class ProcessRequest(BaseModel):
    api_key: str
    question: str

class ProcessResponse(BaseModel):
    answer: str
    processing_time: float
    source_count: Dict[str, int]  # Count of each type of source used

class ProcessDocumentsResponse(BaseModel):
    session_id: str
    summary: Optional[str] = None
    processing_time: float
    source_count: Dict[str, int]

class QueryRequest(BaseModel):
    api_key: str
    question: str
    session_id: str

class QueryResponse(BaseModel):
    answer: str
    query_time: float

class ErrorResponse(BaseModel):
    detail: str
    error_type: str

# --- Session Storage ---
# Dictionary to store session data
SESSION_DATA = {}

# --- Temporary Storage ---
TEMP_DIR = "temp_storage"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(TEMP_DIR, "figures"), exist_ok=True)

# --- Helper Functions ---
def cleanup_session_files(session_path: str) -> None:
    """Clean up temporary session files in the background."""
    try:
        if os.path.exists(session_path):
            shutil.rmtree(session_path)
            logger.info(f"Cleaned up temporary directory: {session_path}")
    except Exception as e:
        logger.error(f"Error cleaning up session directory {session_path}: {e}")

def validate_file_type(filename: str) -> str:
    """Validate and return the file type."""
    lower_filename = filename.lower()
    if lower_filename.endswith(('.pdf')):
        return 'pdf'
    elif lower_filename.endswith(('.jpg', '.jpeg', '.png')):
        return 'image'
    else:
        raise ValueError(f"Unsupported file type: {filename}. Supported types are PDF, JPG, JPEG, PNG.")

async def cleanup_session_after_inactivity(session_id: str, max_inactive_seconds: int):
    """Clean up session data after specified period of inactivity."""
    while session_id in SESSION_DATA:
        # Check if session should be removed
        session_data = SESSION_DATA[session_id]
        inactive_time = time.time() - session_data["last_accessed"]
        
        if inactive_time > max_inactive_seconds:
            # Remove from session data
            if session_id in SESSION_DATA:
                del SESSION_DATA[session_id]
                logger.info(f"Removed inactive session {session_id}")
            
            # Remove files
            session_path = os.path.join(TEMP_DIR, session_id)
            if os.path.exists(session_path):
                try:
                    shutil.rmtree(session_path)
                    logger.info(f"Cleaned up session directory {session_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up session directory {session_path}: {e}")
            
            break
        
        # Wait and check again
        await asyncio.sleep(60)  # Check every minute

# --- API Endpoints ---
@app.get("/")
async def root():
    return {
        "message": "Multimodal RAG API is running",
        "status": "ok",
        "version": "1.0.0"
    }

@app.post("/health")
async def health_check():
    """Endpoint to check if the API and its dependencies are working."""
    try:
        # Check if OpenAI API key environment variable is set (not its value)
        api_key_set = "OPENAI_API_KEY" in os.environ
        return {
            "status": "ok",
            "api_key_configured": api_key_set,
            "temp_directory": os.path.exists(TEMP_DIR)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    start_time = time.time()
    logger.info(f"Query received for session {request.session_id}")
    
    # Validate session exists
    if request.session_id not in SESSION_DATA:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found. Please process documents first."
        )
    
    # Set OpenAI API Key
    os.environ["OPENAI_API_KEY"] = request.api_key
    
    try:
        # Get retriever from session
        session_data = SESSION_DATA[request.session_id]
        retriever = session_data["retriever"]
        
        # Update last accessed time
        session_data["last_accessed"] = time.time()
        
        # Create and invoke RAG chain
        chain = multi_modal_rag_chain(retriever)
        result = chain.invoke(request.question)
        
        # Calculate query time
        query_time = time.time() - start_time
        logger.info(f"Query processed in {query_time:.2f} seconds")
        
        return QueryResponse(
            answer=result,
            query_time=query_time
        )
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {str(e)}"
        )
    finally:
        # Clean up API key
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

@app.post("/health")
async def health_check():
    """Endpoint to check if the API and its dependencies are working."""
    try:
        # Check if OpenAI API key environment variable is set (not its value)
        api_key_set = "OPENAI_API_KEY" in os.environ
        return {
            "status": "ok",
            "api_key_configured": api_key_set,
            "temp_directory": os.path.exists(TEMP_DIR)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/process_documents", response_model=ProcessDocumentsResponse)
async def process_documents(
    background_tasks: BackgroundTasks,
    api_key: str = Form(...),
    question: str = Form(...),  # This will be the initial summarization question
    files: List[UploadFile] = File(...)
):
    start_time = time.time()
    logger.info(f"Processing documents, number of files={len(files)}")
    
    # Validate input
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No files provided"
        )
    
    # Create session ID and paths
    session_id = str(uuid.uuid4())
    session_path = os.path.join(TEMP_DIR, session_id)
    image_output_path = os.path.join(session_path, "figures")
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(image_output_path, exist_ok=True)
    
    # Set OpenAI API Key
    os.environ["OPENAI_API_KEY"] = api_key
    
    file_paths = []
    processed_elements = {'texts': [], 'tables': [], 'images': []}
    source_counts = {'pdf_files': 0, 'image_files': 0, 'text_chunks': 0, 'tables': 0, 'images': 0}
    result = ""  # For the initial summary

    try:
        # Process each file
        for file in files:
            try:
                file_location = os.path.join(session_path, file.filename)
                with open(file_location, "wb+") as file_object:
                    shutil.copyfileobj(file.file, file_object)
                file_paths.append(file_location)
                logger.info(f"Saved file: {file_location}")

                # Determine file type and process accordingly
                file_type = validate_file_type(file.filename)
                
                if file_type == 'pdf':
                    source_counts['pdf_files'] += 1
                    logger.info(f"Processing PDF: {file.filename}")
                    try:
                        raw_elements = extract_pdf_elements(
                            session_path + os.path.sep, 
                            file.filename,
                            image_output_path  # Pass the image output path
                        )
                        if not raw_elements:
                            logger.warning(f"No elements extracted from PDF: {file.filename}")
                            continue
                            
                        texts, tables = categorize_elements(raw_elements)
                        processed_elements['texts'].extend(texts)
                        processed_elements['tables'].extend(tables)
                        logger.info(f"Extracted {len(texts)} text elements and {len(tables)} tables from {file.filename}")
                    except Exception as e:
                        logger.error(f"Error processing PDF {file.filename}: {e}")
                        raise FileProcessingError(f"Failed to process PDF {file.filename}: {str(e)}")
                        
                elif file_type == 'image':
                    source_counts['image_files'] += 1
                    logger.info(f"Processing image: {file.filename}")
                    try:
                        # Directly encode image to base64 and add to processed_elements
                        base64_image = encode_image(file_location)
                        if base64_image:
                            processed_elements['images'].append(base64_image)
                            logger.info(f"Successfully encoded image: {file.filename}")
                        else:
                            logger.warning(f"Failed to encode image: {file.filename}")
                    except Exception as e:
                        logger.error(f"Error processing image {file.filename}: {e}")
                        raise FileProcessingError(f"Failed to process image {file.filename}: {str(e)}")
            except ValueError as e:
                # Skip unsupported file types but log it
                logger.warning(f"Skipping file: {file.filename}. Reason: {str(e)}")
                continue

        # Validate that we have some content to process
        if not processed_elements['texts'] and not processed_elements['tables'] and not processed_elements['images']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid content could be extracted from the provided files."
            )

        # 4. Consolidate and Summarize text content
        if processed_elements['texts']:
            logger.info("Starting text splitting and summarization...")
            try:
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=1500, chunk_overlap=250
                )
                joined_texts = " ".join(processed_elements['texts'])
                texts_chunked = text_splitter.split_text(joined_texts)
                source_counts['text_chunks'] = len(texts_chunked)
                
                # Generate summaries for texts and tables
                text_summaries, table_summaries = generate_text_summaries(
                    texts_chunked, processed_elements['tables'], summarize_texts=True
                )
                source_counts['tables'] = len(processed_elements['tables'])
                logger.info(f"Generated {len(text_summaries)} text summaries and {len(table_summaries)} table summaries.")
            except Exception as e:
                logger.error(f"Error during text summarization: {e}")
                raise SummarizationError(f"Failed to summarize text content: {str(e)}")
        else:
            texts_chunked = []
            text_summaries = []
            table_summaries = []
            logger.info("No text content to summarize.")

        # Generate summaries for extracted images from PDFs or direct image uploads
        try:
            # First, get any images extracted from PDFs
            img_base64_list, image_summaries = generate_img_summaries(image_output_path)
            
            # Then, add any directly uploaded images
            if processed_elements['images']:
                # For directly uploaded images, we need to generate summaries
                direct_img_summaries = []
                for img_base64 in processed_elements['images']:
                    # We'd use image_summarize here, but for simplicity we'll 
                    # just call generate_img_summaries with a temporary directory
                    temp_img_dir = os.path.join(session_path, "direct_images")
                    os.makedirs(temp_img_dir, exist_ok=True)
                    
                    # Save the base64 to a file so we can generate a summary
                    img_path = os.path.join(temp_img_dir, f"direct_img_{len(direct_img_summaries)}.jpg")
                    with open(img_path, "wb") as img_file:
                        img_file.write(base64.b64decode(img_base64))
                    
                    # Now generate summaries for this directory
                    direct_base64_list, direct_summaries = generate_img_summaries(temp_img_dir)
                    if direct_summaries:
                        img_base64_list.extend(direct_base64_list)
                        image_summaries.extend(direct_summaries)
                
            source_counts['images'] = len(img_base64_list)
            logger.info(f"Generated {len(image_summaries)} image summaries from {len(img_base64_list)} images.")
        except Exception as e:
            logger.error(f"Error during image summarization: {e}")
            raise SummarizationError(f"Failed to summarize image content: {str(e)}")

        # 5. Create Vector Store and Retriever
        logger.info("Creating vector store and retriever...")
        try:
            vectorstore = Chroma(
                collection_name=f"mm_rag_{session_id}",
                embedding_function=OpenAIEmbeddings()
            )
            retriever = create_multi_vector_retriever(
                vectorstore,
                text_summaries,
                texts_chunked,
                table_summaries,
                processed_elements['tables'],
                image_summaries,
                img_base64_list,
            )
            logger.info("Retriever created successfully.")
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise RetrieverError(f"Failed to create retriever: {str(e)}")

        # 6. Store the retriever in SESSION_DATA for future queries
        SESSION_DATA[session_id] = {
            "retriever": retriever,
            "created_at": time.time(),
            "last_accessed": time.time()
        }
        
        # Generate an initial summary if requested
        if question.strip():
            try:
                chain = multi_modal_rag_chain(retriever)
                result = chain.invoke(question)
                logger.info("Initial summary generated successfully.")
            except Exception as e:
                logger.error(f"Error generating initial summary: {e}")
                result = "Could not generate summary due to an error."
        
        # Schedule cleanup after inactivity (1 hour)
        background_tasks.add_task(cleanup_session_after_inactivity, session_id, 3600)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Document processing time: {processing_time:.2f} seconds")
        
        return ProcessDocumentsResponse(
            session_id=session_id,
            summary=result,
            processing_time=processing_time,
            source_count=source_counts
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle any other exceptions
        logger.error(f"Error processing documents: {e}", exc_info=True)
        # Clean up any partial session data
        if session_id in SESSION_DATA:
            del SESSION_DATA[session_id]
        # Clean up any files created
        if os.path.exists(session_path):
            try:
                shutil.rmtree(session_path)
            except:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )
    finally:
        # Explicitly clear the API key from environment
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

@app.post("/process", response_model=ProcessResponse)
async def process_files(
    background_tasks: BackgroundTasks,
    api_key: str = Form(...),
    question: str = Form(...),
    files: List[UploadFile] = File(...)
):
    start_time = time.time()
    logger.info(f"Received request: question='{question}', number of files={len(files)}")
    
    # Validate input
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No files provided"
        )
    
    if not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Question cannot be empty"
        )

    # 1. Set API Key (Consider security implications for production)
    os.environ["OPENAI_API_KEY"] = api_key
    logger.info("OpenAI API Key set.")

    # 2. Save uploaded files temporarily
    session_id = str(uuid.uuid4())
    session_path = os.path.join(TEMP_DIR, session_id)
    image_output_path = os.path.join(session_path, "figures")
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(image_output_path, exist_ok=True)

    file_paths = []
    processed_elements = {'texts': [], 'tables': [], 'images': []}
    source_counts = {'pdf_files': 0, 'image_files': 0, 'text_chunks': 0, 'tables': 0, 'images': 0}

    try:
        # Process each file
        for file in files:
            try:
                file_location = os.path.join(session_path, file.filename)
                with open(file_location, "wb+") as file_object:
                    shutil.copyfileobj(file.file, file_object)
                file_paths.append(file_location)
                logger.info(f"Saved file: {file_location}")

                # Determine file type and process accordingly
                file_type = validate_file_type(file.filename)
                
                if file_type == 'pdf':
                    source_counts['pdf_files'] += 1
                    logger.info(f"Processing PDF: {file.filename}")
                    try:
                        raw_elements = extract_pdf_elements(
                            session_path + os.path.sep, 
                            file.filename,
                            image_output_path  # Pass the image output path
                        )
                        if not raw_elements:
                            logger.warning(f"No elements extracted from PDF: {file.filename}")
                            continue
                            
                        texts, tables = categorize_elements(raw_elements)
                        processed_elements['texts'].extend(texts)
                        processed_elements['tables'].extend(tables)
                        logger.info(f"Extracted {len(texts)} text elements and {len(tables)} tables from {file.filename}")
                    except Exception as e:
                        logger.error(f"Error processing PDF {file.filename}: {e}")
                        raise FileProcessingError(f"Failed to process PDF {file.filename}: {str(e)}")
                        
                elif file_type == 'image':
                    source_counts['image_files'] += 1
                    logger.info(f"Processing image: {file.filename}")
                    try:
                        # Directly encode image to base64 and add to processed_elements
                        base64_image = encode_image(file_location)
                        if base64_image:
                            processed_elements['images'].append(base64_image)
                            logger.info(f"Successfully encoded image: {file.filename}")
                        else:
                            logger.warning(f"Failed to encode image: {file.filename}")
                    except Exception as e:
                        logger.error(f"Error processing image {file.filename}: {e}")
                        raise FileProcessingError(f"Failed to process image {file.filename}: {str(e)}")
            except ValueError as e:
                # Skip unsupported file types but log it
                logger.warning(f"Skipping file: {file.filename}. Reason: {str(e)}")
                continue

        # Validate that we have some content to process
        if not processed_elements['texts'] and not processed_elements['tables'] and not processed_elements['images']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid content could be extracted from the provided files."
            )

        # 4. Consolidate and Summarize text content
        if processed_elements['texts']:
            logger.info("Starting text splitting and summarization...")
            try:
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=1500, chunk_overlap=250
                )
                joined_texts = " ".join(processed_elements['texts'])
                texts_chunked = text_splitter.split_text(joined_texts)
                source_counts['text_chunks'] = len(texts_chunked)
                
                # Generate summaries for texts and tables
                text_summaries, table_summaries = generate_text_summaries(
                    texts_chunked, processed_elements['tables'], summarize_texts=True
                )
                source_counts['tables'] = len(processed_elements['tables'])
                logger.info(f"Generated {len(text_summaries)} text summaries and {len(table_summaries)} table summaries.")
            except Exception as e:
                logger.error(f"Error during text summarization: {e}")
                raise SummarizationError(f"Failed to summarize text content: {str(e)}")
        else:
            texts_chunked = []
            text_summaries = []
            table_summaries = []
            logger.info("No text content to summarize.")

        # Generate summaries for extracted images from PDFs or direct image uploads
        try:
            # First, get any images extracted from PDFs
            img_base64_list, image_summaries = generate_img_summaries(image_output_path)
            
            # Then, add any directly uploaded images
            if processed_elements['images']:
                # For directly uploaded images, we need to generate summaries
                direct_img_summaries = []
                for img_base64 in processed_elements['images']:
                    # We'd use image_summarize here, but for simplicity we'll 
                    # just call generate_img_summaries with a temporary directory
                    temp_img_dir = os.path.join(session_path, "direct_images")
                    os.makedirs(temp_img_dir, exist_ok=True)
                    
                    # Save the base64 to a file so we can generate a summary
                    img_path = os.path.join(temp_img_dir, f"direct_img_{len(direct_img_summaries)}.jpg")
                    with open(img_path, "wb") as img_file:
                        img_file.write(base64.b64decode(img_base64))
                    
                    # Now generate summaries for this directory
                    direct_base64_list, direct_summaries = generate_img_summaries(temp_img_dir)
                    if direct_summaries:
                        img_base64_list.extend(direct_base64_list)
                        image_summaries.extend(direct_summaries)
                
            source_counts['images'] = len(img_base64_list)
            logger.info(f"Generated {len(image_summaries)} image summaries from {len(img_base64_list)} images.")
        except Exception as e:
            logger.error(f"Error during image summarization: {e}")
            raise SummarizationError(f"Failed to summarize image content: {str(e)}")

        # 5. Create Vector Store and Retriever
        logger.info("Creating vector store and retriever...")
        try:
            vectorstore = Chroma(
                collection_name=f"mm_rag_{session_id}",
                embedding_function=OpenAIEmbeddings()
            )
            retriever = create_multi_vector_retriever(
                vectorstore,
                text_summaries,
                texts_chunked,
                table_summaries,
                processed_elements['tables'],
                image_summaries,
                img_base64_list,
            )
            logger.info("Retriever created successfully.")
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise RetrieverError(f"Failed to create retriever: {str(e)}")

        # 6. Create and Invoke RAG Chain
        logger.info("Creating and invoking RAG chain...")
        try:
            chain = multi_modal_rag_chain(retriever)
            result = chain.invoke(question)
            logger.info("RAG chain invoked successfully.")
        except Exception as e:
            logger.error(f"Error invoking RAG chain: {e}")
            raise RAGProcessingError(f"Failed to generate answer: {str(e)}")

        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")

        # Schedule cleanup to happen after response is returned
        background_tasks.add_task(cleanup_session_files, session_path)

        return ProcessResponse(
            answer=result,
            processing_time=processing_time,
            source_count=source_counts
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except FileProcessingError as e:
        logger.error(f"File processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": str(e), "error_type": "file_processing_error"}
        )
    except SummarizationError as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": str(e), "error_type": "summarization_error"}
        )
    except RetrieverError as e:
        logger.error(f"Retriever error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": str(e), "error_type": "retriever_error"}
        )
    except RAGProcessingError as e:
        logger.error(f"RAG processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": str(e), "error_type": "rag_processing_error"}
        )
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": f"An unexpected error occurred: {str(e)}", "error_type": "unexpected_error"}
        )
    finally:
        # Explicitly clear the API key from environment after use
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
            logger.info("OpenAI API Key cleared from environment.")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Multimodal RAG API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
