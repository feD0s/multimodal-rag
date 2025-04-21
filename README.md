# Multimodal RAG System

A robust multimodal Retrieval Augmented Generation (RAG) system capable of processing text, tables, and images from various document formats to answer user queries with enhanced context and accuracy.

## 🚀 Features

- **Multimodal Processing**: Handles text, tables, and images from PDFs and image files
- **Advanced Summarization**: Generates concise summaries for different content types
- **Vector-based Retrieval**: Uses embedding models for semantic understanding
- **Optimized Performance**: Efficiently processes large documents with parallel operations
- **Dual Interface**: Accessible via REST API and user-friendly Streamlit interface
- **Robust Error Handling**: Comprehensive error management with detailed logging

## 🏗️ Architecture

The system consists of three main components:

1. **RAG Processing Core** (`rag_processing.py`):
   - Document parsing and content extraction
   - Text and image summarization
   - Vector store creation and retrieval
   - LLM integration for response generation

2. **FastAPI Backend** (`main.py`):
   - RESTful API endpoints for document processing
   - Asynchronous request handling
   - Error management and response formatting
   - API authentication

3. **Streamlit Frontend** (`streamlit_app.py`):
   - User-friendly interface for document upload
   - Question input and answer display
   - Interactive experience

## 🛠️ Technologies Used

- **LangChain**: Core framework for RAG pipeline creation
- **OpenAI**: GPT models for text processing and multimodal understanding
- **ChromaDB**: Vector database for efficient document retrieval
- **Unstructured**: Document parsing and element extraction
- **FastAPI**: High-performance API backend
- **Streamlit**: Interactive web interface
- **Docker**: Containerization for easy deployment

## 🔧 Setup and Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd multimodal-rag
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t multimodal-rag-app .
   docker run -p 8000:8000 -p 8501:8501
   ```

3. Access the applications:
   - FastAPI backend: http://localhost:8000
   - Streamlit frontend: http://localhost:8501


## 📚 API Documentation

### REST API Endpoints

| Endpoint             | Method | Description                                      |
|----------------------|--------|--------------------------------------------------|
| `/`                  | GET    | API status and information                       |
| `/health`            | POST   | Health check endpoint                            |
| `/process_documents` | POST   | Process uploaded documents and get a session ID  |
| `/query`             | POST   | Ask a question using an existing session ID      |

### Request Format (`/process_documents`)

This endpoint expects `multipart/form-data`:
- `files`: One or more file uploads (PDF, JPG, PNG, etc.)
- `api_key`: Your OpenAI API key (form field)
- `question`: An initial question to generate a summary (form field, optional)

### Response Format (`/process_documents`)
```json
{
  "session_id": "unique-session-identifier",
  "summary": "Optional initial summary of the documents...",
  "processing_time": 15.7,
  "source_count": {
    "pdf_files": 1,
    "image_files": 2,
    "text_chunks": 50,
    "tables": 3,
    "images": 5
  }
}
```

### Request Format (`/query`)

This endpoint expects `application/json`:
```json
{
  "api_key": "your_openai_api_key",
  "question": "What specific information is in the documents?",
  "session_id": "unique-session-identifier-from-process_documents"
}
```

### Response Format (`/query`)
```json
{
  "answer": "Detailed answer based on document content and question...",
  "images": [
    "base64_encoded_image_string_1", 
    "base64_encoded_image_string_2"
  ],
  "query_time": 3.1
}
```

## 🔍 Usage Examples

1. **Upload Files & Start Session (Streamlit or API)**:
   - Use the Streamlit interface or send a POST request to `/process_documents` with your files and API key.
   - Receive a `session_id` and an initial summary.

2. **Ask Questions (Streamlit or API)**:
   - In Streamlit: Use the input field that appears after processing.
   - Via API: Send a POST request to `/query` with your `session_id`, API key, and question.
   - View the response, including text answers and relevant images.

3. **API Integration Workflow**:
   - **Step 1:** Send POST request to `/process_documents` with files.
   - **Step 2:** Store the returned `session_id`.
   - **Step 3:** Send POST requests to `/query` using the stored `session_id` for subsequent questions.

## 🧪 Testing

Run the comprehensive test suite to verify functionality:

```bash
pytest -v
```

Tests cover:
- Core RAG processing functions
- API endpoints and error handling
- Content processing capabilities

## 🔒 Security Notes

- API keys should be kept secure and not committed to version control
- Consider implementing additional authentication for production deployments
- The system processes and stores document content temporarily - implement data retention policies as needed

## 📋 License

[MIT License](LICENSE)

## 🤝 Contributions

Contributions are welcome! Please feel free to submit a Pull Request.

---

*Built with Sonnet 3.7 and Gemini 2.5 Pro*
