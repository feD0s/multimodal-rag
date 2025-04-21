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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status and information |
| `/health` | POST | Health check endpoint |
| `/process` | POST | Process documents and answer questions |

### Request Format
```json
{
  "files": [binary_file],
  "api_key": "your_openai_api_key",
  "question": "What information is in these documents?"
}
```

### Response Format
```json
{
  "answer": "Detailed answer based on document content...",
  "processing_time": "2.5s",
  "source_count": {
    "texts": 5,
    "tables": 2,
    "images": 3
  }
}
```

## 🔍 Usage Examples

1. **Upload a PDF**:
   - Use the Streamlit interface to upload a PDF document
   - The system extracts text, tables, and images

2. **Ask Questions**:
   - Type your question in the input field
   - The system retrieves relevant context and generates an answer
   - View the response with information from all modalities

3. **API Integration**:
   - Send POST requests to the `/process` endpoint
   - Include document files and your question
   - Receive structured answers for integration with other systems

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
