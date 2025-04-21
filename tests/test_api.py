import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import io
import uuid

# Add this to mock the session storage
from main import app, SESSION_DATA

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns correct welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Multimodal RAG API" in response.json()["message"]

def test_health_check():
    """Test the health check endpoint."""
    response = client.post("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.fixture(scope="function", autouse=True)
def clear_session_data():
    """Fixture to clear session data before each test."""
    SESSION_DATA.clear()
    yield
    SESSION_DATA.clear()

@pytest.mark.asyncio
@patch('main.OpenAIEmbeddings')
@patch('main.Chroma')
@patch('main.extract_pdf_elements')
@patch('main.categorize_elements') # Added mock for categorize_elements
@patch('main.generate_text_summaries')
@patch('main.generate_img_summaries')
@patch('main.create_multi_vector_retriever')
@patch('main.multi_modal_rag_chain') # Mock the chain for initial summary
@patch('uuid.uuid4') # Mock uuid generation for predictable session ID
async def test_process_documents_endpoint(
    mock_uuid,
    mock_rag_chain_summary, # Renamed for clarity
    mock_create_retriever,
    mock_generate_img_summaries,
    mock_generate_text_summaries,
    mock_categorize_elements, # Added parameter
    mock_extract_pdf_elements,
    mock_chroma,
    mock_embeddings
):
    """Test the /process_documents endpoint with a PDF file."""
    # Setup mocks
    mock_uuid.return_value = uuid.UUID('12345678-1234-5678-1234-567812345678') # Predictable UUID
    test_session_id = str(mock_uuid.return_value)

    mock_elements = [MagicMock(), MagicMock()]
    mock_extract_pdf_elements.return_value = mock_elements

    mock_texts = ["Text 1", "Text 2"]
    mock_tables = ["Table 1"]
    mock_categorize_elements.return_value = (mock_texts, mock_tables)

    mock_generate_text_summaries.return_value = (["Summary 1", "Summary 2"], ["Table Summary 1"])
    mock_generate_img_summaries.return_value = (["img_base64_1"], ["Image Summary 1"])

    mock_retriever = MagicMock()
    mock_create_retriever.return_value = mock_retriever

    # Mock the chain used for the *initial* summary
    mock_summary_chain = MagicMock()
    mock_summary_chain.invoke.return_value = {"answer": "Initial document summary.", "images": []}
    mock_rag_chain_summary.return_value = mock_summary_chain

    # Create a test PDF file
    test_pdf_content = b'%PDF-1.5\nTest PDF content'
    test_pdf = io.BytesIO(test_pdf_content)

    # Make request to the endpoint
    response = client.post(
        "/process_documents",
        files={"files": ("test.pdf", test_pdf, "application/pdf")},
        data={
            "api_key": "test_api_key",
            "question": "Generate initial summary" # The question for the initial summary
        }
    )

    # Check response
    assert response.status_code == 200
    response_data = response.json()
    assert "session_id" in response_data
    assert response_data["session_id"] == test_session_id
    assert "summary" in response_data
    assert response_data["summary"] == "Initial document summary."
    assert "source_count" in response_data
    assert response_data["source_count"]["pdf_files"] == 1
    assert response_data["source_count"]["images"] == 1 # Based on mock_generate_img_summaries

    # Verify mocks were called correctly
    mock_extract_pdf_elements.assert_called_once()
    mock_categorize_elements.assert_called_once_with(mock_elements)
    mock_generate_text_summaries.assert_called_once()
    mock_generate_img_summaries.assert_called_once()
    mock_create_retriever.assert_called_once()
    mock_summary_chain.invoke.assert_called_once_with("Generate initial summary")

    # Check if retriever was stored in session
    assert test_session_id in SESSION_DATA
    assert SESSION_DATA[test_session_id]["retriever"] == mock_retriever

@pytest.mark.asyncio
@patch('main.multi_modal_rag_chain')
async def test_query_endpoint(
    mock_rag_chain_query # Renamed for clarity
):
    """Test the /query endpoint after documents have been processed."""
    # Setup: Mock a processed session
    test_session_id = "existing_session_123"
    mock_retriever_instance = MagicMock()
    SESSION_DATA[test_session_id] = {
        "retriever": mock_retriever_instance,
        "created_at": 12345.67,
        "last_accessed": 12345.67
    }

    # Mock the RAG chain for the query
    mock_query_chain = MagicMock()
    mock_query_chain.invoke.return_value = {
        "answer": "This is the answer to the specific question.",
        "images": ["img_base64_retrieved"]
    }
    mock_rag_chain_query.return_value = mock_query_chain

    # Make request to the /query endpoint
    response = client.post(
        "/query",
        json={
            "api_key": "test_api_key",
            "question": "What is the specific detail?",
            "session_id": test_session_id
        }
    )

    # Check response
    assert response.status_code == 200
    response_data = response.json()
    assert "answer" in response_data
    assert response_data["answer"] == "This is the answer to the specific question."
    assert "images" in response_data
    assert response_data["images"] == ["img_base64_retrieved"]
    assert "query_time" in response_data

    # Verify the correct chain was called with the retriever from the session
    mock_rag_chain_query.assert_called_once_with(mock_retriever_instance)
    mock_query_chain.invoke.assert_called_once_with("What is the specific detail?")

@pytest.mark.asyncio
async def test_query_endpoint_invalid_session():
    """Test the /query endpoint with an invalid session ID."""
    response = client.post(
        "/query",
        json={
            "api_key": "test_api_key",
            "question": "What is the specific detail?",
            "session_id": "non_existent_session"
        }
    )
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]
