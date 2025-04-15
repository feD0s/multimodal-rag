import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import io

from main import app

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

@pytest.mark.asyncio
@patch('main.OpenAIEmbeddings')
@patch('main.Chroma')
@patch('main.extract_pdf_elements')
@patch('main.generate_text_summaries')
@patch('main.generate_img_summaries')
@patch('main.create_multi_vector_retriever')
@patch('main.multi_modal_rag_chain')
async def test_process_endpoint_with_pdf(
    mock_rag_chain,
    mock_create_retriever,
    mock_generate_img_summaries,
    mock_generate_text_summaries,
    mock_extract_pdf_elements,
    mock_chroma,
    mock_embeddings
):
    """Test the /process endpoint with a PDF file."""
    # Setup mocks
    mock_elements = [MagicMock(), MagicMock()]
    mock_extract_pdf_elements.return_value = mock_elements
    
    mock_texts = ["Text 1", "Text 2"]
    mock_tables = ["Table 1"]
    mock_extract_pdf_elements.return_value = mock_elements
    
    # Mock categorize_elements to return some text and tables
    with patch('main.categorize_elements', return_value=(mock_texts, mock_tables)):
        # Mock the summarization functions
        mock_generate_text_summaries.return_value = (["Summary 1", "Summary 2"], ["Table Summary 1"])
        mock_generate_img_summaries.return_value = (["img_base64"], ["Image Summary 1"])
        
        # Mock retriever and chain
        mock_retriever = MagicMock()
        mock_create_retriever.return_value = mock_retriever
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "This is a test answer about the document."
        mock_rag_chain.return_value = mock_chain
        
        # Create a test PDF file
        test_pdf_content = b'%PDF-1.5\nTest PDF content'
        test_pdf = io.BytesIO(test_pdf_content)
        
        # Make request to the endpoint
        response = client.post(
            "/process",
            files={"files": ("test.pdf", test_pdf, "application/pdf")},
            data={
                "api_key": "test_api_key",
                "question": "What is in this document?"
            }
        )
        
        # Check response
        assert response.status_code == 200
        assert "answer" in response.json()
        assert response.json()["answer"] == "This is a test answer about the document."
        
        # Verify mocks were called correctly
        mock_extract_pdf_elements.assert_called_once()
        mock_generate_text_summaries.assert_called_once()
        mock_generate_img_summaries.assert_called_once()
        mock_create_retriever.assert_called_once()
        mock_chain.invoke.assert_called_once_with("What is in this document?")

# Additional tests can be added for error conditions, different file types, etc.
