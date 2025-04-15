import pytest
import os
import base64
from unittest.mock import patch, MagicMock

# Import functions to test
from rag_processing import (
    looks_like_base64,
    is_image_data,
    split_image_text_types,
    generate_text_summaries,
)
from langchain_core.documents import Document

# Test utility functions
def test_looks_like_base64():
    """Test the looks_like_base64 function with various inputs."""
    # Test with valid base64 strings
    assert looks_like_base64("SGVsbG8gV29ybGQ=") == True
    assert looks_like_base64("VGhpcyBpcyBhIHRlc3Q=") == True
    
    # Test with invalid base64 strings
    assert looks_like_base64("Not base64!") == False
    assert looks_like_base64("") == False
    assert looks_like_base64("123@#$%^") == False

def test_is_image_data():
    """Test the is_image_data function with different inputs."""
    # Create a small test image in base64 (JPEG header)
    jpeg_header = b'\xFF\xD8\xFF' + b'\x00' * 10
    jpeg_base64 = base64.b64encode(jpeg_header).decode('utf-8')
    
    # Create a small test image in base64 (PNG header)
    png_header = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A' + b'\x00' * 10
    png_base64 = base64.b64encode(png_header).decode('utf-8')
    
    # Test image detection
    assert is_image_data(jpeg_base64) == True
    assert is_image_data(png_base64) == True
    
    # Test non-image data
    text_data = base64.b64encode(b"This is not an image").decode('utf-8')
    assert is_image_data(text_data) == False

def test_split_image_text_types():
    """Test the split_image_text_types function."""
    # Create test data
    jpeg_header = b'\xFF\xD8\xFF' + b'\x00' * 10
    jpeg_base64 = base64.b64encode(jpeg_header).decode('utf-8')
    
    docs = [
        Document(page_content="This is text content"),
        Document(page_content=jpeg_base64),
        Document(page_content="More text content")
    ]
    
    # Mock the resize_base64_image function to return input unchanged
    with patch('rag_processing.resize_base64_image', return_value=jpeg_base64):
        result = split_image_text_types(docs)
    
    # Verify results
    assert len(result["texts"]) == 2
    assert len(result["images"]) == 1
    assert result["texts"][0] == "This is text content"
    assert result["texts"][1] == "More text content"
    assert result["images"][0] == jpeg_base64

@patch('rag_processing.ChatOpenAI')
def test_generate_text_summaries_no_summarization(mock_chat):
    """Test generate_text_summaries when summarize_texts=False."""
    texts = ["Text 1", "Text 2"]
    tables = ["Table 1"]
    
    text_summaries, table_summaries = generate_text_summaries(texts, tables, summarize_texts=False)
    
    # When summarize_texts=False, should return original texts
    assert text_summaries == texts
    
    # Mock wasn't called for texts, only for tables
    assert mock_chat.return_value.invoke.call_count == 1

# More tests can be added for other functions
