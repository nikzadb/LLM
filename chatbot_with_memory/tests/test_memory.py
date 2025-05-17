
"""Tests for memory management."""

import pytest
from unittest.mock import Mock, MagicMock
from langchain.schema import Document

from src.memory import ChatbotMemory

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock = Mock()
    mock.add_texts = MagicMock()
    mock.similarity_search = MagicMock(return_value=[
        Document(page_content="Human: Test\nAI: Response")
    ])
    return mock


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    return Mock()


@pytest.fixture
def chatbot_memory(mock_vector_store):
    """Create a ChatbotMemory instance with mocks."""
    return ChatbotMemory(
        vector_store=mock_vector_store,
        memory_k=3
    )


def test_add_to_memory(chatbot_memory, mock_vector_store):

    chatbot_memory.add_to_memory("Hello", "Hi there!")
    
    # # Check that vector store was called
    mock_vector_store.add_texts.assert_called_once()
    call_args = mock_vector_store.add_texts.call_args[1]['texts']
    assert call_args == ["Human: Hello\nAi: Hi there!"]


def test_get_relevant_memories(chatbot_memory, mock_vector_store):

    short_mem = chatbot_memory.get_relevant_memories("test query")
    assert short_mem == "Unknown time: Human: Test\nAI: Response"
    
    # Check that similarity search was called
    long_mem = mock_vector_store.similarity_search.assert_called_once_with("test query", k=3)


def test_get_conversation_history(chatbot_memory):
    """Test getting conversation history."""
    # Add some conversation to history
    chatbot_memory.add_to_memory("Hello", "Hi!")
    history = chatbot_memory.get_conversation_history()
    
    assert "history" in history
    assert "Hello" in history["history"]
    assert "Hi!" in history["history"]


def test_clear_short_term_memory(chatbot_memory):
    """Test clearing short-term memory."""
    chatbot_memory.add_to_memory("Hello", "Hi!")
    chatbot_memory.clear_short_term_memory()
    
    history = chatbot_memory.get_conversation_history()
    assert history.get("history", "") == ""

