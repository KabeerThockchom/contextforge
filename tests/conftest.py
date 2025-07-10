# tests/conftest.py
"""Test configuration and fixtures."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock
from contextforge import ContextEngine, ToolRegistry, VectorRetriever
from contextforge.memory import InMemoryStore, SQLiteMemoryStore
from contextforge.providers import Provider


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider for testing."""
    provider = Mock(spec=Provider)
    provider.generate = AsyncMock(return_value="Test response")
    provider.get_info = Mock(return_value={"provider": "mock", "model": "test"})
    return provider


@pytest.fixture
def context_engine(mock_provider):
    """Create a ContextEngine with mock provider."""
    return ContextEngine(
        provider=mock_provider,
        default_system_prompt="You are a test assistant."
    )


@pytest.fixture
def tool_registry():
    """Create a ToolRegistry for testing."""
    registry = ToolRegistry()
    
    @registry.register(description="Test tool")
    def test_tool(text: str) -> str:
        return f"Processed: {text}"
    
    @registry.register(description="Async test tool")
    async def async_test_tool(number: int) -> int:
        return number * 2
    
    return registry


@pytest.fixture
def vector_retriever():
    """Create a VectorRetriever for testing."""
    return VectorRetriever()


@pytest.fixture
def memory_store():
    """Create an InMemoryStore for testing."""
    return InMemoryStore()


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Clean up
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def sqlite_memory_store(temp_sqlite_db):
    """Create a SQLiteMemoryStore for testing."""
    return SQLiteMemoryStore(temp_sqlite_db)


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]


@pytest.fixture
def sample_documents():
    """Sample documents for retrieval testing."""
    return [
        {"content": "Paris is the capital of France.", "source": "Wikipedia"},
        {"content": "The Eiffel Tower is in Paris.", "source": "Travel Guide"},
        {"content": "Python is a programming language.", "source": "Tech Doc"}
    ]