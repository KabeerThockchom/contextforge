# tests/conftest.py
"""Shared test fixtures."""

import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock
from contextforge import ContextEngine, ToolRegistry, VectorRetriever
from contextforge.memory import InMemoryStore
from contextforge.tools import Tool


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = Mock()
    provider.generate = AsyncMock(return_value="Test response")
    provider.get_info = Mock(return_value={"provider": "mock", "model": "test"})
    return provider


@pytest.fixture
def context_engine(mock_provider):
    """Create a ContextEngine instance with mock provider."""
    return ContextEngine(provider=mock_provider, default_system_prompt="You are a test assistant.")


@pytest.fixture
def memory_store():
    """Create an in-memory store."""
    return InMemoryStore()


@pytest.fixture
def tool_registry():
    """Create a tool registry with sample tools."""
    registry = ToolRegistry()
    
    @registry.register(description="Test tool")
    def test_tool(x: int) -> int:
        """Multiply by 2."""
        return x * 2
    
    return registry


@pytest.fixture
def vector_retriever():
    """Create a vector retriever."""
    return VectorRetriever()


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sqlite_memory_store(temp_sqlite_db):
    """Create a SQLite memory store."""
    from contextforge.memory import SQLiteMemoryStore
    return SQLiteMemoryStore(temp_sqlite_db)


@pytest.fixture
def sample_documents():
    """Sample documents for retrieval testing."""
    return [
        {
            "content": "Paris is the capital of France",
            "source": "geography",
            "metadata": {"topic": "cities"}
        },
        {
            "content": "Python is a programming language",
            "source": "technology",
            "metadata": {"topic": "programming"}
        },
        {
            "content": "The Eiffel Tower is in Paris",
            "source": "landmarks",
            "metadata": {"topic": "architecture"}
        }
    ]