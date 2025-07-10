# tests/test_retrieval.py
"""Tests for retrieval components."""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock
from contextforge.retrieval import Retriever, VectorRetriever


class TestRetriever:
    """Test abstract Retriever class."""
    
    def test_retriever_abstract(self):
        """Test that Retriever is abstract."""
        with pytest.raises(TypeError):
            Retriever()


class TestVectorRetriever:
    """Test VectorRetriever implementation."""
    
    def test_init_default(self):
        """Test initialization with default embedding model."""
        retriever = VectorRetriever()
        assert retriever.documents == []
        assert retriever.embeddings == []
        assert retriever.embedding_model is not None
    
    def test_init_custom_embedding_model(self):
        """Test initialization with custom embedding model."""
        def custom_embedding(text):
            return np.array([1.0, 2.0, 3.0])
        
        retriever = VectorRetriever(embedding_model=custom_embedding)
        assert retriever.embedding_model == custom_embedding
    
    def test_simple_embedding(self):
        """Test the simple embedding function."""
        retriever = VectorRetriever()
        
        # Test with short text
        embedding = retriever._simple_embedding("hello")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (128,)
        assert not np.allclose(embedding, 0)  # Should not be all zeros
        
        # Test with longer text
        long_text = "This is a longer text that should be processed correctly"
        embedding_long = retriever._simple_embedding(long_text)
        assert isinstance(embedding_long, np.ndarray)
        assert embedding_long.shape == (128,)
        
        # Different texts should produce different embeddings
        embedding_diff = retriever._simple_embedding("different text")
        assert not np.allclose(embedding, embedding_diff)
    
    def test_simple_embedding_normalization(self):
        """Test that embeddings are normalized."""
        retriever = VectorRetriever()
        embedding = retriever._simple_embedding("test text")
        
        # Check that the embedding is normalized (unit vector)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0)
    
    @pytest.mark.asyncio
    async def test_add_document(self, vector_retriever):
        """Test adding a document."""
        document = {
            "content": "This is a test document",
            "source": "test",
            "metadata": {"type": "article"}
        }
        
        await vector_retriever.add_document(document)
        
        assert len(vector_retriever.documents) == 1
        assert len(vector_retriever.embeddings) == 1
        assert vector_retriever.documents[0] == document
        assert isinstance(vector_retriever.embeddings[0], np.ndarray)
    
    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, vector_retriever):
        """Test adding multiple documents."""
        documents = [
            {"content": "First document", "source": "test1"},
            {"content": "Second document", "source": "test2"},
            {"content": "Third document", "source": "test3"}
        ]
        
        for doc in documents:
            await vector_retriever.add_document(doc)
        
        assert len(vector_retriever.documents) == 3
        assert len(vector_retriever.embeddings) == 3
        
        # Each document should have a corresponding embedding
        for i, doc in enumerate(documents):
            assert vector_retriever.documents[i] == doc
            assert isinstance(vector_retriever.embeddings[i], np.ndarray)
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_documents(self, vector_retriever):
        """Test retrieving from empty document store."""
        results = await vector_retriever.retrieve("test query")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_retrieve_with_documents(self, vector_retriever, sample_documents):
        """Test retrieving documents with similarity."""
        # Add documents
        for doc in sample_documents:
            await vector_retriever.add_document(doc)
        
        # Query for Paris-related content
        results = await vector_retriever.retrieve("Paris France")
        
        assert len(results) > 0
        assert len(results) <= 5  # Default limit
        
        # Check result structure
        for result in results:
            assert "content" in result
            assert "relevance_score" in result
            assert isinstance(result["relevance_score"], float)
        
        # Results should be sorted by relevance (highest first)
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_retrieve_with_limit(self, vector_retriever, sample_documents):
        """Test retrieving with custom limit."""
        # Add documents
        for doc in sample_documents:
            await vector_retriever.add_document(doc)
        
        # Query with limit
        results = await vector_retriever.retrieve("text", limit=2)
        
        assert len(results) <= 2
    
    @pytest.mark.asyncio
    async def test_retrieve_relevance_scoring(self, vector_retriever):
        """Test that relevance scoring works correctly."""
        # Add documents with different relevance to query
        documents = [
            {"content": "Python programming language", "source": "high_relevance"},
            {"content": "Java programming language", "source": "medium_relevance"},
            {"content": "Cooking recipes", "source": "low_relevance"}
        ]
        
        for doc in documents:
            await vector_retriever.add_document(doc)
        
        # Query for Python-related content
        results = await vector_retriever.retrieve("Python")
        
        assert len(results) == 3
        
        # The Python document should have highest relevance
        assert results[0]["source"] == "high_relevance"
        
        # Scores should be in descending order
        scores = [r["relevance_score"] for r in results]
        assert scores[0] >= scores[1] >= scores[2]
    
    @pytest.mark.asyncio
    async def test_retrieve_preserves_original_document(self, vector_retriever):
        """Test that retrieved documents preserve original data."""
        original_doc = {
            "content": "Test content",
            "source": "test_source",
            "metadata": {"key": "value"},
            "id": "12345"
        }
        
        await vector_retriever.add_document(original_doc)
        results = await vector_retriever.retrieve("test")
        
        assert len(results) == 1
        result = results[0]
        
        # Should contain all original fields plus relevance score
        assert result["content"] == original_doc["content"]
        assert result["source"] == original_doc["source"]
        assert result["metadata"] == original_doc["metadata"]
        assert result["id"] == original_doc["id"]
        assert "relevance_score" in result
    
    @pytest.mark.asyncio
    async def test_custom_embedding_model(self):
        """Test using a custom embedding model."""
        # Mock embedding model
        def mock_embedding(text):
            # Simple mock: return length-based embedding
            return np.array([len(text)] * 128) / 128
        
        retriever = VectorRetriever(embedding_model=mock_embedding)
        
        # Add documents
        await retriever.add_document({"content": "short", "id": "1"})
        await retriever.add_document({"content": "this is a longer text", "id": "2"})
        
        # Query should find longer text for longer query
        results = await retriever.retrieve("this is a longer query text")
        
        assert len(results) == 2
        # Longer text should have higher similarity
        assert results[0]["id"] == "2"
    
    @pytest.mark.asyncio
    async def test_empty_content_document(self, vector_retriever):
        """Test handling document with empty content."""
        document = {"content": "", "source": "empty"}
        
        await vector_retriever.add_document(document)
        
        assert len(vector_retriever.documents) == 1
        assert len(vector_retriever.embeddings) == 1
        
        # Should still be retrievable
        results = await vector_retriever.retrieve("test")
        assert len(results) == 1
    
    @pytest.mark.asyncio
    async def test_document_without_content_field(self, vector_retriever):
        """Test handling document without content field."""
        document = {"source": "no_content", "data": "some data"}
        
        await vector_retriever.add_document(document)
        
        assert len(vector_retriever.documents) == 1
        assert len(vector_retriever.embeddings) == 1
        
        # Should use empty string for content
        results = await vector_retriever.retrieve("test")
        assert len(results) == 1
    
    def test_embedding_consistency(self, vector_retriever):
        """Test that embedding function produces consistent results."""
        text = "consistent test text"
        
        embedding1 = vector_retriever._simple_embedding(text)
        embedding2 = vector_retriever._simple_embedding(text)
        
        # Should produce identical embeddings for same text
        assert np.allclose(embedding1, embedding2)
    
    @pytest.mark.asyncio
    async def test_large_document_set(self, vector_retriever):
        """Test retrieval with larger document set."""
        # Add many documents
        for i in range(100):
            await vector_retriever.add_document({
                "content": f"Document {i} with some content",
                "id": str(i)
            })
        
        assert len(vector_retriever.documents) == 100
        assert len(vector_retriever.embeddings) == 100
        
        # Should still retrieve correctly
        results = await vector_retriever.retrieve("document", limit=10)
        assert len(results) == 10
        
        # All should have relevance scores
        for result in results:
            assert "relevance_score" in result
            assert isinstance(result["relevance_score"], float)
    
    @pytest.mark.asyncio
    async def test_unicode_content(self, vector_retriever):
        """Test handling of unicode content."""
        unicode_doc = {
            "content": "Document with unicode: 你好世界 🌍 café naïve",
            "source": "unicode_test"
        }
        
        await vector_retriever.add_document(unicode_doc)
        
        results = await vector_retriever.retrieve("unicode 你好")
        assert len(results) == 1
        assert results[0]["source"] == "unicode_test"