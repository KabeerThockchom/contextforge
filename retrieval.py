# contextforge/retrieval.py
"""Retrieval components for RAG."""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import numpy as np


class Retriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    async def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        pass
    
    @abstractmethod
    async def add_document(self, document: Dict[str, Any]) -> None:
        """Add a document to the retrieval index."""
        pass


class VectorRetriever(Retriever):
    """Simple vector-based retriever using embeddings."""
    
    def __init__(self, embedding_model: Optional[Callable] = None):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
        self.embedding_model = embedding_model or self._simple_embedding
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple embedding using character frequencies (for demo)."""
        # In production, use proper embeddings (OpenAI, sentence-transformers, etc.)
        embedding = np.zeros(128)
        for i, char in enumerate(text[:128]):
            embedding[i % 128] += ord(char) / 1000
        return embedding / np.linalg.norm(embedding)
    
    async def add_document(self, document: Dict[str, Any]) -> None:
        """Add document with its embedding."""
        content = document.get("content", "")
        embedding = self.embedding_model(content)
        
        self.documents.append(document)
        self.embeddings.append(embedding)
    
    async def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents by similarity."""
        if not self.documents:
            return []
        
        query_embedding = self.embedding_model(query)
        
        # Compute similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Return top documents
        results = []
        for similarity, idx in similarities[:limit]:
            doc = self.documents[idx].copy()
            doc["relevance_score"] = float(similarity)
            results.append(doc)
        
        return results

