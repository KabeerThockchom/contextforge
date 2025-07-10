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
        text_lower = text.lower()
        
        # Simple approach: use word hashes for embedding
        words = text_lower.split()
        for i, word in enumerate(words[:128]):
            # Hash each word to a position in the embedding
            hash_val = hash(word) % 128
            embedding[hash_val] += 1
            
        # Also include character-based features for robustness
        for i, char in enumerate(text_lower[:64]):
            embedding[64 + (i % 64)] += ord(char) / 1000
            
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
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

