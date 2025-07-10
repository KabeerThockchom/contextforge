# contextforge/__init__.py
"""
ContextForge: A Python package for building LLM applications with sophisticated context engineering.
"""

from typing import AsyncIterator
from .core import ContextEngine, Context, Message
from .memory import MemoryStore, ConversationMemory, InMemoryStore, SQLiteMemoryStore
from .providers import Provider, create_provider
from .tools import Tool, ToolRegistry
from .retrieval import Retriever, VectorRetriever

__version__ = "0.1.0"
__all__ = [
    "ContextEngine",
    "Context",
    "Message",
    "MemoryStore",
    "ConversationMemory",
    "InMemoryStore",
    "SQLiteMemoryStore",
    "Provider",
    "create_provider",
    "Tool",
    "ToolRegistry",
    "Retriever",
    "VectorRetriever",
    "AsyncIterator"
]