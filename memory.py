# contextforge/memory.py
"""Memory management for context engineering."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import json
import sqlite3
import asyncio
from abc import ABC, abstractmethod


class MemoryStore(ABC):
    """Abstract base class for memory stores."""
    
    @abstractmethod
    async def add_memory(self, memory: Dict[str, Any]) -> None:
        """Add a memory to the store."""
        pass
    
    @abstractmethod
    async def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query."""
        pass
    
    @abstractmethod
    async def clear_memories(self, before: Optional[datetime] = None) -> None:
        """Clear memories, optionally before a certain date."""
        pass


class InMemoryStore(MemoryStore):
    """Simple in-memory store with TTL support."""
    
    def __init__(self, max_memories: int = 1000, default_ttl: timedelta = timedelta(hours=24)):
        self.memories: deque = deque(maxlen=max_memories)
        self.default_ttl = default_ttl
    
    async def add_memory(self, memory: Dict[str, Any]) -> None:
        """Add a memory with TTL."""
        memory["expires_at"] = datetime.now() + self.default_ttl
        self.memories.append(memory)
    
    async def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval."""
        now = datetime.now()
        relevant = []
        
        for memory in self.memories:
            if memory.get("expires_at", now) > now:
                # Simple relevance: check if query words appear in content
                content = str(memory.get("content", "")).lower()
                if any(word.lower() in content for word in query.split()):
                    relevant.append(memory)
        
        return relevant[:limit]
    
    async def clear_memories(self, before: Optional[datetime] = None) -> None:
        """Clear expired memories."""
        if before is None:
            before = datetime.now()
        
        self.memories = deque(
            (m for m in self.memories if m.get("expires_at", before) > before),
            maxlen=self.memories.maxlen
        )


class SQLiteMemoryStore(MemoryStore):
    """SQLite-based persistent memory store."""
    
    def __init__(self, db_path: str = "memories.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON memories(expires_at)
        """)
        conn.commit()
        conn.close()
    
    async def add_memory(self, memory: Dict[str, Any]) -> None:
        """Add memory to SQLite."""
        conn = sqlite3.connect(self.db_path)
        metadata = {k: v for k, v in memory.items() if k != "content"}
        conn.execute(
            "INSERT INTO memories (content, metadata, expires_at) VALUES (?, ?, ?)",
            (
                memory.get("content", ""),
                json.dumps(metadata),
                (datetime.now() + timedelta(hours=24)).isoformat()
            )
        )
        conn.commit()
        conn.close()
    
    async def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories using FTS or simple LIKE."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            SELECT content, metadata FROM memories 
            WHERE expires_at > datetime('now') 
            AND content LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (f"%{query}%", limit)
        )
        
        memories = []
        for row in cursor:
            memory = {"content": row[0]}
            memory.update(json.loads(row[1]))
            memories.append(memory)
        
        conn.close()
        return memories
    
    async def clear_memories(self, before: Optional[datetime] = None) -> None:
        """Clear expired memories."""
        conn = sqlite3.connect(self.db_path)
        if before:
            conn.execute("DELETE FROM memories WHERE expires_at < ?", (before.isoformat(),))
        else:
            conn.execute("DELETE FROM memories WHERE expires_at < datetime('now')")
        conn.commit()
        conn.close()


class ConversationMemory:
    """Manages conversation history for a session."""
    
    def __init__(self, session_id: str, max_messages: int = 100):
        self.session_id = session_id
        self.messages: deque = deque(maxlen=max_messages)
        self.context_window = 10  # Recent messages to include
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, message: 'Message') -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
    
    def get_messages(self, limit: Optional[int] = None) -> List['Message']:
        """Get conversation messages."""
        if limit:
            return list(self.messages)[-limit:]
        return list(self.messages)
    
    def get_recent_context(self) -> List[Dict[str, Any]]:
        """Get recent conversation context."""
        recent = list(self.messages)[-self.context_window:]
        return [
            {
                "role": msg.role,
                "content": msg.content[:200],  # Truncate for context
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in recent
        ]
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
    
    def summarize(self) -> str:
        """Create a summary of the conversation."""
        if not self.messages:
            return "No conversation history."
        
        summary_parts = [f"Conversation with {len(self.messages)} messages:"]
        
        # Get key topics (simple word frequency)
        words = {}
        for msg in self.messages:
            if msg.role == "user":
                for word in msg.content.split():
                    if len(word) > 4:  # Skip short words
                        words[word.lower()] = words.get(word.lower(), 0) + 1
        
        top_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_words:
            summary_parts.append(f"Key topics: {', '.join(w[0] for w in top_words)}")
        
        return " ".join(summary_parts)

