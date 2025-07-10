# tests/test_memory.py
"""Tests for memory management functionality."""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from contextforge.memory import (
    MemoryStore, InMemoryStore, SQLiteMemoryStore, ConversationMemory
)
from contextforge.core import Message


class TestMemoryStore:
    """Test abstract MemoryStore class."""
    
    def test_memory_store_abstract(self):
        """Test that MemoryStore is abstract."""
        with pytest.raises(TypeError):
            MemoryStore()


class TestInMemoryStore:
    """Test InMemoryStore implementation."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        store = InMemoryStore()
        assert store.memories.maxlen == 1000
        assert store.default_ttl == timedelta(hours=24)
    
    def test_init_custom(self):
        """Test initialization with custom values."""
        store = InMemoryStore(
            max_memories=500,
            default_ttl=timedelta(hours=12)
        )
        assert store.memories.maxlen == 500
        assert store.default_ttl == timedelta(hours=12)
    
    @pytest.mark.asyncio
    async def test_add_memory(self, memory_store):
        """Test adding a memory."""
        memory = {"content": "Test memory", "type": "note"}
        await memory_store.add_memory(memory)
        
        assert len(memory_store.memories) == 1
        stored = memory_store.memories[0]
        assert stored["content"] == "Test memory"
        assert stored["type"] == "note"
        assert "expires_at" in stored
        assert isinstance(stored["expires_at"], datetime)
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories(self, memory_store):
        """Test retrieving relevant memories."""
        # Add some memories
        await memory_store.add_memory({"content": "Python is great"})
        await memory_store.add_memory({"content": "JavaScript is useful"})
        await memory_store.add_memory({"content": "Python frameworks are helpful"})
        
        # Search for Python-related memories
        results = await memory_store.get_relevant_memories("Python")
        assert len(results) == 2
        assert all("Python" in mem["content"] for mem in results)
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories_with_limit(self, memory_store):
        """Test retrieving memories with limit."""
        # Add multiple memories
        for i in range(10):
            await memory_store.add_memory({"content": f"Python memory {i}"})
        
        # Get limited results
        results = await memory_store.get_relevant_memories("Python", limit=3)
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories_empty(self, memory_store):
        """Test retrieving memories when none exist."""
        results = await memory_store.get_relevant_memories("test")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_clear_memories(self, memory_store):
        """Test clearing memories."""
        # Add memories
        await memory_store.add_memory({"content": "Memory 1"})
        await memory_store.add_memory({"content": "Memory 2"})
        assert len(memory_store.memories) == 2
        
        # Clear all memories
        await memory_store.clear_memories()
        assert len(memory_store.memories) == 0
    
    @pytest.mark.asyncio
    async def test_clear_memories_before_date(self, memory_store):
        """Test clearing memories before a specific date."""
        # Add old memory
        old_memory = {"content": "Old memory"}
        await memory_store.add_memory(old_memory)
        
        # Manually set expiration to past
        memory_store.memories[0]["expires_at"] = datetime.now() - timedelta(hours=1)
        
        # Add new memory
        await memory_store.add_memory({"content": "New memory"})
        
        # Clear old memories
        await memory_store.clear_memories(datetime.now())
        
        # Should have one memory left
        assert len(memory_store.memories) == 1
        assert memory_store.memories[0]["content"] == "New memory"
    
    @pytest.mark.asyncio
    async def test_expired_memories_not_retrieved(self, memory_store):
        """Test that expired memories are not retrieved."""
        # Add memory
        await memory_store.add_memory({"content": "Expired memory"})
        
        # Manually expire it
        memory_store.memories[0]["expires_at"] = datetime.now() - timedelta(hours=1)
        
        # Should not be retrieved
        results = await memory_store.get_relevant_memories("memory")
        assert len(results) == 0


class TestSQLiteMemoryStore:
    """Test SQLiteMemoryStore implementation."""
    
    def test_init_creates_db(self, temp_sqlite_db):
        """Test initialization creates database."""
        store = SQLiteMemoryStore(temp_sqlite_db)
        assert os.path.exists(temp_sqlite_db)
    
    @pytest.mark.asyncio
    async def test_add_memory(self, sqlite_memory_store):
        """Test adding a memory to SQLite."""
        memory = {
            "content": "Test memory",
            "type": "note",
            "importance": "high"
        }
        await sqlite_memory_store.add_memory(memory)
        
        # Verify in database
        import sqlite3
        conn = sqlite3.connect(sqlite_memory_store.db_path)
        cursor = conn.execute("SELECT content, metadata FROM memories")
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "Test memory"
        
        import json
        metadata = json.loads(row[1])
        assert metadata["type"] == "note"
        assert metadata["importance"] == "high"
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories(self, sqlite_memory_store):
        """Test retrieving relevant memories from SQLite."""
        # Add some memories
        await sqlite_memory_store.add_memory({"content": "Python is great"})
        await sqlite_memory_store.add_memory({"content": "JavaScript is useful"})
        await sqlite_memory_store.add_memory({"content": "Python frameworks"})
        
        # Search for Python-related memories
        results = await sqlite_memory_store.get_relevant_memories("Python")
        assert len(results) == 2
        assert all("Python" in mem["content"] for mem in results)
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories_with_limit(self, sqlite_memory_store):
        """Test retrieving memories with limit from SQLite."""
        # Add multiple memories
        for i in range(10):
            await sqlite_memory_store.add_memory({"content": f"Python memory {i}"})
        
        # Get limited results
        results = await sqlite_memory_store.get_relevant_memories("Python", limit=3)
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_clear_memories(self, sqlite_memory_store):
        """Test clearing memories from SQLite."""
        # Add memories
        await sqlite_memory_store.add_memory({"content": "Memory 1"})
        await sqlite_memory_store.add_memory({"content": "Memory 2"})
        
        # Clear all memories
        await sqlite_memory_store.clear_memories()
        
        # Verify database is empty
        import sqlite3
        conn = sqlite3.connect(sqlite_memory_store.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_clear_memories_before_date(self, sqlite_memory_store):
        """Test clearing memories before a specific date from SQLite."""
        # Add memory that will expire in the future (24 hours by default)
        await sqlite_memory_store.add_memory({"content": "Future memory"})
        
        # Add another memory and manually set it to expire in the past
        await sqlite_memory_store.add_memory({"content": "Past memory"})
        
        # Update the second memory to expire in the past
        import sqlite3
        conn = sqlite3.connect(sqlite_memory_store.db_path)
        past_date = (datetime.now() - timedelta(hours=1)).isoformat()
        conn.execute("UPDATE memories SET expires_at = ? WHERE content = ?", (past_date, "Past memory"))
        conn.commit()
        conn.close()
        
        # Clear memories that expire before now
        await sqlite_memory_store.clear_memories(datetime.now())
        
        # Should have one memory left (the future one)
        conn = sqlite3.connect(sqlite_memory_store.db_path)
        cursor = conn.execute("SELECT content FROM memories")
        rows = cursor.fetchall()
        conn.close()
        
        assert len(rows) == 1
        assert rows[0][0] == "Future memory"


class TestConversationMemory:
    """Test ConversationMemory implementation."""
    
    def test_init(self):
        """Test initialization."""
        memory = ConversationMemory("test_session")
        assert memory.session_id == "test_session"
        assert memory.messages.maxlen == 100
        assert memory.context_window == 10
        assert memory.metadata == {}
    
    def test_init_custom(self):
        """Test initialization with custom values."""
        memory = ConversationMemory("test", max_messages=50)
        assert memory.session_id == "test"
        assert memory.messages.maxlen == 50
    
    def test_add_message(self):
        """Test adding a message."""
        memory = ConversationMemory("test")
        message = Message(role="user", content="Hello")
        
        memory.add_message(message)
        assert len(memory.messages) == 1
        assert memory.messages[0] == message
    
    def test_get_messages(self):
        """Test getting messages."""
        memory = ConversationMemory("test")
        
        # Add messages
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="assistant", content="Hi")
        memory.add_message(msg1)
        memory.add_message(msg2)
        
        # Get all messages
        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0] == msg1
        assert messages[1] == msg2
    
    def test_get_messages_with_limit(self):
        """Test getting messages with limit."""
        memory = ConversationMemory("test")
        
        # Add multiple messages
        for i in range(5):
            msg = Message(role="user", content=f"Message {i}")
            memory.add_message(msg)
        
        # Get limited messages
        messages = memory.get_messages(limit=3)
        assert len(messages) == 3
        # Should get the last 3 messages
        assert messages[0].content == "Message 2"
        assert messages[1].content == "Message 3"
        assert messages[2].content == "Message 4"
    
    def test_get_recent_context(self):
        """Test getting recent context."""
        memory = ConversationMemory("test")
        
        # Add messages
        for i in range(15):
            msg = Message(role="user", content=f"Long message {i}" * 10)
            memory.add_message(msg)
        
        # Get recent context
        context = memory.get_recent_context()
        assert len(context) == 10  # context_window
        
        # Check structure
        assert all("role" in ctx for ctx in context)
        assert all("content" in ctx for ctx in context)
        assert all("timestamp" in ctx for ctx in context)
        
        # Check content is truncated
        assert all(len(ctx["content"]) <= 200 for ctx in context)
    
    def test_clear(self):
        """Test clearing conversation."""
        memory = ConversationMemory("test")
        
        # Add messages
        memory.add_message(Message(role="user", content="Hello"))
        memory.add_message(Message(role="assistant", content="Hi"))
        assert len(memory.messages) == 2
        
        # Clear
        memory.clear()
        assert len(memory.messages) == 0
    
    def test_summarize_empty(self):
        """Test summarizing empty conversation."""
        memory = ConversationMemory("test")
        summary = memory.summarize()
        assert summary == "No conversation history."
    
    def test_summarize_with_messages(self):
        """Test summarizing conversation with messages."""
        memory = ConversationMemory("test")
        
        # Add messages
        memory.add_message(Message(role="user", content="Hello there"))
        memory.add_message(Message(role="assistant", content="Hi"))
        memory.add_message(Message(role="user", content="How are you doing today"))
        
        summary = memory.summarize()
        assert "3 messages" in summary
        assert "Conversation with" in summary
    
    def test_summarize_with_keywords(self):
        """Test summarizing extracts keywords."""
        memory = ConversationMemory("test")
        
        # Add messages with repeated keywords
        memory.add_message(Message(role="user", content="I love Python programming"))
        memory.add_message(Message(role="user", content="Python is great for machine learning"))
        memory.add_message(Message(role="user", content="Programming in Python is fun"))
        
        summary = memory.summarize()
        assert "Python" in summary or "programming" in summary
        assert "Key topics" in summary
    
    def test_message_limit_enforcement(self):
        """Test that message limit is enforced."""
        memory = ConversationMemory("test", max_messages=3)
        
        # Add more messages than limit
        for i in range(5):
            memory.add_message(Message(role="user", content=f"Message {i}"))
        
        # Should only have 3 messages (the last 3)
        messages = memory.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "Message 2"
        assert messages[1].content == "Message 3"
        assert messages[2].content == "Message 4"