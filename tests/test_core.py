# tests/test_core.py
"""Tests for core ContextEngine functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from contextforge.core import ContextEngine, Context, Message, Chain
from contextforge.memory import InMemoryStore


class TestMessage:
    """Test Message class."""
    
    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}
    
    def test_message_with_metadata(self):
        """Test creating a message with metadata."""
        metadata = {"source": "test", "priority": "high"}
        msg = Message(role="assistant", content="Response", metadata=metadata)
        assert msg.metadata == metadata


class TestContext:
    """Test Context class."""
    
    def test_context_creation(self):
        """Test creating a context."""
        context = Context(
            system_prompt="You are helpful",
            instructions=["Be concise", "Be accurate"]
        )
        assert context.system_prompt == "You are helpful"
        assert context.instructions == ["Be concise", "Be accurate"]
        assert context.messages == []
    
    def test_context_to_messages_basic(self):
        """Test converting context to messages."""
        context = Context(
            system_prompt="You are helpful",
            instructions=["Be concise"],
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!")
            ]
        )
        
        messages = context.to_messages()
        assert len(messages) == 3  # system + 2 conversation messages
        assert messages[0]["role"] == "system"
        assert "You are helpful" in messages[0]["content"]
        assert "Be concise" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi there!"
    
    def test_context_to_messages_with_memory(self):
        """Test context with memory components."""
        context = Context(
            system_prompt="You are helpful",
            long_term_memory=[{"content": "User likes coffee"}],
            short_term_memory=[{"content": "Recent chat about weather"}],
            retrieved_info=[{"content": "Weather info", "source": "API"}],
            messages=[Message(role="user", content="What's the weather?")]
        )
        
        messages = context.to_messages()
        system_msg = messages[0]["content"]
        assert "Long-term Memory" in system_msg
        assert "User likes coffee" in system_msg
        
        user_msg = messages[1]["content"]
        assert "Recent Context" in user_msg
        assert "Retrieved Information" in user_msg
        assert "What's the weather?" in user_msg
    
    def test_context_with_tools(self):
        """Test context with available tools."""
        context = Context(
            system_prompt="You are helpful",
            available_tools=[
                {"name": "search", "description": "Search the web"},
                {"name": "calculate", "description": "Do math"}
            ]
        )
        
        messages = context.to_messages()
        system_msg = messages[0]["content"]
        assert "Available Tools" in system_msg
        assert "search" in system_msg
        assert "calculate" in system_msg
    
    def test_context_with_output_schema(self):
        """Test context with output schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
        context = Context(
            system_prompt="You are helpful",
            output_schema=schema
        )
        
        messages = context.to_messages()
        system_msg = messages[0]["content"]
        assert "Output Schema" in system_msg
        assert "object" in system_msg


class TestContextEngine:
    """Test ContextEngine class."""
    
    @pytest.mark.asyncio
    async def test_context_engine_creation(self, mock_provider):
        """Test creating a ContextEngine."""
        engine = ContextEngine(
            provider=mock_provider,
            default_system_prompt="Test prompt"
        )
        assert engine.provider == mock_provider
        assert engine.default_system_prompt == "Test prompt"
        assert isinstance(engine.memory_store, InMemoryStore)
        assert engine.sessions == {}
    
    @pytest.mark.asyncio
    async def test_create_session(self, context_engine):
        """Test creating a conversation session."""
        session = context_engine.create_session("test_session")
        assert session.session_id == "test_session"
        assert "test_session" in context_engine.sessions
    
    @pytest.mark.asyncio
    async def test_get_session(self, context_engine):
        """Test getting or creating a session."""
        # Get non-existent session (should create)
        session1 = context_engine.get_session("new_session")
        assert session1.session_id == "new_session"
        
        # Get existing session
        session2 = context_engine.get_session("new_session")
        assert session1 is session2
    
    @pytest.mark.asyncio
    async def test_basic_generation(self, context_engine):
        """Test basic text generation."""
        response = await context_engine.generate("Hello")
        assert response == "Test response"
        
        # Check that provider was called
        context_engine.provider.generate.assert_called_once()
        args, kwargs = context_engine.provider.generate.call_args
        messages = args[0]
        assert len(messages) >= 2  # system + user message
        assert messages[-1]["content"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_generation_with_session(self, context_engine):
        """Test generation with session management."""
        # First message
        await context_engine.generate("Hello", session_id="test_session")
        session = context_engine.get_session("test_session")
        assert len(session.messages) == 2  # user + assistant
        
        # Second message should include conversation history
        context_engine.provider.generate.reset_mock()
        await context_engine.generate("How are you?", session_id="test_session")
        
        # Check that conversation history was included
        args, kwargs = context_engine.provider.generate.call_args
        messages = args[0]
        # Should have system + previous conversation + new message
        assert len(messages) >= 4
    
    @pytest.mark.asyncio
    async def test_generation_with_custom_system_prompt(self, context_engine):
        """Test generation with custom system prompt."""
        custom_prompt = "You are a specialized assistant."
        await context_engine.generate(
            "Hello", 
            system_prompt=custom_prompt
        )
        
        args, kwargs = context_engine.provider.generate.call_args
        messages = args[0]
        assert custom_prompt in messages[0]["content"]
    
    @pytest.mark.asyncio
    async def test_generation_with_instructions(self, context_engine):
        """Test generation with additional instructions."""
        instructions = ["Be concise", "Use bullet points"]
        await context_engine.generate(
            "Explain Python",
            instructions=instructions
        )
        
        args, kwargs = context_engine.provider.generate.call_args
        messages = args[0]
        system_msg = messages[0]["content"]
        assert "Be concise" in system_msg
        assert "Use bullet points" in system_msg
    
    @pytest.mark.asyncio
    async def test_generation_with_retrieval(self, context_engine):
        """Test generation with retrieval."""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.retrieve = AsyncMock(return_value=[
            {"content": "Retrieved info", "source": "test"}
        ])
        context_engine.retriever = mock_retriever
        
        await context_engine.generate("What is Python?", retrieve=True)
        
        # Check retriever was called
        mock_retriever.retrieve.assert_called_once_with("What is Python?")
        
        # Check retrieved info was included
        args, kwargs = context_engine.provider.generate.call_args
        messages = args[0]
        user_msg = messages[-1]["content"]
        assert "Retrieved Information" in user_msg
        assert "Retrieved info" in user_msg
    
    @pytest.mark.asyncio
    async def test_generation_with_tools(self, context_engine, tool_registry):
        """Test generation with tools."""
        context_engine.tool_registry = tool_registry
        
        await context_engine.generate("Use a tool", use_tools=True)
        
        args, kwargs = context_engine.provider.generate.call_args
        messages = args[0]
        system_msg = messages[0]["content"]
        assert "Available Tools" in system_msg
        assert "test_tool" in system_msg
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self, context_engine):
        """Test streaming generation."""
        # Mock streaming response
        async def mock_stream():
            for chunk in ["Hello", " ", "world", "!"]:
                yield chunk
        
        context_engine.provider.generate = AsyncMock(return_value=mock_stream())
        
        response_chunks = []
        async for chunk in await context_engine.generate("Hello", stream=True):
            response_chunks.append(chunk)
        
        assert response_chunks == ["Hello", " ", "world", "!"]
        
        # Check that full response was saved to session
        session = context_engine.get_session("default")
        assert len(session.messages) == 2
        assert session.messages[-1].content == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self, context_engine):
        """Test that memories are saved."""
        # Mock memory store
        mock_memory = Mock()
        mock_memory.get_relevant_memories = AsyncMock(return_value=[])
        mock_memory.add_memory = AsyncMock()
        context_engine.memory_store = mock_memory
        
        # Generate with a significant prompt (>50 chars)
        long_prompt = "This is a long prompt that should be saved to memory because it's important"
        await context_engine.generate(long_prompt)
        
        # Check memory was saved
        mock_memory.add_memory.assert_called_once()
        call_args = mock_memory.add_memory.call_args[0][0]
        assert call_args["content"] == long_prompt
        assert call_args["response"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_output_schema(self, context_engine):
        """Test generation with output schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
        
        await context_engine.generate(
            "Generate a person",
            output_schema=schema
        )
        
        args, kwargs = context_engine.provider.generate.call_args
        messages = args[0]
        system_msg = messages[0]["content"]
        assert "Output Schema" in system_msg
        assert "object" in system_msg


class TestChain:
    """Test Chain class."""
    
    @pytest.mark.asyncio
    async def test_chain_creation(self, context_engine):
        """Test creating a chain."""
        async def op1(x):
            return x + 1
        
        async def op2(x):
            return x * 2
        
        chain = context_engine.chain(op1, op2)
        assert isinstance(chain, Chain)
        assert len(chain.operations) == 2
    
    @pytest.mark.asyncio
    async def test_chain_execution(self, context_engine):
        """Test executing a chain."""
        async def add_one(x):
            return x + 1
        
        async def multiply_two(x):
            return x * 2
        
        chain = context_engine.chain(add_one, multiply_two)
        result = await chain.run(5)
        assert result == 12  # (5 + 1) * 2
    
    @pytest.mark.asyncio
    async def test_chain_with_dict_input(self, context_engine):
        """Test chain with dictionary input."""
        async def extract_value(data):
            return data["value"]
        
        async def double_value(x):
            return x * 2
        
        chain = context_engine.chain(extract_value, double_value)
        result = await chain.run({"value": 10})
        assert result == 20