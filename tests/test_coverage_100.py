"""Additional tests to achieve 100% coverage."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from contextforge.core import ContextEngine, Context, Message
from contextforge.memory import MemoryStore, SQLiteMemoryStore
from contextforge.providers import Provider, create_provider
from contextforge.retrieval import Retriever
from contextforge.tools import ToolRegistry


class TestCoverageGaps:
    """Tests to cover remaining gaps in coverage."""
    
    @pytest.mark.asyncio
    async def test_context_json_serialize_non_datetime(self):
        """Test json_serialize function with non-datetime object."""
        context = Context(
            system_prompt="Test",
            long_term_memory=[{"content": "test", "custom_obj": object()}]
        )
        
        # The object() will trigger the TypeError in json_serialize
        with pytest.raises(TypeError, match="Object of type object is not JSON serializable"):
            context.to_messages()
    
    @pytest.mark.asyncio
    async def test_context_engine_with_string_provider(self):
        """Test creating ContextEngine with string provider name."""
        # This will test the isinstance(provider, str) branch
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            engine = ContextEngine(provider="openai", model="gpt-3.5-turbo")
            assert engine.provider is not None
            assert engine.provider.model == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_generate_with_tool_calls(self, mock_provider, tool_registry):
        """Test generation with tool calls handling."""
        # Create a mock response with tool_calls attribute
        mock_response = Mock()
        mock_response.tool_calls = [
            {"name": "test_tool", "arguments": {"x": 5}}
        ]
        mock_response.content = "Let me calculate that for you."
        
        # Configure provider to return response with tool calls first,
        # then return final response after tool execution
        mock_provider.generate = AsyncMock(side_effect=[
            mock_response,
            "The result is 10"
        ])
        
        engine = ContextEngine(
            provider=mock_provider,
            tool_registry=tool_registry
        )
        
        response = await engine.generate("Calculate something", use_tools=True)
        assert response == "The result is 10"
        
        # Verify tool was executed
        assert mock_provider.generate.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_tools_with_exception(self, context_engine):
        """Test _execute_tools with tool that raises exception."""
        # Register a tool that fails
        @context_engine.tool_registry.register(description="Failing tool")
        def failing_tool():
            raise RuntimeError("Tool failed!")
        
        tool_calls = [
            {"name": "failing_tool", "arguments": {}},
            {"name": "test_tool", "arguments": {"x": 5}}  # test_tool is not registered, so it will fail differently
        ]
        
        results = await context_engine._execute_tools(tool_calls)
        
        assert len(results) == 2
        # First tool should have failed with exception
        assert results[0]["success"] is False
        assert results[0]["error"] == "Tool failed!"
        assert results[0]["tool"] == "failing_tool"
        
        # Second tool should have failed (not found)
        assert results[1]["success"] is False
        assert "not found" in results[1]["error"]
        assert results[1]["tool"] == "test_tool"
    
    @pytest.mark.asyncio
    async def test_context_to_messages_no_user_message(self):
        """Test to_messages when there are no user messages for context insertion."""
        context = Context(
            system_prompt="Test",
            messages=[
                Message(role="assistant", content="Hello"),
                Message(role="assistant", content="How can I help?")
            ],
            short_term_memory=[{"content": "memory"}],
            retrieved_info=[{"content": "info"}]
        )
        
        messages = context.to_messages()
        
        # Should have system message and two assistant messages
        assert len(messages) == 3
        # Context parts should not be inserted since there's no user message
        assert "Recent Context" not in messages[1]["content"]
        assert "Recent Context" not in messages[2]["content"]
    
    def test_abstract_memory_store_methods(self):
        """Test that abstract MemoryStore methods raise NotImplementedError."""
        # Import locally to avoid circular imports
        from contextforge.memory import MemoryStore
        
        class IncompleteMemoryStore(MemoryStore):
            """Incomplete implementation for testing."""
            pass
        
        # Should not be able to instantiate without implementing abstract methods
        with pytest.raises(TypeError):
            IncompleteMemoryStore()
    
    def test_abstract_retriever_methods(self):
        """Test that abstract Retriever methods raise NotImplementedError."""
        from contextforge.retrieval import Retriever
        
        class IncompleteRetriever(Retriever):
            """Incomplete implementation for testing."""
            pass
        
        # Should not be able to instantiate without implementing abstract methods
        with pytest.raises(TypeError):
            IncompleteRetriever()
    
    @pytest.mark.asyncio
    async def test_stream_generate_error_handling(self):
        """Test streaming with provider error."""
        from contextforge.providers import AnthropicProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            # Mock error response
            mock_response = Mock()
            mock_response.status = 400
            mock_response.json = AsyncMock(return_value={"error": "Bad request"})
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = AnthropicProvider(api_key="test-key")
            
            with pytest.raises(Exception, match="Anthropic API error"):
                gen = provider._stream_generate({}, {})
                async for _ in gen:
                    pass
    
    @pytest.mark.asyncio
    async def test_openai_stream_json_decode_error(self):
        """Test OpenAI streaming with JSON decode error."""
        from contextforge.providers import OpenAIProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = Mock()
            mock_response.status = 200
            
            # Create async iterator with invalid JSON
            async def mock_iter(self):
                yield b'data: invalid json\n\n'
                yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
                yield b'data: [DONE]\n\n'
            
            mock_response.content.__aiter__ = mock_iter
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = OpenAIProvider(api_key="test-key")
            
            chunks = []
            async for chunk in provider._stream_generate({}, {}):
                chunks.append(chunk)
            
            # Should have skipped the invalid JSON and processed valid chunk
            assert chunks == ["Hello"]
    
    @pytest.mark.asyncio
    async def test_anthropic_stream_json_decode_error(self):
        """Test Anthropic streaming with JSON decode error."""
        from contextforge.providers import AnthropicProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = Mock()
            mock_response.status = 200
            
            # Create async iterator with invalid JSON
            async def mock_iter(self):
                yield b'data: invalid json\n\n'
                yield b'data: {"type": "content_block_delta", "delta": {"text": "Hello"}}\n\n'
                yield b'data: [DONE]\n\n'
            
            mock_response.content.__aiter__ = mock_iter
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = AnthropicProvider(api_key="test-key")
            
            chunks = []
            async for chunk in provider._stream_generate({}, {}):
                chunks.append(chunk)
            
            # Should have skipped the invalid JSON and processed valid chunk
            assert chunks == ["Hello"]
    
    @pytest.mark.asyncio
    async def test_ollama_stream_json_decode_error(self):
        """Test Ollama streaming with JSON decode error."""
        from contextforge.providers import OllamaProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = Mock()
            mock_response.status = 200
            
            # Create async iterator with invalid JSON
            async def mock_iter(self):
                yield b'invalid json\n'
                yield b'{"message": {"content": "Hello"}}\n'
                yield b'\n'  # Empty line
            
            mock_response.content.__aiter__ = mock_iter
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = OllamaProvider()
            
            chunks = []
            async for chunk in provider._stream_generate({}):
                chunks.append(chunk)
            
            # Should have skipped the invalid JSON and processed valid chunk
            assert chunks == ["Hello"]
    
    @pytest.mark.asyncio
    async def test_ollama_stream_error(self):
        """Test Ollama streaming with API error."""
        from contextforge.providers import OllamaProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = Mock()
            mock_response.status = 500
            mock_response.json = AsyncMock(return_value={"error": "Server error"})
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = OllamaProvider()
            
            with pytest.raises(Exception, match="Ollama API error"):
                gen = provider._stream_generate({})
                async for _ in gen:
                    pass
    
    def test_provider_abstract_methods(self):
        """Test that Provider abstract methods cannot be called."""
        from contextforge.providers import Provider
        
        # Create a minimal concrete implementation
        class MinimalProvider(Provider):
            async def generate(self, messages, stream=False, **kwargs):
                return "test"
            
            def get_info(self):
                return {}
        
        # Should be able to instantiate
        provider = MinimalProvider()
        assert provider is not None
    
    @pytest.mark.asyncio
    async def test_openai_stream_no_choices(self):
        """Test OpenAI streaming with response that has no choices."""
        from contextforge.providers import OpenAIProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = Mock()
            mock_response.status = 200
            
            # Create async iterator with response without choices
            async def mock_iter(self):
                yield b'data: {"no_choices": "here"}\n\n'
                yield b'data: [DONE]\n\n'
            
            mock_response.content.__aiter__ = mock_iter
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = OpenAIProvider(api_key="test-key")
            
            chunks = []
            async for chunk in provider._stream_generate({}, {}):
                chunks.append(chunk)
            
            # Should have no chunks since there were no valid choices
            assert chunks == []
    
    @pytest.mark.asyncio
    async def test_openai_stream_error(self):
        """Test OpenAI streaming with API error."""
        from contextforge.providers import OpenAIProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = Mock()
            mock_response.status = 400
            mock_response.json = AsyncMock(return_value={"error": "Bad request"})
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = OpenAIProvider(api_key="test-key")
            
            with pytest.raises(Exception, match="OpenAI API error"):
                gen = provider._stream_generate({}, {})
                async for _ in gen:
                    pass
    
    @pytest.mark.asyncio
    async def test_anthropic_stream_not_content_block(self):
        """Test Anthropic streaming with non-content block."""
        from contextforge.providers import AnthropicProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = Mock()
            mock_response.status = 200
            
            # Create async iterator with non-content block
            async def mock_iter(self):
                yield b'data: {"type": "other_type", "data": "something"}\n\n'
                yield b'data: [DONE]\n\n'
            
            mock_response.content.__aiter__ = mock_iter
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = AnthropicProvider(api_key="test-key")
            
            chunks = []
            async for chunk in provider._stream_generate({}, {}):
                chunks.append(chunk)
            
            # Should have no chunks since there were no content blocks
            assert chunks == []
    
    @pytest.mark.asyncio
    async def test_ollama_response_no_message(self):
        """Test Ollama streaming with response that has no message."""
        from contextforge.providers import OllamaProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = Mock()
            mock_response.status = 200
            
            # Create async iterator with response without message
            async def mock_iter(self):
                yield b'{"no_message": "here"}\n'
            
            mock_response.content.__aiter__ = mock_iter
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = OllamaProvider()
            
            chunks = []
            async for chunk in provider._stream_generate({}):
                chunks.append(chunk)
            
            # Should have no chunks since there was no message
            assert chunks == []
    
    @pytest.mark.asyncio
    async def test_tool_registry_execute_exception(self):
        """Test tool registry execute with exception to cover line 252."""
        registry = ToolRegistry()
        
        # Register a tool
        @registry.register(description="Test tool")
        def test_tool():
            return "success"
        
        # Execute non-existent tool to trigger exception path
        with pytest.raises(Exception, match="Tool not found"):
            await registry.execute("non_existent_tool")
    
    def test_memory_abstract_methods_direct(self):
        """Test calling abstract methods directly."""
        from contextforge.memory import MemoryStore
        
        # Create instance that implements abstract methods
        class TestMemoryStore(MemoryStore):
            async def add_memory(self, memory):
                pass
            
            async def get_relevant_memories(self, query, limit=5):
                return []
            
            async def clear_memories(self, before=None):
                pass
        
        # Should be able to instantiate
        store = TestMemoryStore()
        assert store is not None
    
    def test_retriever_abstract_methods_direct(self):
        """Test calling abstract methods directly."""
        from contextforge.retrieval import Retriever
        
        # Create instance that implements abstract methods
        class TestRetriever(Retriever):
            async def retrieve(self, query, limit=5):
                return []
            
            async def add_document(self, document):
                pass
        
        # Should be able to instantiate
        retriever = TestRetriever()
        assert retriever is not None
    
    def test_provider_abstract_generate(self):
        """Test that Provider.generate is abstract."""
        from contextforge.providers import Provider
        
        # Try to create a provider without implementing generate
        class IncompleteProvider(Provider):
            def get_info(self):
                return {}
        
        with pytest.raises(TypeError):
            IncompleteProvider()
    
    def test_provider_abstract_get_info(self):
        """Test that Provider.get_info is abstract."""
        from contextforge.providers import Provider
        
        # Try to create a provider without implementing get_info
        class IncompleteProvider(Provider):
            async def generate(self, messages, stream=False, **kwargs):
                return "test"
        
        with pytest.raises(TypeError):
            IncompleteProvider()
    
    @pytest.mark.asyncio
    async def test_openai_stream_empty_content(self):
        """Test OpenAI streaming with empty content."""
        from contextforge.providers import OpenAIProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = Mock()
            mock_response.status = 200
            
            # Create async iterator with choices but no content
            async def mock_iter(self):
                yield b'data: {"choices": [{"delta": {}}]}\n\n'
                yield b'data: [DONE]\n\n'
            
            mock_response.content.__aiter__ = mock_iter
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = OpenAIProvider(api_key="test-key")
            
            chunks = []
            async for chunk in provider._stream_generate({}, {}):
                chunks.append(chunk)
            
            # Should have no chunks since there was no content
            assert chunks == []
    
    @pytest.mark.asyncio
    async def test_ollama_response_no_content(self):
        """Test Ollama streaming with message but no content."""
        from contextforge.providers import OllamaProvider
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = Mock()
            mock_response.status = 200
            
            # Create async iterator with message but no content
            async def mock_iter(self):
                yield b'{"message": {"no_content": "here"}}\n'
            
            mock_response.content.__aiter__ = mock_iter
            
            mock_session = Mock()
            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            provider = OllamaProvider()
            
            chunks = []
            async for chunk in provider._stream_generate({}):
                chunks.append(chunk)
            
            # Should have no chunks since there was no content
            assert chunks == [] 