# tests/test_providers.py
"""Tests for LLM provider implementations."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from contextforge.providers import (
    Provider, OpenAIProvider, AnthropicProvider, 
    OllamaProvider, create_provider
)


class TestProvider:
    """Test abstract Provider class."""
    
    def test_provider_abstract(self):
        """Test that Provider is abstract."""
        with pytest.raises(TypeError):
            Provider()


class TestOpenAIProvider:
    """Test OpenAI provider."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4"
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider()
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'})
    def test_init_with_env_api_key(self):
        """Test initialization with environment variable."""
        provider = OpenAIProvider()
        assert provider.api_key == "env-key"
    
    def test_get_info(self):
        """Test provider info."""
        provider = OpenAIProvider(api_key="test-key")
        info = provider.get_info()
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4"
        assert info["supports_streaming"] is True
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_generate_success(self, mock_session_class):
        """Test successful generation."""
        # Mock response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Hello, world!"}}]
        })
        
        # Mock session and post method
        mock_session = Mock()
        mock_post = Mock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = Mock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session
        
        provider = OpenAIProvider(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]
        
        result = await provider.generate(messages)
        assert result == "Hello, world!"
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_generate_error(self, mock_session_class):
        """Test generation with API error."""
        mock_response = Mock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={"error": "Bad request"})
        
        # Mock session and post method
        mock_session = Mock()
        mock_post = Mock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = Mock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session
        
        provider = OpenAIProvider(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception, match="OpenAI API error"):
            await provider.generate(messages)
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_generate_streaming(self, mock_session_class):
        """Test streaming generation."""
        # Mock streaming response
        mock_response = Mock()
        mock_response.status = 200
        
        # Create async iterator for streaming content
        async def mock_iter(self):
            stream_data = [
                b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
                b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
                b'data: {"choices": [{"delta": {"content": "!"}}]}\n\n',
                b'data: [DONE]\n\n'
            ]
            for data in stream_data:
                yield data
        
        mock_response.content.__aiter__ = mock_iter
        
        # Mock session and post method
        mock_session = Mock()
        mock_post = Mock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = Mock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session
        
        provider = OpenAIProvider(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]
        
        chunks = []
        async for chunk in await provider.generate(messages, stream=True):
            chunks.append(chunk)
        
        assert chunks == ["Hello", " world", "!"]


class TestAnthropicProvider:
    """Test Anthropic provider."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-opus-20240229"
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with pytest.raises(ValueError, match="Anthropic API key required"):
            AnthropicProvider()
    
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'env-key'})
    def test_init_with_env_api_key(self):
        """Test initialization with environment variable."""
        provider = AnthropicProvider()
        assert provider.api_key == "env-key"
    
    def test_get_info(self):
        """Test provider info."""
        provider = AnthropicProvider(api_key="test-key")
        info = provider.get_info()
        assert info["provider"] == "anthropic"
        assert info["model"] == "claude-3-opus-20240229"
        assert info["supports_streaming"] is True
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_generate_success(self, mock_session_class):
        """Test successful generation."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "content": [{"text": "Hello, world!"}]
        })
        
        # Mock session and post method
        mock_session = Mock()
        mock_post = Mock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = Mock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session
        
        provider = AnthropicProvider(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]
        
        result = await provider.generate(messages)
        assert result == "Hello, world!"
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_generate_with_system_message(self, mock_session_class):
        """Test generation with system message conversion."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "content": [{"text": "Hello, world!"}]
        })
        
        # Mock session and post method
        mock_session = Mock()
        mock_post = Mock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = Mock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session
        
        provider = AnthropicProvider(api_key="test-key")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        
        await provider.generate(messages)
        
        # Check that system message was converted properly
        call_args = mock_session.post.call_args
        data = call_args[1]["json"]
        assert "system" in data
        assert data["system"] == "You are helpful"
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"


class TestOllamaProvider:
    """Test Ollama provider."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        provider = OllamaProvider()
        assert provider.model == "qwen3:4b"
        assert provider.base_url == "http://localhost:11434"
    
    def test_init_custom(self):
        """Test initialization with custom values."""
        provider = OllamaProvider(
            model="mistral",
            base_url="http://custom:8080"
        )
        assert provider.model == "mistral"
        assert provider.base_url == "http://custom:8080"
    
    def test_get_info(self):
        """Test provider info."""
        provider = OllamaProvider()
        info = provider.get_info()
        assert info["provider"] == "ollama"
        assert info["model"] == "qwen3:4b"
        assert info["supports_streaming"] is True
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_generate_success(self, mock_session_class):
        """Test successful generation."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "message": {"content": "Hello, world!"}
        })
        
        # Mock session and post method
        mock_session = Mock()
        mock_post = Mock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = Mock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session
        
        provider = OllamaProvider()
        messages = [{"role": "user", "content": "Hello"}]
        
        result = await provider.generate(messages)
        assert result == "Hello, world!"
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_generate_streaming(self, mock_session_class):
        """Test streaming generation."""
        mock_response = Mock()
        mock_response.status = 200
        
        # Create async iterator for streaming content
        async def mock_iter(self):
            stream_data = [
                b'{"message": {"content": "Hello"}}\n',
                b'{"message": {"content": " world"}}\n',
                b'{"message": {"content": "!"}}\n'
            ]
            for data in stream_data:
                yield data
        
        mock_response.content.__aiter__ = mock_iter
        
        # Mock session and post method
        mock_session = Mock()
        mock_post = Mock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = Mock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session
        
        provider = OllamaProvider()
        messages = [{"role": "user", "content": "Hello"}]
        
        chunks = []
        async for chunk in await provider.generate(messages, stream=True):
            chunks.append(chunk)
        
        assert chunks == ["Hello", " world", "!"]


class TestCreateProvider:
    """Test provider factory function."""
    
    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        provider = create_provider("openai", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "test-key"
    
    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        provider = create_provider("anthropic", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == "test-key"
    
    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        provider = create_provider("ollama", model="mistral")
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "mistral"
    
    def test_create_unknown_provider(self):
        """Test creating unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("unknown")
    
    def test_create_with_custom_args(self):
        """Test creating provider with custom arguments."""
        provider = create_provider(
            "openai", 
            api_key="test-key", 
            model="gpt-3.5-turbo"
        )
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-3.5-turbo"