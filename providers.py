# contextforge/providers.py
"""LLM provider abstractions."""

from typing import List, Dict, Any, Optional, Union, AsyncIterator
from abc import ABC, abstractmethod
import os
import aiohttp
import json


class Provider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any], AsyncIterator[str]]:
        """
        Generate a response from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            **kwargs: Additional provider-specific arguments
            
        Returns:
            If stream=False: Complete response string
            If stream=True: AsyncIterator yielding response chunks
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        pass


class OpenAIProvider(Provider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        
        if not self.api_key:
            raise ValueError("OpenAI API key required")
    
    async def generate(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, AsyncIterator[str]]:
        """Generate using OpenAI API with optional streaming."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "stream": stream
        }
        
        if stream:
            return self._stream_generate(headers, data)
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    result = await response.json()
                    
                    if response.status != 200:
                        raise Exception(f"OpenAI API error: {result}")
                    
                    return result["choices"][0]["message"]["content"]
    
    async def _stream_generate(self, headers: Dict[str, str], data: Dict[str, Any]) -> AsyncIterator[str]:
        """Stream responses from OpenAI."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error = await response.json()
                    raise Exception(f"OpenAI API error: {error}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        chunk = line[6:]  # Remove "data: " prefix
                        if chunk == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(chunk)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model,
            "max_tokens": 128000 if "gpt-4" in self.model else 4096,
            "supports_streaming": True
        }


class AnthropicProvider(Provider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Anthropic API key required")
    
    async def generate(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, AsyncIterator[str]]:
        """Generate using Anthropic API with optional streaming."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Convert messages to Anthropic format
        system_prompt = None
        claude_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        data = {
            "model": kwargs.get("model", self.model),
            "messages": claude_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": stream
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        if stream:
            return self._stream_generate(headers, data)
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data
                ) as response:
                    result = await response.json()
                    
                    if response.status != 200:
                        raise Exception(f"Anthropic API error: {result}")
                    
                    return result["content"][0]["text"]
    
    async def _stream_generate(self, headers: Dict[str, str], data: Dict[str, Any]) -> AsyncIterator[str]:
        """Stream responses from Anthropic."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error = await response.json()
                    raise Exception(f"Anthropic API error: {error}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(chunk)
                            if data.get("type") == "content_block_delta":
                                yield data["delta"]["text"]
                        except json.JSONDecodeError:
                            continue
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "provider": "anthropic",
            "model": self.model,
            "max_tokens": 200000,
            "supports_streaming": True
        }


class OllamaProvider(Provider):
    """Ollama local model provider."""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def generate(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, AsyncIterator[str]]:
        """Generate using Ollama API with optional streaming."""
        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "stream": stream
        }
        
        if stream:
            return self._stream_generate(data)
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=data
                ) as response:
                    result = await response.json()
                    
                    if response.status != 200:
                        raise Exception(f"Ollama API error: {result}")
                    
                    return result["message"]["content"]
    
    async def _stream_generate(self, data: Dict[str, Any]) -> AsyncIterator[str]:
        """Stream responses from Ollama."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=data
            ) as response:
                if response.status != 200:
                    error = await response.json()
                    raise Exception(f"Ollama API error: {error}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except json.JSONDecodeError:
                            continue
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "provider": "ollama",
            "model": self.model,
            "max_tokens": 4096,
            "supports_streaming": True
        }


def create_provider(provider_name: str, **kwargs) -> Provider:
    """Factory function to create providers."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return providers[provider_name](**kwargs)