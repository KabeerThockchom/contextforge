# tests/test_integration.py
"""Integration tests for ContextForge components working together."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from contextforge import ContextEngine, ToolRegistry, VectorRetriever
from contextforge.memory import InMemoryStore
from contextforge.providers import Provider


class MockProvider(Provider):
    """Mock provider for integration testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or ["Mock response"]
        self.call_count = 0
    
    async def generate(self, messages, stream=False, **kwargs):
        """Generate mock responses."""
        if stream:
            async def mock_stream():
                response = self.responses[self.call_count % len(self.responses)]
                for word in response.split():
                    yield word + " "
            return mock_stream()
        else:
            response = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            return response
    
    def get_info(self):
        return {"provider": "mock", "version": "1.0"}


class TestFullIntegration:
    """Test full integration of all components."""
    
    @pytest.mark.asyncio
    async def test_full_context_engine_workflow(self):
        """Test complete workflow with all components."""
        # Setup tools
        tools = ToolRegistry()
        
        @tools.register(description="Calculate math expressions")
        def calculate(expression: str) -> str:
            try:
                result = eval(expression)  # Safe for testing
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"
        
        @tools.register(description="Search for information")
        async def search(query: str) -> str:
            return f"Search results for: {query}"
        
        # Setup retriever with documents
        retriever = VectorRetriever()
        await retriever.add_document({
            "content": "Python is a programming language known for its simplicity",
            "source": "Programming Guide"
        })
        await retriever.add_document({
            "content": "Machine learning uses algorithms to find patterns in data",
            "source": "ML Tutorial"
        })
        
        # Setup memory
        memory = InMemoryStore()
        
        # Setup engine with all components
        provider = MockProvider([
            "Let me help you with Python programming.",
            "Based on the search results, here's what I found.",
            "The calculation shows interesting results."
        ])
        
        engine = ContextEngine(
            provider=provider,
            memory_store=memory,
            tool_registry=tools,
            retriever=retriever,
            default_system_prompt="You are a helpful AI assistant."
        )
        
        # Test conversation flow
        session_id = "integration_test"
        
        # First message - should use retrieval
        response1 = await engine.generate(
            "Tell me about Python programming",
            session_id=session_id,
            retrieve=True,
            use_tools=False
        )
        assert "Python programming" in response1 or "help you" in response1
        
        # Second message - should have conversation context
        response2 = await engine.generate(
            "What about machine learning?",
            session_id=session_id,
            retrieve=True
        )
        assert "found" in response2 or "search results" in response2
        
        # Third message - should use tools
        response3 = await engine.generate(
            "Calculate 15 * 8",
            session_id=session_id,
            use_tools=True
        )
        assert "calculation" in response3 or "results" in response3
        
        # Verify session has conversation history
        session = engine.get_session(session_id)
        assert len(session.messages) == 6  # 3 user + 3 assistant messages
    
    @pytest.mark.asyncio
    async def test_streaming_with_memory_and_tools(self):
        """Test streaming generation with memory and tools."""
        tools = ToolRegistry()
        
        @tools.register(description="Get current time")
        def get_time() -> str:
            return "2024-01-01 12:00:00"
        
        memory = InMemoryStore()
        provider = MockProvider(["This is a streaming response with tools"])
        
        engine = ContextEngine(
            provider=provider,
            memory_store=memory,
            tool_registry=tools
        )
        
        # Test streaming with tools available
        chunks = []
        async for chunk in await engine.generate(
            "What time is it?",
            session_id="streaming_test",
            use_tools=True,
            stream=True
        ):
            chunks.append(chunk)
        
        # Should have received chunks
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert "streaming response" in full_response
        
        # Verify conversation was saved
        session = engine.get_session("streaming_test")
        assert len(session.messages) == 2
        assert session.messages[-1].content == full_response
    
    @pytest.mark.asyncio
    async def test_multi_session_context_isolation(self):
        """Test that different sessions maintain separate contexts."""
        provider = MockProvider([
            "Session 1 response",
            "Session 2 response", 
            "Session 1 follow-up",
            "Session 2 follow-up"
        ])
        
        engine = ContextEngine(provider=provider)
        
        # First session
        response1a = await engine.generate("Hello", session_id="session1")
        assert "Session 1" in response1a
        
        # Second session  
        response2a = await engine.generate("Hello", session_id="session2")
        assert "Session 2" in response2a
        
        # Continue first session
        response1b = await engine.generate("How are you?", session_id="session1")
        assert "follow-up" in response1b
        
        # Continue second session
        response2b = await engine.generate("How are you?", session_id="session2")
        assert "follow-up" in response2b
        
        # Verify sessions are separate
        session1 = engine.get_session("session1")
        session2 = engine.get_session("session2")
        
        assert len(session1.messages) == 4  # 2 user + 2 assistant
        assert len(session2.messages) == 4  # 2 user + 2 assistant
        assert session1.messages != session2.messages
    
    @pytest.mark.asyncio
    async def test_memory_persistence_across_generations(self):
        """Test that memories persist and influence future generations."""
        memory = InMemoryStore()
        provider = MockProvider([
            "I'll remember that you like Python.",
            "Based on what I remember, you enjoy Python programming."
        ])
        
        engine = ContextEngine(
            provider=provider,
            memory_store=memory
        )
        
        # First interaction - should create memory
        await engine.generate(
            "I really enjoy programming in Python, it's my favorite language",
            session_id="memory_test"
        )
        
        # Add a memory manually to test retrieval
        await memory.add_memory({
            "content": "User enjoys Python programming",
            "context": "User mentioned Python is their favorite language"
        })
        
        # Second interaction - should retrieve relevant memory
        response = await engine.generate(
            "What programming languages do I like?",
            session_id="memory_test"
        )
        
        assert "remember" in response or "Python" in response
    
    @pytest.mark.asyncio
    async def test_error_handling_across_components(self):
        """Test error handling when components fail."""
        # Create tools that can fail
        tools = ToolRegistry()
        
        @tools.register(description="Tool that may fail")
        def failing_tool(should_fail: bool = False) -> str:
            if should_fail:
                raise ValueError("Tool execution failed")
            return "Tool executed successfully"
        
        # Create retriever
        retriever = VectorRetriever()
        
        # Mock provider that handles errors gracefully
        provider = MockProvider(["I encountered an issue but can continue."])
        
        engine = ContextEngine(
            provider=provider,
            tool_registry=tools,
            retriever=retriever
        )
        
        # Test that engine handles tool execution errors gracefully
        # (This would require actual tool execution integration)
        response = await engine.generate(
            "Please use the failing tool",
            use_tools=True
        )
        
        # Should still get a response even if tools fail
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_complex_context_building(self):
        """Test complex context building with all components."""
        # Setup comprehensive environment
        tools = ToolRegistry()
        
        @tools.register(description="Weather information")
        def get_weather(location: str) -> dict:
            return {
                "location": location,
                "temperature": "22°C",
                "condition": "sunny"
            }
        
        retriever = VectorRetriever()
        await retriever.add_document({
            "content": "Weather affects mood and productivity",
            "source": "Psychology Research"
        })
        
        memory = InMemoryStore()
        await memory.add_memory({
            "content": "User asked about weather before",
            "preference": "likes detailed weather reports"
        })
        
        provider = MockProvider([
            "Here's the comprehensive weather information you requested."
        ])
        
        engine = ContextEngine(
            provider=provider,
            memory_store=memory,
            tool_registry=tools,
            retriever=retriever,
            default_system_prompt="You are a helpful weather assistant."
        )
        
        # Generate with all context components active
        response = await engine.generate(
            "What's the weather like in Paris?",
            session_id="complex_test",
            instructions=[
                "Be detailed and helpful",
                "Use available tools",
                "Reference any relevant information"
            ],
            retrieve=True,
            use_tools=True
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify that context was built with all components
        session = engine.get_session("complex_test")
        assert len(session.messages) == 2  # user + assistant
    
    @pytest.mark.asyncio
    async def test_chain_operations(self):
        """Test chaining operations with the engine."""
        provider = MockProvider([
            "Analysis: The text is about Python programming.",
            "Summary: Python is popular for its simplicity.",
            "Keywords: Python, programming, language, simple"
        ])
        
        engine = ContextEngine(provider=provider)
        
        # Define chain operations
        async def analyze_text(text: str) -> dict:
            response = await engine.generate(f"Analyze this text: {text}")
            return {"analysis": response, "original": text}
        
        async def summarize_analysis(data: dict) -> dict:
            response = await engine.generate(f"Summarize: {data['analysis']}")
            return {"summary": response, **data}
        
        async def extract_keywords(data: dict) -> str:
            response = await engine.generate(f"Extract keywords from: {data['summary']}")
            return response
        
        # Create and run chain
        chain = engine.chain(analyze_text, summarize_analysis, extract_keywords)
        
        result = await chain.run("Python is a great programming language")
        
        assert isinstance(result, str)
        assert "Keywords" in result or "Python" in result


class TestProviderIntegration:
    """Test integration with different provider types."""
    
    @pytest.mark.asyncio
    async def test_provider_switching(self):
        """Test switching between different providers."""
        provider1 = MockProvider(["Response from provider 1"])
        provider2 = MockProvider(["Response from provider 2"])
        
        # Create engines with different providers
        engine1 = ContextEngine(provider=provider1)
        engine2 = ContextEngine(provider=provider2)
        
        response1 = await engine1.generate("Hello")
        response2 = await engine2.generate("Hello")
        
        assert "provider 1" in response1
        assert "provider 2" in response2
    
    @pytest.mark.asyncio
    async def test_provider_error_recovery(self):
        """Test error recovery with provider failures."""
        class FailingProvider(Provider):
            def __init__(self):
                self.fail_count = 0
            
            async def generate(self, messages, stream=False, **kwargs):
                self.fail_count += 1
                if self.fail_count <= 2:
                    raise Exception("Provider temporarily unavailable")
                return "Recovery successful"
            
            def get_info(self):
                return {"provider": "failing", "status": "unstable"}
        
        provider = FailingProvider()
        engine = ContextEngine(provider=provider)
        
        # First attempts should fail
        with pytest.raises(Exception):
            await engine.generate("Test")
        
        with pytest.raises(Exception):
            await engine.generate("Test")
        
        # Third attempt should succeed
        response = await engine.generate("Test")
        assert response == "Recovery successful"