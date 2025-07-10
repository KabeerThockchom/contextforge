# examples/basic_usage.py
"""Basic usage examples for ContextForge."""

import asyncio
from contextforge import ContextEngine, create_provider

async def basic_chat_example():
    """Simple chat with context management."""
    # Initialize with OpenAI
    engine = ContextEngine(
        provider="openai",
        default_system_prompt="You are a helpful AI assistant.",
        api_key="your-api-key-here"  # Or set OPENAI_API_KEY env var
    )
    
    # Chat with automatic context management
    response1 = await engine.generate("What is the capital of France?")
    print(f"Assistant: {response1}")
    
    # Follow-up uses conversation history automatically
    response2 = await engine.generate("What is its population?")
    print(f"Assistant: {response2}")


async def multi_session_example():
    """Multiple conversation sessions."""
    engine = ContextEngine(provider="ollama", model="llama2")
    
    # First session
    await engine.generate("Let's talk about Python", session_id="python_chat")
    await engine.generate("What are decorators?", session_id="python_chat")
    
    # Different session
    await engine.generate("Let's discuss cooking", session_id="cooking_chat")
    await engine.generate("How do I make pasta?", session_id="cooking_chat")
    
    # Return to first session - context preserved
    response = await engine.generate(
        "Can you show me an example?", 
        session_id="python_chat"
    )
    print(response)


async def tool_usage_example():
    """Using tools with the engine."""
    from contextforge import ToolRegistry
    
    # Create tool registry
    tools = ToolRegistry()
    
    # Register tools
    @tools.register(description="Search the web for information")
    def web_search(query: str) -> str:
        return f"Search results for '{query}': [simulated results]"
    
    @tools.register(description="Calculate mathematical expressions")
    def calculate(expression: str) -> float:
        return eval(expression)  # In production, use safe evaluation
    
    # Create engine with tools
    engine = ContextEngine(
        provider="openai",
        tool_registry=tools
    )
    
    # Generate with tools available
    response = await engine.generate(
        "What's the weather in Paris and what's 15% of 120?",
        use_tools=True
    )
    print(response)


async def rag_example():
    """Retrieval-augmented generation example."""
    from contextforge import VectorRetriever
    
    # Create retriever
    retriever = VectorRetriever()
    
    # Add documents
    await retriever.add_document({
        "content": "The Eiffel Tower is 330 meters tall and was built in 1889.",
        "source": "Wikipedia"
    })
    await retriever.add_document({
        "content": "The Louvre Museum houses the Mona Lisa and has 35,000 artworks.",
        "source": "Museum Guide"
    })
    
    # Create engine with retriever
    engine = ContextEngine(
        provider="anthropic",
        retriever=retriever
    )
    
    # Query with retrieval
    response = await engine.generate(
        "Tell me about famous landmarks in Paris",
        retrieve=True
    )
    print(response)


async def structured_output_example():
    """Generate structured output."""
    engine = ContextEngine("openai")
    
    # Define output schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age", "skills"]
    }
    
    response = await engine.generate(
        "Create a profile for a Python developer",
        output_schema=schema
    )
    print(response)


async def custom_context_example():
    """Advanced context manipulation."""
    engine = ContextEngine("ollama")
    
    # Generate with custom instructions
    response = await engine.generate(
        "Explain quantum computing",
        instructions=[
            "Use simple language suitable for a 10-year-old",
            "Include a fun analogy",
            "Keep it under 100 words"
        ]
    )
    print(response)


async def memory_persistence_example():
    """Using persistent memory."""
    from contextforge.memory import SQLiteMemoryStore
    
    # Use SQLite for persistent memory
    memory_store = SQLiteMemoryStore("chat_memory.db")
    
    engine = ContextEngine(
        provider="openai",
        memory_store=memory_store
    )
    
    # Conversations are automatically saved
    await engine.generate("Remember that my favorite color is blue")
    
    # Later queries can retrieve this information
    response = await engine.generate("What's my favorite color?")
    print(response)


async def streaming_example():
    """Using streaming for real-time responses."""
    engine = ContextEngine("openai")
    
    print("Assistant: ", end="", flush=True)
    
    # Stream the response token by token
    async for chunk in await engine.generate(
        "Write a short poem about coding",
        stream=True
    ):
        print(chunk, end="", flush=True)
        # In a web app, you'd send each chunk to the frontend
    
    print("\n\nStreaming complete!")


async def streaming_with_memory():
    """Streaming while maintaining conversation context."""
    engine = ContextEngine(
        provider="anthropic",
        memory_store=SQLiteMemoryStore("chat.db")
    )
    
    # First message (streamed)
    print("User: Hi, I'm learning Python")
    print("Assistant: ", end="", flush=True)
    
    async for chunk in await engine.generate(
        "Hi, I'm learning Python",
        session_id="learner_001",
        stream=True
    ):
        print(chunk, end="", flush=True)
    
    print("\n\nUser: What should I learn first?")
    print("Assistant: ", end="", flush=True)
    
    # Follow-up uses conversation history
    async for chunk in await engine.generate(
        "What should I learn first?",
        session_id="learner_001", 
        stream=True
    ):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    # Run examples
    asyncio.run(basic_chat_example())
    # asyncio.run(streaming_example())
    # asyncio.run(streaming_with_memory())

