# README.md
# ContextForge

A powerful Python package for building LLM applications with sophisticated context engineering. ContextForge simplifies the process of managing conversations, memory, tools, and retrieval-augmented generation (RAG) across multiple LLM providers.

## Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, Ollama, and custom providers
- **Streaming Responses**: Real-time token-by-token streaming for better user experience
- **Context Engineering**: Implements the complete context engineering pattern with system prompts, instructions, memory management, and tool integration
- **Conversation Management**: Automatic session handling with conversation history
- **Memory Systems**: Both short-term and long-term memory with SQLite persistence
- **Tool Integration**: Easy function calling with automatic parameter extraction
- **RAG Support**: Built-in retrieval system for augmenting responses with relevant information
- **Async-First**: Fully asynchronous for optimal performance
- **Type-Safe**: Comprehensive type hints throughout

## Installation

```bash
pip install contextforge
```

For additional features:
```bash
pip install contextforge[embeddings]  # For advanced embedding support
pip install contextforge[dev]         # For development tools
```

## Quick Start

```python
import asyncio
from contextforge import ContextEngine

async def main():
    # Initialize the engine
    engine = ContextEngine(
        provider="openai",  # or "anthropic", "ollama"
        default_system_prompt="You are a helpful assistant."
    )
    
    # Simple generation
    response = await engine.generate("What is the capital of France?")
    print(response)
    
    # Follow-up with automatic context
    response = await engine.generate("What is its population?")
    print(response)
    
    # Streaming response
    print("Assistant: ", end="", flush=True)
    async for chunk in await engine.generate("Tell me a story", stream=True):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## Advanced Usage

### Using Tools

```python
from contextforge import ContextEngine, ToolRegistry

# Create tool registry
tools = ToolRegistry()

@tools.register(description="Search the web")
def web_search(query: str) -> str:
    return f"Results for {query}..."

@tools.register(description="Calculate math")
def calculate(expression: str) -> float:
    return eval(expression)

# Use with engine
engine = ContextEngine(provider="openai", tool_registry=tools)
response = await engine.generate("Search for Python tutorials", use_tools=True)
```

### Retrieval-Augmented Generation

```python
from contextforge import ContextEngine, VectorRetriever

# Setup retriever
retriever = VectorRetriever()
await retriever.add_document({
    "content": "Paris is the capital of France with 2.2 million residents.",
    "source": "Wikipedia"
})

# Use with engine
engine = ContextEngine(provider="anthropic", retriever=retriever)
response = await engine.generate("Tell me about Paris", retrieve=True)
```

### Streaming Responses

```python
# Basic streaming
async for chunk in await engine.generate("Tell me a story", stream=True):
    print(chunk, end="", flush=True)

# Streaming with callbacks (production pattern)
async def on_token(token: str):
    # Send to frontend via WebSocket
    await websocket.send(token)

response_chunks = []
async for chunk in await engine.generate(prompt, stream=True):
    response_chunks.append(chunk)
    await on_token(chunk)

full_response = "".join(response_chunks)
```

### Structured Output

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}

response = await engine.generate(
    "Generate a person profile",
    output_schema=schema
)
```

## Documentation

Full documentation is available at [docs.contextforge.dev](https://docs.contextforge.dev).

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.