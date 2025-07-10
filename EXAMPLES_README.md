# ContextForge Examples with Ollama

This directory contains example scripts demonstrating how to use the ContextForge package to create AI agents with Ollama as the LLM provider.

## 🎉 Native Tool Calling Support

**Great news!** Ollama now supports native tool calling (as of May 2024). The ContextForge package has been updated to fully support this feature, making tool execution seamless and automatic.

### Models that support native tool calling:
- `qwen3:4b` and `qwen3:4b-coder`
- `llama3.1` (8b, 70b, 405b)
- `gemma2`
- `mistral`
- `qwen3`
- And more!

## Prerequisites

1. **Install Ollama** (latest version required for tool support)
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Or visit https://ollama.ai for other installation methods
   ```

2. **Start Ollama**
   ```bash
   ollama serve
   ```

3. **Pull a model that supports tools**
   ```bash
   ollama pull qwen3:4b      # Recommended for tool calling
   ollama pull llama3.1     # Also great for tools
   ollama pull gemma2       # Good alternative
   ```

4. **Install ContextForge**
   ```bash
   # If using the local package
   pip install -e .
   
   # Or if published
   pip install contextforge
   ```

## Example Files

### 1. `ollama_native_tools_example.py` - 🌟 Recommended
**Native tool calling with automatic execution.** This is the modern approach using Ollama's built-in tool support.

Features:
- Automatic tool detection and execution
- Streaming support with tools
- Memory integration
- Multiple demo modes

```bash
python ollama_native_tools_example.py
# Choose from: interactive, examples, streaming, memory demos
```

### 2. `minimal_example.py` - Bare Minimum
The simplest possible example showing:
- Creating an agent with Ollama
- Adding a calculator tool
- Using memory

```bash
python minimal_example.py
```

### 3. `simple_agent_example.py` - Basic Features
A more complete example with:
- Multiple calculator tools
- Both simple demo and interactive chat modes
- Streaming responses
- Persistent memory with SQLite

```bash
python simple_agent_example.py
# Then choose option 1 or 2
```

### 4. `example_agent.py` - Full Featured
Comprehensive example including:
- Complete calculator tool suite (add, subtract, multiply, divide, power, sqrt)
- Interactive chat interface
- Conversation management (history, clear)
- Advanced examples with complex calculations
- Error handling

```bash
python example_agent.py
```



## How Native Tool Calling Works

With the updated ContextForge package and a compatible Ollama model, tool calling is automatic:

```python
# Define your tools
tools = ToolRegistry()

@tools.register(description="Multiply two numbers")
def multiply(a: float, b: float) -> float:
    return a * b

# Create the engine
engine = ContextEngine(
    provider="ollama",
    model="qwen3:4b",  # Use a model that supports tools
    tool_registry=tools
)

# Just call generate - tools are executed automatically!
response = await engine.generate("What is 15 times 25?")
# The engine will:
# 1. Send the query to Ollama with tool definitions
# 2. Receive tool call requests from the model
# 3. Execute the tools automatically
# 4. Send results back to the model
# 5. Return the final response
```

## Key Concepts Demonstrated

### 1. Creating Tools
```python
tools = ToolRegistry()

@tools.register(description="Add two numbers")
def add(a: float, b: float) -> float:
    return a + b
```

### 2. Setting Up the Agent
```python
agent = ContextEngine(
    provider="ollama",
    model="qwen3:4b",  # Use a tool-capable model
    memory_store=SQLiteMemoryStore("memories.db"),
    tool_registry=tools,
    default_system_prompt="You are a helpful assistant."
)
```

### 3. Using the Agent
```python
# Simple generation with automatic tool execution
response = await agent.generate("What is 2 + 2?")

# With session for memory
response = await agent.generate(
    "Remember my favorite color is blue",
    session_id="user123"
)

# Streaming responses (tools still work!)
async for chunk in await agent.generate(prompt, stream=True):
    print(chunk, end="", flush=True)
```

## Memory Options

### In-Memory Store (Temporary)
```python
memory_store = InMemoryStore(max_memories=100)
```

### SQLite Store (Persistent)
```python
memory_store = SQLiteMemoryStore("agent_memories.db")
```

## Tips for Better Tool Calling

1. **Use a compatible model**: Not all models support tools. Use `qwen3:4b`, `llama3.1`, etc.

2. **Increase context window for complex tools**:
   ```python
   response = await engine.generate(
       prompt,
       options={"num_ctx": 32000}  # Larger context for better tool handling
   )
   ```

3. **Clear tool descriptions**: Make sure your tool descriptions are clear and specific.

4. **Lower temperature for accuracy**:
   ```python
   response = await engine.generate(prompt, temperature=0.2)
   ```

## Troubleshooting

1. **"Connection refused" error**
   - Make sure Ollama is running: `ollama serve`
   - Check if it's running on the default port (11434)

2. **"Model not found" error**
   - Pull the model first: `ollama pull qwen3:4b`
   - Check available models: `ollama list`

3. **Tools not being called**
   - Ensure you're using a model that supports tools
   - Update Ollama to the latest version
   - Check the model's tool calling format in Ollama docs

4. **Slow responses**
   - Tool calling may be slower than regular generation
   - Try a smaller model or increase system resources
   - Use streaming to see responses as they generate

## Available Ollama Models for Tools

Models verified to work with native tool calling:
- `qwen3:4b` - Excellent tool support, recommended
- `qwen3:4b-coder` - Great for code-related tools
- `llama3.1` - Strong tool calling capabilities
- `gemma2` - Good balance of speed and accuracy
- `mistral` - Fast with decent tool support
- `qwen3` - Original tool-capable model

Check Ollama's documentation for the latest list of tool-capable models.

## Customization

### Different Ollama Endpoint
```python
from contextforge import create_provider

engine = ContextEngine(
    provider=create_provider("ollama", base_url="http://localhost:11434"),
    # ...
)
```

### Different Models
```python
engine = ContextEngine(
    provider="ollama",
    model="llama3.1",  # Change to any tool-capable model
    # ...
)
```

### Custom System Prompts
```python
engine = ContextEngine(
    # ...
    default_system_prompt="""You are a specialized math tutor.
    Always use the calculator tools for accuracy.
    Explain each step of your calculations."""
)
```

### Advanced Options
```python
# For better tool performance
response = await engine.generate(
    prompt,
    options={
        "num_ctx": 32000,      # Larger context window
        "temperature": 0.2,     # Lower randomness
        "top_p": 0.9           # Nucleus sampling
    }
)
``` 