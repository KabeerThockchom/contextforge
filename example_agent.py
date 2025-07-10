#!/usr/bin/env python3
"""
Example of creating an AI agent using ContextForge with:
- Ollama as the LLM provider
- A simple calculator tool
- Memory capabilities (both short-term and long-term)
"""

import asyncio
from contextforge import (
    ContextEngine,
    ToolRegistry,
    SQLiteMemoryStore,
    create_provider
)

# Create a tool registry and define calculator tools
tool_registry = ToolRegistry()

@tool_registry.register(description="Add two numbers")
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool_registry.register(description="Subtract two numbers")
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@tool_registry.register(description="Multiply two numbers")
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool_registry.register(description="Divide two numbers")
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises error if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

@tool_registry.register(description="Calculate the power of a number")
def power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent."""
    return base ** exponent

@tool_registry.register(description="Calculate the square root of a number")
def sqrt(number: float) -> float:
    """Calculate the square root of a number."""
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number!")
    return number ** 0.5


async def main():
    """Main function demonstrating the agent capabilities."""
    
    # Initialize memory stores
    # You can use InMemoryStore for temporary storage or SQLiteMemoryStore for persistence
    # memory_store = InMemoryStore(max_memories=100)  # For temporary memory
    memory_store = SQLiteMemoryStore("agent_memories.db")  # For persistent memory
    
    # Create the context engine with Ollama provider
    # Make sure Ollama is running locally (default port 11434)
    # You can specify different models like "qwen3:4b", "llama3.1", "gemma2", etc.
    engine = ContextEngine(
        provider="ollama",
        model="qwen3:4b",  # Tool-capable model (also try llama3.1, gemma2)
        memory_store=memory_store,
        tool_registry=tool_registry,
        default_system_prompt="""You are a helpful AI assistant with access to calculator tools.
You can perform mathematical calculations when asked.
Remember important information from our conversations for future reference.
Always be polite and helpful."""
    )
    
    # Create a conversation session
    session_id = "math_session_1"
    
    print("🤖 AI Calculator Assistant")
    print("=" * 50)
    print("I can help you with mathematical calculations!")
    print("I have access to: add, subtract, multiply, divide, power, and sqrt functions.")
    print("I also remember our conversation history.")
    print("\nType 'quit' to exit, 'history' to see conversation summary, or 'clear' to clear history.")
    print("=" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() == 'quit':
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == 'history':
                session = engine.get_session(session_id)
                print(f"\n📜 Conversation Summary: {session.summarize()}")
                continue
            
            elif user_input.lower() == 'clear':
                session = engine.get_session(session_id)
                session.clear()
                print("\n🗑️  Conversation history cleared!")
                continue
            
            # Generate response with tools and memory
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # You can use streaming for real-time response
            response = await engine.generate(
                prompt=user_input,
                session_id=session_id,
                use_tools=True,  # Enable tool usage
                retrieve=True,   # Enable memory retrieval
                temperature=0.7,
                max_tokens=500
            )
            
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


async def advanced_example():
    """Advanced example showing more features."""
    
    # Create engine with custom configuration
    engine = ContextEngine(
        provider=create_provider("ollama", model="qwen3:4b", base_url="http://localhost:11434"),
        memory_store=SQLiteMemoryStore("advanced_memories.db"),
        tool_registry=tool_registry,
        default_system_prompt="You are an advanced mathematical assistant."
    )
    
    # Example 1: Simple calculation
    print("\n🔢 Example 1: Simple Calculation")
    response = await engine.generate(
        "What is 25 * 4 + 10?",
        session_id="example_session"
    )
    print(f"Response: {response}")
    
    # Example 2: Complex calculation with memory
    print("\n🔢 Example 2: Calculation with Memory")
    response = await engine.generate(
        "Calculate 2 to the power of 8. Remember this result as 'my special number'.",
        session_id="example_session"
    )
    print(f"Response: {response}")
    
    # Example 3: Using previous memory
    print("\n🔢 Example 3: Using Previous Memory")
    response = await engine.generate(
        "What was my special number? Divide it by 4.",
        session_id="example_session"
    )
    print(f"Response: {response}")
    
    # Example 4: Multiple operations
    print("\n🔢 Example 4: Multiple Operations")
    response = await engine.generate(
        "I need to calculate: (100 + 50) * 2 - 25. Then find the square root of the result.",
        session_id="example_session"
    )
    print(f"Response: {response}")
    
    # Example 5: Streaming response
    print("\n🔢 Example 5: Streaming Response")
    print("Response: ", end="", flush=True)
    async for chunk in await engine.generate(
        "Explain how to calculate the area of a circle with radius 5, then calculate it.",
        session_id="example_session",
        stream=True
    ):
        print(chunk, end="", flush=True)
    print()  # New line after streaming


if __name__ == "__main__":
    # Run the main interactive example
    asyncio.run(main())
    
    # Uncomment the following line to run the advanced examples
    # asyncio.run(advanced_example()) 