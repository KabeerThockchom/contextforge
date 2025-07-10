# examples/advanced_patterns.py
"""Advanced usage patterns for ContextForge."""

import asyncio
from contextforge import ContextEngine, ToolRegistry, VectorRetriever


async def chain_example():
    """Chaining operations for complex workflows."""
    engine = ContextEngine("openai")
    
    # Define processing steps
    async def summarize(text: str) -> dict:
        summary = await engine.generate(f"Summarize this text: {text}")
        return {"summary": summary, "original": text}
    
    async def extract_keywords(summary: str, **kwargs) -> dict:
        keywords = await engine.generate(f"Extract 5 keywords from: {summary}")
        return {"keywords": keywords, **kwargs}
    
    async def generate_title(keywords: str, summary: str, **kwargs) -> str:
        title = await engine.generate(
            f"Create a title based on keywords: {keywords} and summary: {summary}"
        )
        return title
    
    # Create and run chain
    chain = engine.chain(summarize, extract_keywords, generate_title)
    
    long_text = "Your long article text here..."
    result = await chain.run(long_text)
    print(f"Generated title: {result}")


async def multi_provider_example():
    """Using multiple providers for different tasks."""
    # Fast model for simple tasks
    fast_engine = ContextEngine("ollama", model="mistral")
    
    # Powerful model for complex tasks
    strong_engine = ContextEngine("anthropic", model="claude-3-opus-20240229")
    
    # Router function
    async def smart_generate(prompt: str) -> str:
        # Analyze complexity (simple heuristic)
        if len(prompt.split()) < 10 and "?" in prompt:
            # Simple question - use fast model
            return await fast_engine.generate(prompt)
        else:
            # Complex task - use strong model
            return await strong_engine.generate(prompt)
    
    # Test routing
    simple = await smart_generate("What is 2+2?")
    complex = await smart_generate("Write a detailed analysis of...long prompt...")


async def custom_provider_example():
    """Creating a custom provider."""
    from contextforge.providers import Provider
    
    class CustomProvider(Provider):
        """Example custom provider."""
        
        async def generate(self, messages, **kwargs):
            # Your custom logic here
            prompt = messages[-1]["content"]
            return f"Custom response to: {prompt}"
        
        def get_info(self):
            return {"provider": "custom", "version": "1.0"}
    
    # Use custom provider
    engine = ContextEngine(provider=CustomProvider())
    response = await engine.generate("Hello!")
    print(response)


async def agent_example():
    """Building an autonomous agent."""
    tools = ToolRegistry()
    
    @tools.register(description="Execute Python code")
    async def python_exec(code: str) -> str:
        # In production, use safe execution environment
        try:
            result = eval(code)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    @tools.register(description="Read file contents")
    async def read_file(filepath: str) -> str:
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    
    # Agent with planning capabilities
    engine = ContextEngine(
        provider="openai",
        tool_registry=tools,
        default_system_prompt="""You are an autonomous agent that can:
        1. Break down complex tasks into steps
        2. Use available tools to complete tasks
        3. Verify results and retry if needed
        """
    )
    
    # Complex task
    response = await engine.generate(
        "Read the config.json file and calculate the sum of all numeric values in it",
        use_tools=True
    )
    print(response)


async def streaming_example():
    """Streaming responses for real-time output."""
    # Note: Requires provider support for streaming
    from contextforge import StreamingProvider
    
    class StreamingProvider(Provider):
        async def generate(self, messages, **kwargs):
            if kwargs.get("stream", False):
                # Return async generator
                async def stream_response():
                    words = "This is a streaming response".split()
                    for word in words:
                        yield word + " "
                        await asyncio.sleep(0.1)
                
                return stream_response()
            else:
                return "Non-streaming response"
    
    engine = ContextEngine(provider=StreamingProvider())
    
    # Stream response
    response = await engine.generate("Tell me a story", stream=True)
    async for chunk in response:
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(chain_example())
