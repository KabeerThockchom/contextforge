# contextforge/core.py
"""Core context engineering components."""

from typing import Dict, List, Optional, Any, Union, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
import json
from .memory import MemoryStore, ConversationMemory, InMemoryStore
from .providers import Provider
from .tools import ToolRegistry
from .retrieval import Retriever
from .utils import merge_contexts


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """Represents the complete context for an LLM request."""
    system_prompt: Optional[str] = None
    instructions: List[str] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    long_term_memory: List[Dict[str, Any]] = field(default_factory=list)
    short_term_memory: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_info: List[Dict[str, Any]] = field(default_factory=list)
    available_tools: List[Dict[str, Any]] = field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert context to provider-compatible message format."""
        messages = []
        
        # Add system message
        system_parts = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        
        if self.instructions:
            system_parts.append("\n## Instructions\n" + "\n".join(f"- {inst}" for inst in self.instructions))
        
        if self.long_term_memory:
            system_parts.append("\n## Long-term Memory\n" + json.dumps(self.long_term_memory, indent=2))
        
        if self.available_tools:
            system_parts.append("\n## Available Tools\n" + json.dumps(self.available_tools, indent=2))
        
        if self.output_schema:
            system_parts.append("\n## Output Schema\n" + json.dumps(self.output_schema, indent=2))
        
        if system_parts:
            messages.append({"role": "system", "content": "\n".join(system_parts)})
        
        # Add conversation messages
        for msg in self.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add retrieved info and short-term memory as user context if present
        context_parts = []
        if self.short_term_memory:
            context_parts.append("## Recent Context\n" + json.dumps(self.short_term_memory, indent=2))
        
        if self.retrieved_info:
            context_parts.append("## Retrieved Information\n" + json.dumps(self.retrieved_info, indent=2))
        
        if context_parts and messages:
            # Insert context before the last user message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i]["content"] = "\n".join(context_parts) + "\n\n" + messages[i]["content"]
                    break
        
        return messages


class ContextEngine:
    """Main engine for managing context in LLM applications."""
    
    def __init__(
        self,
        provider: Union[str, Provider],
        memory_store: Optional[MemoryStore] = None,
        tool_registry: Optional[ToolRegistry] = None,
        retriever: Optional[Retriever] = None,
        default_system_prompt: Optional[str] = None,
        **provider_kwargs
    ):
        """
        Initialize the ContextEngine.
        
        Args:
            provider: LLM provider name or Provider instance
            memory_store: Optional memory store for conversation history
            tool_registry: Optional tool registry for function calling
            retriever: Optional retriever for RAG
            default_system_prompt: Default system prompt
            **provider_kwargs: Additional arguments for provider initialization
        """
        if isinstance(provider, str):
            from .providers import create_provider
            self.provider = create_provider(provider, **provider_kwargs)
        else:
            self.provider = provider
        
        self.memory_store = memory_store or InMemoryStore()
        self.tool_registry = tool_registry or ToolRegistry()
        self.retriever = retriever
        self.default_system_prompt = default_system_prompt
        self.sessions: Dict[str, ConversationMemory] = {}
    
    def create_session(self, session_id: str = "default") -> ConversationMemory:
        """Create a new conversation session."""
        session = ConversationMemory(session_id)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str = "default") -> ConversationMemory:
        """Get or create a conversation session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(session_id)
        return self.sessions[session_id]
    
    async def generate(
        self,
        prompt: str,
        session_id: str = "default",
        system_prompt: Optional[str] = None,
        instructions: Optional[List[str]] = None,
        retrieve: bool = True,
        use_tools: bool = True,
        output_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate a response with full context management.
        
        Args:
            prompt: User prompt
            session_id: Conversation session ID
            system_prompt: Override system prompt
            instructions: Additional instructions
            retrieve: Whether to use retriever
            use_tools: Whether to make tools available
            output_schema: Schema for structured output
            stream: Whether to stream the response
            **kwargs: Additional arguments for the provider
        
        Returns:
            If stream=False: Generated response string
            If stream=True: AsyncIterator yielding response chunks
        """
        # Get or create session
        session = self.get_session(session_id)
        
        # Build context
        context = Context(
            system_prompt=system_prompt or self.default_system_prompt,
            instructions=instructions or [],
            messages=session.get_messages(),
            output_schema=output_schema
        )
        
        # Add user message
        user_msg = Message(role="user", content=prompt)
        context.messages.append(user_msg)
        
        # Retrieve relevant information
        if retrieve and self.retriever:
            retrieved = await self.retriever.retrieve(prompt)
            context.retrieved_info = retrieved
        
        # Get memory context
        long_term = await self.memory_store.get_relevant_memories(prompt)
        short_term = session.get_recent_context()
        context.long_term_memory = long_term
        context.short_term_memory = short_term
        
        # Add available tools
        if use_tools:
            context.available_tools = self.tool_registry.get_tool_descriptions()
        
        # Generate response
        messages = context.to_messages()
        
        if stream:
            # For streaming, we need to handle the response differently
            return self._stream_generate(messages, session, user_msg, **kwargs)
        else:
            response = await self.provider.generate(messages, stream=False, **kwargs)
            
            # Handle tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_results = await self._execute_tools(response.tool_calls)
                # Add tool results to context and regenerate
                context.messages.append(Message(role="assistant", content=response.content))
                context.messages.append(Message(role="user", content=f"Tool results: {json.dumps(tool_results)}"))
                messages = context.to_messages()
                response = await self.provider.generate(messages, stream=False, **kwargs)
            
            # Save to session
            session.add_message(user_msg)
            session.add_message(Message(role="assistant", content=response))
            
            # Save to long-term memory if significant
            if len(prompt) > 50:  # Simple heuristic
                await self.memory_store.add_memory({
                    "content": prompt,
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id
                })
            
            return response
    
    async def _stream_generate(
        self, 
        messages: List[Dict[str, str]], 
        session: ConversationMemory,
        user_msg: Message,
        **kwargs
    ) -> AsyncIterator[str]:
        """Handle streaming generation with context management."""
        collected_response = []
        
        async for chunk in await self.provider.generate(messages, stream=True, **kwargs):
            collected_response.append(chunk)
            yield chunk
        
        # After streaming is complete, save the full response
        full_response = "".join(collected_response)
        
        # Save to session
        session.add_message(user_msg)
        session.add_message(Message(role="assistant", content=full_response))
        
        # Save to long-term memory if significant
        if len(user_msg.content) > 50:
            await self.memory_store.add_memory({
                "content": user_msg.content,
                "response": full_response,
                "timestamp": datetime.now().isoformat(),
                "session_id": session.session_id
            })
    
    async def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        results = []
        for call in tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("arguments", {})
            
            try:
                result = await self.tool_registry.execute(tool_name, **tool_args)
                results.append({
                    "tool": tool_name,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def chain(self, *operations: Callable) -> 'Chain':
        """Create a chain of operations."""
        return Chain(self, operations)


class Chain:
    """Represents a chain of operations."""
    
    def __init__(self, engine: ContextEngine, operations: tuple):
        self.engine = engine
        self.operations = operations
    
    async def run(self, initial_input: Any, **kwargs) -> Any:
        """Run the chain of operations."""
        result = initial_input
        for operation in self.operations:
            if isinstance(result, dict):
                result = await operation(**result)
            else:
                result = await operation(result)
        return result

