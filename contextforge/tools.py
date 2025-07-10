# contextforge/tools.py
"""Tool management for function calling."""

from typing import Dict, Any, Callable, List, Optional
import inspect
import asyncio
from dataclasses import dataclass


@dataclass
class Tool:
    """Represents a callable tool."""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any]
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        else:
            return self.func(**kwargs)


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Callable:
        """
        Register a function as a tool.
        
        Can be used as a decorator:
        @tool_registry.register(description="Search the web")
        def search(query: str) -> str:
            ...
        """
        def decorator(f: Callable) -> Callable:
            tool_name = name or f.__name__
            tool_desc = description or f.__doc__ or f"Tool: {tool_name}"
            
            # Extract parameters from function signature
            sig = inspect.signature(f)
            parameters = {}
            
            for param_name, param in sig.parameters.items():
                param_info = {"type": "string"}  # Default type
                
                if param.annotation != inspect.Parameter.empty:
                    # Try to map Python types to JSON schema types
                    type_mapping = {
                        str: "string",
                        int: "integer",
                        float: "number",
                        bool: "boolean",
                        list: "array",
                        dict: "object"
                    }
                    param_info["type"] = type_mapping.get(param.annotation, "string")
                
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                else:
                    param_info["required"] = True
                
                parameters[param_name] = param_info
            
            tool = Tool(
                name=tool_name,
                description=tool_desc,
                func=f,
                parameters=parameters
            )
            
            self.tools[tool_name] = tool
            return f
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    async def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        return await tool.execute(**kwargs)
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all tools for LLM context."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]

