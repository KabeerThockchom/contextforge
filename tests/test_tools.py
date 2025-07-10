# tests/test_tools.py
"""Tests for tool registry and execution functionality."""

import pytest
import asyncio
import inspect
from unittest.mock import Mock, AsyncMock
from contextforge.tools import Tool, ToolRegistry


class TestTool:
    """Test Tool class."""
    
    def test_tool_creation(self):
        """Test creating a tool."""
        def test_func(x: int) -> int:
            return x * 2
        
        tool = Tool(
            name="test_tool",
            description="Test tool",
            func=test_func,
            parameters={"x": {"type": "integer", "required": True}}
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"
        assert tool.func == test_func
        assert tool.parameters["x"]["type"] == "integer"
    
    @pytest.mark.asyncio
    async def test_tool_execute_sync(self):
        """Test executing a synchronous tool."""
        def multiply(x: int, y: int) -> int:
            return x * y
        
        tool = Tool(
            name="multiply",
            description="Multiply two numbers",
            func=multiply,
            parameters={}
        )
        
        result = await tool.execute(x=3, y=4)
        assert result == 12
    
    @pytest.mark.asyncio
    async def test_tool_execute_async(self):
        """Test executing an asynchronous tool."""
        async def async_multiply(x: int, y: int) -> int:
            return x * y
        
        tool = Tool(
            name="async_multiply",
            description="Async multiply",
            func=async_multiply,
            parameters={}
        )
        
        result = await tool.execute(x=3, y=4)
        assert result == 12


class TestToolRegistry:
    """Test ToolRegistry class."""
    
    def test_registry_creation(self):
        """Test creating a tool registry."""
        registry = ToolRegistry()
        assert registry.tools == {}
    
    def test_register_function_decorator(self):
        """Test registering a function as a tool using decorator."""
        registry = ToolRegistry()
        
        @registry.register(description="Add two numbers")
        def add(x: int, y: int) -> int:
            return x + y
        
        assert "add" in registry.tools
        tool = registry.tools["add"]
        assert tool.name == "add"
        assert tool.description == "Add two numbers"
        assert tool.func == add
    
    def test_register_with_custom_name(self):
        """Test registering with custom name."""
        registry = ToolRegistry()
        
        @registry.register(name="custom_add", description="Custom add")
        def add_func(x: int, y: int) -> int:
            return x + y
        
        assert "custom_add" in registry.tools
        assert "add_func" not in registry.tools
        tool = registry.tools["custom_add"]
        assert tool.name == "custom_add"
    
    def test_register_without_description(self):
        """Test registering without explicit description."""
        registry = ToolRegistry()
        
        @registry.register
        def documented_func(x: int) -> int:
            """This is a documented function."""
            return x * 2
        
        tool = registry.tools["documented_func"]
        assert tool.description == "This is a documented function."
    
    def test_register_without_description_or_docstring(self):
        """Test registering without description or docstring."""
        registry = ToolRegistry()
        
        @registry.register
        def undocumented_func(x: int) -> int:
            return x * 2
        
        tool = registry.tools["undocumented_func"]
        assert tool.description == "Tool: undocumented_func"
    
    def test_register_parameter_extraction(self):
        """Test parameter extraction from function signature."""
        registry = ToolRegistry()
        
        @registry.register(description="Test function")
        def test_func(
            text: str,
            number: int = 10,
            flag: bool = False,
            items: list = None,
            config: dict = None
        ) -> str:
            return "test"
        
        tool = registry.tools["test_func"]
        params = tool.parameters
        
        # Check parameter types
        assert params["text"]["type"] == "string"
        assert params["number"]["type"] == "integer"
        assert params["flag"]["type"] == "boolean"
        assert params["items"]["type"] == "array"
        assert params["config"]["type"] == "object"
        
        # Check defaults
        assert params["text"]["required"] is True
        assert params["number"]["default"] == 10
        assert params["flag"]["default"] is False
    
    def test_register_unannotated_parameters(self):
        """Test registering function with unannotated parameters."""
        registry = ToolRegistry()
        
        @registry.register(description="Unannotated function")
        def unannotated_func(x, y=5):
            return x + y
        
        tool = registry.tools["unannotated_func"]
        params = tool.parameters
        
        # Should default to string type
        assert params["x"]["type"] == "string"
        assert params["y"]["type"] == "string"
        assert params["x"]["required"] is True
        assert params["y"]["default"] == 5
    
    def test_register_direct_call(self):
        """Test registering by direct call instead of decorator."""
        registry = ToolRegistry()
        
        def multiply(x: int, y: int) -> int:
            return x * y
        
        # Register directly
        registry.register(multiply, description="Multiply numbers")
        
        assert "multiply" in registry.tools
        tool = registry.tools["multiply"]
        assert tool.description == "Multiply numbers"
    
    def test_get_tool_exists(self):
        """Test getting an existing tool."""
        registry = ToolRegistry()
        
        @registry.register(description="Test tool")
        def test_tool() -> str:
            return "test"
        
        tool = registry.get_tool("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
    
    def test_get_tool_not_exists(self):
        """Test getting a non-existent tool."""
        registry = ToolRegistry()
        tool = registry.get_tool("nonexistent")
        assert tool is None
    
    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        """Test executing a synchronous tool."""
        registry = ToolRegistry()
        
        @registry.register(description="Add numbers")
        def add(x: int, y: int) -> int:
            return x + y
        
        result = await registry.execute("add", x=3, y=4)
        assert result == 7
    
    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        """Test executing an asynchronous tool."""
        registry = ToolRegistry()
        
        @registry.register(description="Async add")
        async def async_add(x: int, y: int) -> int:
            return x + y
        
        result = await registry.execute("async_add", x=3, y=4)
        assert result == 7
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing a non-existent tool."""
        registry = ToolRegistry()
        
        with pytest.raises(ValueError, match="Tool not found"):
            await registry.execute("nonexistent")
    
    @pytest.mark.asyncio
    async def test_execute_with_wrong_params(self):
        """Test executing tool with wrong parameters."""
        registry = ToolRegistry()
        
        @registry.register(description="Add numbers")
        def add(x: int, y: int) -> int:
            return x + y
        
        # Should raise TypeError for missing required parameter
        with pytest.raises(TypeError):
            await registry.execute("add", x=3)  # missing y
    
    def test_get_tool_descriptions(self):
        """Test getting tool descriptions."""
        registry = ToolRegistry()
        
        @registry.register(description="Add two numbers")
        def add(x: int, y: int) -> int:
            return x + y
        
        @registry.register(description="Multiply two numbers")
        def multiply(x: int, y: int) -> int:
            return x * y
        
        descriptions = registry.get_tool_descriptions()
        assert len(descriptions) == 2
        
        # Check structure
        for desc in descriptions:
            assert "name" in desc
            assert "description" in desc
            assert "parameters" in desc
        
        # Check content
        names = [desc["name"] for desc in descriptions]
        assert "add" in names
        assert "multiply" in names
    
    def test_get_tool_descriptions_empty(self):
        """Test getting tool descriptions when registry is empty."""
        registry = ToolRegistry()
        descriptions = registry.get_tool_descriptions()
        assert descriptions == []
    
    @pytest.mark.asyncio
    async def test_complex_tool_example(self):
        """Test a more complex tool example."""
        registry = ToolRegistry()
        
        @registry.register(description="Process text with options")
        def process_text(
            text: str,
            uppercase: bool = False,
            prefix: str = "",
            repeat: int = 1
        ) -> str:
            result = text
            if uppercase:
                result = result.upper()
            if prefix:
                result = prefix + result
            return result * repeat
        
        # Test with various parameter combinations
        result1 = await registry.execute("process_text", text="hello")
        assert result1 == "hello"
        
        result2 = await registry.execute(
            "process_text", 
            text="hello", 
            uppercase=True
        )
        assert result2 == "HELLO"
        
        result3 = await registry.execute(
            "process_text",
            text="hello",
            prefix=">>> ",
            repeat=2
        )
        assert result3 == ">>> hello>>> hello"
    
    def test_tool_with_complex_return_type(self):
        """Test tool with complex return type."""
        registry = ToolRegistry()
        
        @registry.register(description="Return complex data")
        def get_user_info(user_id: str) -> dict:
            return {
                "id": user_id,
                "name": "John Doe",
                "email": "john@example.com"
            }
        
        tool = registry.get_tool("get_user_info")
        assert tool is not None
        
        # Check that the function is stored correctly
        assert callable(tool.func)
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test tool error handling."""
        registry = ToolRegistry()
        
        @registry.register(description="Tool that raises error")
        def error_tool(should_error: bool) -> str:
            if should_error:
                raise ValueError("Test error")
            return "success"
        
        # Should not raise error when tool succeeds
        result = await registry.execute("error_tool", should_error=False)
        assert result == "success"
        
        # Should propagate error when tool fails
        with pytest.raises(ValueError, match="Test error"):
            await registry.execute("error_tool", should_error=True)
    
    def test_parameter_type_mapping(self):
        """Test that parameter types are correctly mapped."""
        registry = ToolRegistry()
        
        @registry.register(description="Test all types")
        def test_types(
            text: str,
            number: int,
            decimal: float,
            flag: bool,
            items: list,
            data: dict,
            unknown_type: object
        ) -> str:
            return "test"
        
        tool = registry.get_tool("test_types")
        params = tool.parameters
        
        assert params["text"]["type"] == "string"
        assert params["number"]["type"] == "integer"
        assert params["decimal"]["type"] == "number"
        assert params["flag"]["type"] == "boolean"
        assert params["items"]["type"] == "array"
        assert params["data"]["type"] == "object"
        assert params["unknown_type"]["type"] == "string"  # defaults to string