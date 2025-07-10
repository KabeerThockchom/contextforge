# ContextForge Test Suite

This directory contains comprehensive unit tests for the ContextForge package.

## Test Structure

The test suite is organized into separate modules that mirror the package structure:

- `test_core.py` - Tests for the core ContextEngine functionality
- `test_providers.py` - Tests for LLM provider implementations  
- `test_memory.py` - Tests for memory management components
- `test_tools.py` - Tests for tool registry and execution
- `test_retrieval.py` - Tests for retrieval components (RAG)
- `test_utils.py` - Tests for utility functions
- `conftest.py` - Test fixtures and configuration

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=contextforge --cov-report=html
```

### Using the Test Runner

```bash
# Run the comprehensive test suite
python run_tests.py
```

### Running Specific Tests

```bash
# Test a specific module
pytest tests/test_core.py -v

# Test a specific class
pytest tests/test_core.py::TestContextEngine -v

# Test a specific function
pytest tests/test_core.py::TestContextEngine::test_basic_generation -v

# Run tests with specific markers
pytest tests/ -m "not slow" -v
```

## Test Categories

### Unit Tests
- Test individual components in isolation
- Use mocking for external dependencies
- Fast execution
- High coverage of edge cases

### Integration Tests
- Test interaction between components
- Use real implementations where possible
- Marked with `@pytest.mark.integration`

### Async Tests
- Test asynchronous functionality
- Use `@pytest.mark.asyncio`
- Proper async/await patterns

## Key Testing Patterns

### Mocking External Dependencies

The tests extensively mock external dependencies:

```python
@patch('aiohttp.ClientSession')
async def test_provider_api_call(self, mock_session):
    # Mock HTTP responses
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"result": "success"})
    mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
```

### Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_function(self):
    result = await some_async_function()
    assert result == expected_value
```

### Using Fixtures

```python
@pytest.fixture
def context_engine(mock_provider):
    return ContextEngine(provider=mock_provider)

def test_something(context_engine):
    # Use the fixture
    result = context_engine.some_method()
    assert result is not None
```

## Test Coverage

The test suite aims for high coverage across all modules:

- **Core Engine**: Context management, session handling, streaming
- **Providers**: All LLM providers (OpenAI, Anthropic, Ollama)
- **Memory**: Both in-memory and persistent storage
- **Tools**: Registration, execution, parameter handling
- **Retrieval**: Document storage and similarity search
- **Utils**: All utility functions

## Fixtures

Common test fixtures are defined in `conftest.py`:

- `mock_provider` - Mock LLM provider for testing
- `context_engine` - Pre-configured ContextEngine instance
- `tool_registry` - Registry with sample tools
- `vector_retriever` - Retriever for testing RAG functionality
- `memory_store` - In-memory storage for testing
- `sqlite_memory_store` - SQLite storage for testing
- `sample_documents` - Sample documents for retrieval tests

## Best Practices

### Test Organization
- One test class per main class being tested
- Descriptive test method names
- Group related tests together
- Use docstrings to explain complex test scenarios

### Assertions
- Use specific assertions (`assert x == y` vs `assert x`)
- Test both success and failure cases
- Check return values, side effects, and error conditions

### Mocking
- Mock external dependencies (APIs, databases, file system)
- Use `Mock` for simple cases, `AsyncMock` for async functions
- Verify that mocks are called correctly

### Async Testing
- Use `@pytest.mark.asyncio` for async tests
- Properly await async functions
- Test both successful and error scenarios

## Common Test Patterns

### Testing Error Handling
```python
def test_error_handling(self):
    with pytest.raises(ValueError, match="expected error message"):
        function_that_should_raise_error()
```

### Testing State Changes
```python
def test_state_change(self):
    initial_state = obj.get_state()
    obj.modify_state()
    final_state = obj.get_state()
    assert final_state != initial_state
```

### Testing Collections
```python
def test_collection_operations(self):
    items = get_items()
    assert len(items) == 3
    assert all(isinstance(item, ExpectedType) for item in items)
    assert items[0].property == "expected_value"
```

## Running Tests in CI/CD

The tests are designed to run in automated environments:

```bash
# Install dependencies
pip install -r requirements-test.txt
pip install -e .

# Run tests with coverage
pytest tests/ --cov=contextforge --cov-report=xml --cov-fail-under=80

# Run specific test categories
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -m "unit"      # Run only unit tests
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the package is installed in development mode (`pip install -e .`)
2. **Async Test Failures**: Check that `pytest-asyncio` is installed and `@pytest.mark.asyncio` is used
3. **Mock Issues**: Verify that mocks are properly configured and reset between tests
4. **Database Tests**: Ensure temporary files are cleaned up after tests

### Debug Mode

```bash
# Run tests with verbose output
pytest tests/ -v -s

# Run tests with debugging
pytest tests/ --pdb

# Run a single test with maximum verbosity
pytest tests/test_core.py::TestContextEngine::test_basic_generation -vvv -s
```

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Add appropriate docstrings
3. Include both positive and negative test cases
4. Mock external dependencies
5. Update this README if adding new test categories or patterns

## Performance Testing

For performance-critical components, consider adding benchmarks:

```python
@pytest.mark.slow
def test_performance_benchmark(self):
    import time
    start = time.time()
    # Run performance-critical code
    end = time.time()
    assert end - start < 1.0  # Should complete within 1 second
```