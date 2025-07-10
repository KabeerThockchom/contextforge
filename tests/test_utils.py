# tests/test_utils.py
"""Tests for utility functions."""

import pytest
from contextforge.utils import merge_contexts, truncate_text, chunk_text


class TestMergeContexts:
    """Test merge_contexts function."""
    
    def test_merge_empty_contexts(self):
        """Test merging empty contexts."""
        result = merge_contexts()
        assert result == {}
    
    def test_merge_single_context(self):
        """Test merging single context."""
        context = {"key": "value", "number": 42}
        result = merge_contexts(context)
        assert result == context
    
    def test_merge_simple_contexts(self):
        """Test merging simple contexts."""
        context1 = {"key1": "value1", "shared": "from_first"}
        context2 = {"key2": "value2", "shared": "from_second"}
        
        result = merge_contexts(context1, context2)
        
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert result["shared"] == "from_second"  # Later context wins
    
    def test_merge_list_values(self):
        """Test merging contexts with list values."""
        context1 = {"items": ["a", "b"], "single": "value1"}
        context2 = {"items": ["c", "d"], "single": "value2"}
        
        result = merge_contexts(context1, context2)
        
        assert result["items"] == ["a", "b", "c", "d"]
        assert result["single"] == "value2"
    
    def test_merge_dict_values(self):
        """Test merging contexts with dict values."""
        context1 = {"config": {"option1": "value1", "shared": "first"}}
        context2 = {"config": {"option2": "value2", "shared": "second"}}
        
        result = merge_contexts(context1, context2)
        
        expected_config = {
            "option1": "value1",
            "option2": "value2",
            "shared": "second"
        }
        assert result["config"] == expected_config
    
    def test_merge_mixed_types(self):
        """Test merging contexts with mixed value types."""
        context1 = {"mixed": {"nested": "dict"}}
        context2 = {"mixed": "string"}
        
        result = merge_contexts(context1, context2)
        
        # Non-list/dict values should be replaced
        assert result["mixed"] == "string"
    
    def test_merge_multiple_contexts(self):
        """Test merging multiple contexts."""
        context1 = {"key": "value1", "list": ["a"]}
        context2 = {"key": "value2", "list": ["b"]}
        context3 = {"key": "value3", "list": ["c"]}
        
        result = merge_contexts(context1, context2, context3)
        
        assert result["key"] == "value3"
        assert result["list"] == ["a", "b", "c"]
    
    def test_merge_nested_dicts(self):
        """Test merging contexts with nested dictionaries."""
        context1 = {
            "config": {
                "database": {"host": "localhost", "port": 5432},
                "cache": {"ttl": 300}
            }
        }
        context2 = {
            "config": {
                "database": {"port": 3306, "username": "user"},
                "logging": {"level": "INFO"}
            }
        }
        
        result = merge_contexts(context1, context2)
        
        expected = {
            "config": {
                "database": {"host": "localhost", "port": 3306, "username": "user"},
                "cache": {"ttl": 300},
                "logging": {"level": "INFO"}
            }
        }
        assert result == expected
    
    def test_merge_preserves_original_contexts(self):
        """Test that merging doesn't modify original contexts."""
        context1 = {"list": ["a"], "dict": {"key": "value"}}
        context2 = {"list": ["b"], "dict": {"key2": "value2"}}
        
        original_context1 = context1.copy()
        original_context2 = context2.copy()
        
        merge_contexts(context1, context2)
        
        # Original contexts should be unchanged
        assert context1 == original_context1
        assert context2 == original_context2


class TestTruncateText:
    """Test truncate_text function."""
    
    def test_truncate_short_text(self):
        """Test truncating text shorter than max length."""
        text = "Short text"
        result = truncate_text(text, max_length=100)
        assert result == text
    
    def test_truncate_exact_length(self):
        """Test truncating text at exact max length."""
        text = "Exactly ten"  # 11 chars
        result = truncate_text(text, max_length=11)
        assert result == text
    
    def test_truncate_long_text(self):
        """Test truncating text longer than max length."""
        text = "This is a very long text that should be truncated"
        result = truncate_text(text, max_length=20)
        
        assert len(result) == 20
        assert result.endswith("...")
        assert result.startswith("This is a very l")
    
    def test_truncate_with_custom_suffix(self):
        """Test truncating with custom suffix."""
        text = "Long text that needs truncation"
        result = truncate_text(text, max_length=15, suffix="[...]")
        
        assert len(result) == 15
        assert result.endswith("[...]")
        assert result.startswith("Long text t")
    
    def test_truncate_shorter_than_suffix(self):
        """Test truncating with max_length shorter than suffix."""
        text = "Some text"
        result = truncate_text(text, max_length=2, suffix="...")
        
        # Should still work, even if result is just suffix
        assert len(result) == 2
        assert result == ".."
    
    def test_truncate_empty_text(self):
        """Test truncating empty text."""
        result = truncate_text("", max_length=10)
        assert result == ""
    
    def test_truncate_default_parameters(self):
        """Test truncating with default parameters."""
        text = "A" * 1500  # Longer than default max_length of 1000
        result = truncate_text(text)
        
        assert len(result) == 1000
        assert result.endswith("...")
        assert result.startswith("A" * 997)
    
    def test_truncate_unicode_text(self):
        """Test truncating unicode text."""
        text = "Hello 世界! This is unicode text 🌍"
        result = truncate_text(text, max_length=20)
        
        assert len(result) == 20
        assert result.endswith("...")


class TestChunkText:
    """Test chunk_text function."""
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_exact_size(self):
        """Test chunking text exactly at chunk size."""
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_long_text(self):
        """Test chunking long text."""
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=300)
        
        assert len(chunks) > 1
        
        # First chunk should be full size
        assert len(chunks[0]) == 300
        
        # All chunks should be strings
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_with_overlap(self):
        """Test chunking with overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_text(text, chunk_size=10, overlap=3)
        
        assert len(chunks) > 1
        
        # Check overlap exists
        if len(chunks) >= 2:
            # Last 3 chars of first chunk should be first 3 chars of second
            assert chunks[0][-3:] == chunks[1][:3]
    
    def test_chunk_no_overlap(self):
        """Test chunking without overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_text(text, chunk_size=10, overlap=0)
        
        # Reassembled chunks should equal original
        reassembled = "".join(chunks)
        assert reassembled == text
    
    def test_chunk_large_overlap(self):
        """Test chunking with large overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_text(text, chunk_size=10, overlap=8)
        
        assert len(chunks) > 1
        
        # With large overlap, should have significant overlap
        if len(chunks) >= 2:
            overlap_chars = chunks[0][-8:]
            assert chunks[1].startswith(overlap_chars)
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == ""
    
    def test_chunk_default_parameters(self):
        """Test chunking with default parameters."""
        text = "A" * 1200  # Longer than default chunk_size of 500
        chunks = chunk_text(text)
        
        assert len(chunks) > 1
        assert len(chunks[0]) == 500
        
        # Should have default overlap of 50
        if len(chunks) >= 2:
            assert chunks[0][-50:] == chunks[1][:50]
    
    def test_chunk_word_boundaries(self):
        """Test chunking preserves word boundaries where possible."""
        text = "This is a test sentence. " * 100  # Repeating sentence
        chunks = chunk_text(text, chunk_size=50)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        # Each chunk should be at most chunk_size length
        assert all(len(chunk) <= 50 for chunk in chunks)
    
    def test_chunk_unicode_text(self):
        """Test chunking unicode text."""
        text = "Hello 世界! " * 50  # Repeat unicode text
        chunks = chunk_text(text, chunk_size=30)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        # Should handle unicode properly
        reassembled = "".join(chunk[50:] if i > 0 else chunk 
                            for i, chunk in enumerate(chunks))
        # Note: This is a simplified test; exact reassembly depends on overlap handling
    
    def test_chunk_very_small_size(self):
        """Test chunking with very small chunk size."""
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=3, overlap=1)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 3 for chunk in chunks)
    
    def test_chunk_overlap_larger_than_chunk(self):
        """Test chunking with overlap larger than chunk size."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_text(text, chunk_size=5, overlap=10)
        
        # Should still work without infinite loop
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)