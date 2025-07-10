# contextforge/utils.py
"""Utility functions."""

from typing import Dict, Any, List


def merge_contexts(*contexts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple context dictionaries."""
    merged = {}
    
    for context in contexts:
        for key, value in context.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                merged[key].extend(value)
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key].update(value)
            else:
                merged[key] = value
    
    return merged


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks