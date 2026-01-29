"""
Core Package

Provides:
- Document chunking
- Embedding generation
- Retrieval orchestration
- Response generation
"""
from .chunker import chunker
from .embedder import embedder
from .retriever import retriever
from .generator import generator

__all__ = [
    'chunker',
    'embedder',
    'retriever',
    'generator'
]
