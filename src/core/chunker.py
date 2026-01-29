"""
Smart Document Chunker

Implements multiple chunking strategies:
1. Fixed-size chunking with overlap
2. Semantic chunking (respects paragraphs/sentences)
3. Recursive chunking for hierarchical documents
"""
import re
import tiktoken
from typing import List, Dict
from loguru import logger

from config import config


class DocumentChunker:
    """
    Smart document chunker with multiple strategies
    
    Features:
    - Token counting
    - Overlap management
    - Metadata preservation
    - Semantic boundary respect
    """
    
    def __init__(self):
        """Initialize chunker with tiktoken encoder"""
        try:
            # Initialize token encoder (for token counting)
            self.encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            logger.info("✅ Chunker initialized with tiktoken")
        except Exception as e:
            logger.warning(f"Tiktoken initialization failed: {e}. Using character-based fallback")
            self.encoder = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Fallback: approximate as chars/4
            return len(text) // 4
    
    def chunk_fixed(
        self,
        text: str,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[Dict]:
        """
        Fixed-size chunking with overlap
        
        Args:
            text: Input text
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Skip empty chunks
                token_count = self.count_tokens(chunk_text)
                
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "token_count": token_count,
                    "metadata": {
                        "strategy": "fixed",
                        "start_pos": start,
                        "end_pos": end
                    }
                })
                chunk_index += 1
            
            # Move to next chunk with overlap
            start = end - overlap
            
            # Prevent infinite loop
            if end >= len(text):
                break
        
        logger.info(f"Fixed chunking: {len(chunks)} chunks created")
        return chunks
    
    def chunk_semantic(
        self,
        text: str,
        max_chunk_size: int = None,
        overlap: int = None
    ) -> List[Dict]:
        """
        Semantic chunking - respects paragraphs and sentences
        
        Strategy:
        1. Split by paragraphs
        2. Combine small paragraphs
        3. Split large paragraphs by sentences
        4. Ensure no chunk exceeds max_chunk_size
        
        Args:
            text: Input text
            max_chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        max_chunk_size = max_chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds max size
            if len(current_chunk) + len(para) > max_chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    token_count = self.count_tokens(current_chunk)
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_index": chunk_index,
                        "token_count": token_count,
                        "metadata": {
                            "strategy": "semantic",
                            "type": "paragraph_group"
                        }
                    })
                    chunk_index += 1
                
                # If single paragraph is too large, split by sentences
                if len(para) > max_chunk_size:
                    sentence_chunks = self._split_by_sentences(para, max_chunk_size)
                    for sent_chunk in sentence_chunks:
                        token_count = self.count_tokens(sent_chunk)
                        chunks.append({
                            "text": sent_chunk,
                            "chunk_index": chunk_index,
                            "token_count": token_count,
                            "metadata": {
                                "strategy": "semantic",
                                "type": "sentence_group"
                            }
                        })
                        chunk_index += 1
                    current_chunk = ""
                else:
                    current_chunk = para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            token_count = self.count_tokens(current_chunk)
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "token_count": token_count,
                "metadata": {
                    "strategy": "semantic",
                    "type": "paragraph_group"
                }
            })
        
        logger.info(f"Semantic chunking: {len(chunks)} chunks created")
        return chunks
    
    def _split_by_sentences(self, text: str, max_size: int) -> List[str]:
        """
        Split text by sentences, respecting max size
        
        Args:
            text: Input text
            max_size: Maximum characters per chunk
            
        Returns:
            List of sentence-based chunks
        """
        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) > max_size:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                if current:
                    current += " " + sentence
                else:
                    current = sentence
        
        if current:
            chunks.append(current.strip())
        
        return chunks
    
    def chunk_recursive(
        self,
        text: str,
        separators: List[str] = None,
        max_chunk_size: int = None
    ) -> List[Dict]:
        """
        Recursive chunking - tries multiple separators in order
        
        Strategy:
        1. Try to split by double newline (paragraphs)
        2. If chunks still too large, try single newline
        3. If still too large, try sentences
        4. If still too large, use fixed chunking
        
        Args:
            text: Input text
            separators: List of separators to try (in order)
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of chunk dictionaries
        """
        max_chunk_size = max_chunk_size or config.CHUNK_SIZE
        separators = separators or ["\n\n", "\n", ". ", " "]
        
        def split_text(text: str, sep_index: int = 0) -> List[str]:
            """Recursively split text"""
            if sep_index >= len(separators):
                # No more separators, use fixed chunking
                return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            
            separator = separators[sep_index]
            splits = text.split(separator)
            
            result_chunks = []
            current_chunk = ""
            
            for split in splits:
                if len(split) > max_chunk_size:
                    # This split is too large, try next separator
                    sub_chunks = split_text(split, sep_index + 1)
                    result_chunks.extend(sub_chunks)
                elif len(current_chunk) + len(split) > max_chunk_size:
                    if current_chunk:
                        result_chunks.append(current_chunk)
                    current_chunk = split
                else:
                    if current_chunk:
                        current_chunk += separator + split
                    else:
                        current_chunk = split
            
            if current_chunk:
                result_chunks.append(current_chunk)
            
            return result_chunks
        
        # Perform recursive splitting
        text_chunks = split_text(text)
        
        # Convert to chunk dictionaries
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk_text = chunk_text.strip()
            if chunk_text:
                token_count = self.count_tokens(chunk_text)
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": idx,
                    "token_count": token_count,
                    "metadata": {
                        "strategy": "recursive"
                    }
                })
        
        logger.info(f"Recursive chunking: {len(chunks)} chunks created")
        return chunks
    
    def chunk_document(
        self,
        text: str,
        strategy: str = None
    ) -> List[Dict]:
        """
        Chunk document using specified strategy
        
        Args:
            text: Input text
            strategy: Chunking strategy (semantic, fixed, recursive)
            
        Returns:
            List of chunk dictionaries
        """
        strategy = strategy or config.CHUNKING_STRATEGY
        
        if strategy == "fixed":
            return self.chunk_fixed(text)
        elif strategy == "semantic":
            return self.chunk_semantic(text)
        elif strategy == "recursive":
            return self.chunk_recursive(text)
        else:
            logger.warning(f"Unknown chunking strategy: {strategy}, using semantic")
            return self.chunk_semantic(text)


# Singleton instance
chunker = DocumentChunker()


if __name__ == "__main__":
    # Test chunker
    print("Testing Document Chunker...")
    
    test_text = """
    Python is a high-level programming language. It was created by Guido van Rossum.
    Python emphasizes code readability and simplicity.
    
    Python supports multiple programming paradigms. These include object-oriented,
    functional, and procedural programming.
    
    The language has a large standard library. This makes Python very versatile.
    Python is used in web development, data science, and automation.
    
    Many popular frameworks are built with Python. Django and Flask are examples
    for web development. NumPy and Pandas are used for data analysis.
    """
    
    # Test different strategies
    for strategy in ["fixed", "semantic", "recursive"]:
        print(f"\n{'='*60}")
        print(f"Testing {strategy.upper()} chunking:")
        print('='*60)
        
        chunks = chunker.chunk_document(test_text, strategy=strategy)
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Tokens: {chunk['token_count']}")
            print(f"  Text: {chunk['text'][:100]}...")
    
    print("\n✅ Document Chunker test completed")
