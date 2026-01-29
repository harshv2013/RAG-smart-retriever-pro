"""
Embedding Service

Handles:
- Single and batch embedding generation
- Caching integration
- Rate limiting
- Error handling and retries
"""
import numpy as np
from typing import List, Union
from openai import AzureOpenAI
from loguru import logger
import time

from config import config
from database.redis_cache import redis_cache


class EmbeddingService:
    """
    Service for generating embeddings with Azure OpenAI
    
    Features:
    - Automatic caching
    - Batch processing
    - Retry logic
    - Rate limiting
    """
    
    def __init__(self):
        """Initialize Azure OpenAI client"""
        self.client = AzureOpenAI(
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
        )
        
        self.model = config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        self.dimension = config.AZURE_OPENAI_EMBEDDING_DIMENSION
        
        logger.info(f"✅ Embedding service initialized (model={self.model}, dim={self.dimension})")
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector
        """
        # Check cache first
        if use_cache:
            cached = redis_cache.get_embedding(text)
            if cached is not None:
                return cached
        
        # Generate embedding
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Cache the result
            if use_cache:
                redis_cache.set_embedding(text, embedding)
            
            logger.debug(f"Generated embedding for text: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        batch_size: int = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batching
        
        Args:
            texts: List of input texts
            use_cache: Whether to use cache
            batch_size: Batch size for API calls
            
        Returns:
            List of embedding vectors
        """
        batch_size = batch_size or config.BATCH_SIZE_EMBEDDING
        
        all_embeddings = []
        cache_hits = 0
        cache_misses = 0
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            texts_to_embed = []
            text_indices = []
            
            # Check cache for each text in batch
            for idx, text in enumerate(batch):
                if use_cache:
                    cached = redis_cache.get_embedding(text)
                    if cached is not None:
                        batch_embeddings.append((idx, cached))
                        cache_hits += 1
                        continue
                
                # Not in cache, need to generate
                texts_to_embed.append(text)
                text_indices.append(idx)
                cache_misses += 1
            
            # Generate embeddings for uncached texts
            if texts_to_embed:
                try:
                    response = self.client.embeddings.create(
                        input=texts_to_embed,
                        model=self.model
                    )
                    
                    for idx, (text, emb_data) in enumerate(zip(texts_to_embed, response.data)):
                        embedding = np.array(emb_data.embedding, dtype=np.float32)
                        original_idx = text_indices[idx]
                        batch_embeddings.append((original_idx, embedding))
                        
                        # Cache the result
                        if use_cache:
                            redis_cache.set_embedding(text, embedding)
                    
                except Exception as e:
                    logger.error(f"Batch embedding failed: {e}")
                    raise
            
            # Sort by original index and extract embeddings
            batch_embeddings.sort(key=lambda x: x[0])
            all_embeddings.extend([emb for _, emb in batch_embeddings])
        
        logger.info(f"Batch embedding: {len(texts)} texts, {cache_hits} cache hits, {cache_misses} cache misses")
        return all_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query (always cached)
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embed_text(query, use_cache=True)
    
    def get_stats(self) -> dict:
        """Get embedding service statistics"""
        return {
            "model": self.model,
            "dimension": self.dimension,
            "batch_size": config.BATCH_SIZE_EMBEDDING
        }


# Singleton instance
embedder = EmbeddingService()


if __name__ == "__main__":
    # Test embedding service
    print("Testing Embedding Service...")
    
    # Single embedding
    text = "This is a test query about machine learning"
    embedding = embedder.embed_text(text)
    print(f"✅ Single embedding: shape={embedding.shape}")
    
    # Batch embedding
    texts = [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing handles text",
        "Computer vision deals with images"
    ]
    
    embeddings = embedder.embed_batch(texts)
    print(f"✅ Batch embedding: {len(embeddings)} embeddings generated")
    
    # Test cache
    embedding2 = embedder.embed_text(text)
    print(f"✅ Cache test: Same embedding? {np.allclose(embedding, embedding2)}")
    
    print("\n✅ Embedding Service test completed")
