"""
Redis Cache Manager

Handles:
- Query embedding caching
- Retrieval results caching
- Generated response caching
- Cache invalidation
"""
import redis
import json
import hashlib
import numpy as np
from typing import Optional, List, Any
from loguru import logger

from config import config


class RedisCache:
    """
    Redis-based caching system for RAG pipeline
    
    Caches:
    1. Embeddings (24h TTL)
    2. Retrieval results (1h TTL)
    3. Generated responses (30min TTL)
    """
    
    def __init__(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                config.REDIS_URL,
                decode_responses=False,  # We'll handle encoding/decoding
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Redis connected: {config.REDIS_HOST}:{config.REDIS_PORT}")
            
        except redis.ConnectionError as e:
            logger.warning(f"⚠️  Redis connection failed: {e}")
            logger.warning("Cache will be disabled")
            self.redis_client = None
    
    def _is_available(self) -> bool:
        """Check if Redis is available"""
        return self.redis_client is not None
    
    def _generate_key(self, prefix: str, text: str) -> str:
        """
        Generate cache key
        
        Args:
            prefix: Key prefix (e.g., 'emb', 'ret', 'resp')
            text: Text to hash
            
        Returns:
            Cache key
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{prefix}:{text_hash}"
    
    # ==================== Embedding Cache ====================
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if not cached
        """
        if not self._is_available():
            return None
        
        try:
            key = self._generate_key("emb", text)
            cached = self.redis_client.get(key)
            
            if cached:
                # Deserialize numpy array
                embedding = np.frombuffer(cached, dtype=np.float32)
                logger.debug(f"✅ Cache hit: embedding for '{text[:30]}...'")
                return embedding
            
            logger.debug(f"Cache miss: embedding for '{text[:30]}...'")
            return None
            
        except Exception as e:
            logger.error(f"Redis get_embedding error: {e}")
            return None
    
    def set_embedding(self, text: str, embedding: np.ndarray):
        """
        Cache embedding
        
        Args:
            text: Input text
            embedding: Embedding vector
        """
        if not self._is_available():
            return
        
        try:
            key = self._generate_key("emb", text)
            
            # Serialize numpy array
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            # Set with TTL
            self.redis_client.setex(
                key,
                config.CACHE_TTL_EMBEDDING,
                embedding_bytes
            )
            
            logger.debug(f"✅ Cached: embedding for '{text[:30]}...'")
            
        except Exception as e:
            logger.error(f"Redis set_embedding error: {e}")
    
    # ==================== Retrieval Cache ====================
    
    def get_retrieval(self, query: str) -> Optional[List[dict]]:
        """
        Get cached retrieval results
        
        Args:
            query: Query text
            
        Returns:
            List of retrieved chunks or None
        """
        if not self._is_available():
            return None
        
        try:
            key = self._generate_key("ret", query)
            cached = self.redis_client.get(key)
            
            if cached:
                results = json.loads(cached.decode('utf-8'))
                logger.debug(f"✅ Cache hit: retrieval for '{query[:30]}...'")
                return results
            
            logger.debug(f"Cache miss: retrieval for '{query[:30]}...'")
            return None
            
        except Exception as e:
            logger.error(f"Redis get_retrieval error: {e}")
            return None
    
    def set_retrieval(self, query: str, results: List[dict]):
        """
        Cache retrieval results
        
        Args:
            query: Query text
            results: Retrieved chunks
        """
        if not self._is_available():
            return
        
        try:
            key = self._generate_key("ret", query)
            
            # Serialize results
            results_json = json.dumps(results)
            
            # Set with TTL
            self.redis_client.setex(
                key,
                config.CACHE_TTL_RETRIEVAL,
                results_json
            )
            
            logger.debug(f"✅ Cached: retrieval for '{query[:30]}...'")
            
        except Exception as e:
            logger.error(f"Redis set_retrieval error: {e}")
    
    # ==================== Response Cache ====================
    
    def get_response(self, query: str, context_hash: str) -> Optional[str]:
        """
        Get cached response
        
        Args:
            query: Query text
            context_hash: Hash of retrieved context (to ensure same context)
            
        Returns:
            Generated response or None
        """
        if not self._is_available():
            return None
        
        try:
            key = self._generate_key("resp", f"{query}:{context_hash}")
            cached = self.redis_client.get(key)
            
            if cached:
                response = cached.decode('utf-8')
                logger.debug(f"✅ Cache hit: response for '{query[:30]}...'")
                return response
            
            logger.debug(f"Cache miss: response for '{query[:30]}...'")
            return None
            
        except Exception as e:
            logger.error(f"Redis get_response error: {e}")
            return None
    
    def set_response(self, query: str, context_hash: str, response: str):
        """
        Cache generated response
        
        Args:
            query: Query text
            context_hash: Hash of retrieved context
            response: Generated response
        """
        if not self._is_available():
            return
        
        try:
            key = self._generate_key("resp", f"{query}:{context_hash}")
            
            # Set with TTL
            self.redis_client.setex(
                key,
                config.CACHE_TTL_RESPONSE,
                response.encode('utf-8')
            )
            
            logger.debug(f"✅ Cached: response for '{query[:30]}...'")
            
        except Exception as e:
            logger.error(f"Redis set_response error: {e}")
    
    # ==================== Utility Methods ====================
    
    def invalidate_pattern(self, pattern: str):
        """
        Invalidate all keys matching pattern
        
        Args:
            pattern: Redis key pattern (e.g., 'emb:*')
        """
        if not self._is_available():
            return
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"✅ Invalidated {len(keys)} keys matching '{pattern}'")
        except Exception as e:
            logger.error(f"Redis invalidate_pattern error: {e}")
    
    def clear_all(self):
        """Clear all cache"""
        if not self._is_available():
            return
        
        try:
            self.redis_client.flushdb()
            logger.info("✅ All cache cleared")
        except Exception as e:
            logger.error(f"Redis clear_all error: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self._is_available():
            return {"status": "unavailable"}
        
        try:
            info = self.redis_client.info()
            
            # Count keys by prefix
            emb_keys = len(self.redis_client.keys("emb:*"))
            ret_keys = len(self.redis_client.keys("ret:*"))
            resp_keys = len(self.redis_client.keys("resp:*"))
            
            return {
                "status": "available",
                "total_keys": info.get('db0', {}).get('keys', 0),
                "embedding_keys": emb_keys,
                "retrieval_keys": ret_keys,
                "response_keys": resp_keys,
                "memory_used_mb": info.get('used_memory', 0) / 1024 / 1024,
                "hit_rate": info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1), 1)
            }
        except Exception as e:
            logger.error(f"Redis get_stats error: {e}")
            return {"status": "error", "error": str(e)}


# Singleton instance
redis_cache = RedisCache()


if __name__ == "__main__":
    # Test Redis cache
    print("Testing Redis Cache...")
    
    # Test embedding cache
    test_text = "This is a test query"
    test_embedding = np.random.rand(1536).astype(np.float32)
    
    redis_cache.set_embedding(test_text, test_embedding)
    cached_embedding = redis_cache.get_embedding(test_text)
    
    if cached_embedding is not None:
        print(f"✅ Embedding cache works: {np.allclose(test_embedding, cached_embedding)}")
    
    # Test retrieval cache
    test_results = [
        {"chunk_id": 1, "content": "Test chunk 1", "score": 0.9},
        {"chunk_id": 2, "content": "Test chunk 2", "score": 0.8}
    ]
    
    redis_cache.set_retrieval(test_text, test_results)
    cached_results = redis_cache.get_retrieval(test_text)
    
    if cached_results:
        print(f"✅ Retrieval cache works: {len(cached_results)} results")
    
    # Test response cache
    test_response = "This is a test response"
    context_hash = "test_hash"
    
    redis_cache.set_response(test_text, context_hash, test_response)
    cached_response = redis_cache.get_response(test_text, context_hash)
    
    if cached_response:
        print(f"✅ Response cache works: '{cached_response}'")
    
    # Get stats
    stats = redis_cache.get_stats()
    print(f"\nCache Stats: {json.dumps(stats, indent=2)}")
    
    print("\n✅ Redis Cache test completed")
