"""
Retrieval Service

Orchestrates the retrieval process:
1. Query embedding
2. FAISS similarity search
3. PostgreSQL metadata fetch
4. Result ranking and filtering
5. Caching
"""
from typing import List, Dict, Optional
import hashlib
from loguru import logger
import time

from config import config
from database.postgres import db_manager
from database.faiss_store import faiss_store
from database.redis_cache import redis_cache
from core.embedder import embedder


class RetrievalService:
    """
    Handles document retrieval using FAISS + PostgreSQL
    
    Pipeline:
    1. Check cache
    2. Embed query
    3. FAISS similarity search
    4. Fetch chunk metadata from PostgreSQL
    5. Filter by threshold
    6. Cache results
    """
    
    def __init__(self):
        """Initialize retrieval service"""
        self.top_k = config.TOP_K_RESULTS
        self.similarity_threshold = config.SIMILARITY_THRESHOLD
        
        logger.info(f"✅ Retrieval service initialized (top_k={self.top_k}, threshold={self.similarity_threshold})")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            use_cache: Whether to use cache
            
        Returns:
            List of retrieved chunks with metadata
        """
        start_time = time.time()
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        logger.info(f"🔍 Retrieving documents for: '{query[:50]}...'")
        
        # Check cache
        if use_cache:
            cached = redis_cache.get_retrieval(query)
            if cached is not None:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"✅ Cache hit! Retrieved in {elapsed:.2f}ms")
                return cached
        
        # Step 1: Embed query
        logger.debug("Embedding query...")
        query_embedding = embedder.embed_query(query)
        
        # Step 2: FAISS similarity search
        logger.debug(f"Searching FAISS index (top_k={top_k * 2})...")  # Get more, then filter
        chunk_ids, similarities = faiss_store.search(
            query_embedding,
            k=top_k * 2  # Retrieve more candidates for filtering
        )
        
        if not chunk_ids:
            logger.warning("No results found in FAISS index")
            return []
        
        # Step 3: Fetch chunk metadata from PostgreSQL
        logger.debug(f"Fetching metadata for {len(chunk_ids)} chunks...")
        chunks = db_manager.get_chunks_by_faiss_ids(chunk_ids)
        
        # Step 4: Build results with scores
        results = []
        for chunk, similarity in zip(chunks, similarities):
            if similarity >= threshold:
                # Get document metadata
                logger.info(f"chunkv: {chunk}")

                # doc = db_manager.get_document(chunk.document_id)
                doc = chunk['document']
                logger.info(f"doc : {doc}")

                results.append({
                    "chunk_id": chunk['id'],
                    "faiss_id": chunk['faiss_id'],
                    "content": chunk['content'],
                    "similarity": float(similarity),
                    "token_count": chunk['token_count'],
                    "chunk_index": chunk['chunk_index'],
                    "document": {
                        "id": doc['id'],
                        "filename": doc['filename'],
                        # "filepath": doc['filepath']
                    } if doc else None,
                    "chunk_metadata": chunk['chunk_metadata']
                })
        
        # Step 5: Sort by similarity and limit to top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:top_k]
        
        # Cache the results
        if use_cache and results:
            redis_cache.set_retrieval(query, results)
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"✅ Retrieved {len(results)} chunks in {elapsed:.2f}ms")
        
        # Log retrieval stats
        for i, result in enumerate(results[:3]):  # Log top 3
            logger.debug(f"  {i+1}. {result['document']['filename']} (similarity: {result['similarity']:.3f})")
        
        return results
    
    def retrieve_by_document(
        self,
        document_id: int,
        top_k: int = None
    ) -> List[Dict]:
        """
        Retrieve all chunks from a specific document
        
        Args:
            document_id: Document ID
            top_k: Maximum chunks to return
            
        Returns:
            List of chunks from the document
        """
        chunks = db_manager.get_chunks_by_document(document_id)
        
        if top_k:
            chunks = chunks[:top_k]
        
        results = []
        for chunk in chunks:
            doc = db_manager.get_document(chunk.document_id)
            results.append({
                "chunk_id": chunk.id,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "chunk_index": chunk.chunk_index,
                "document": {
                    "id": doc.id,
                    "filename": doc.filename,
                    "filepath": doc.filepath
                } if doc else None,
                "metadata": chunk.chunk_metadata
            })
        
        logger.info(f"✅ Retrieved {len(results)} chunks from document {document_id}")
        return results
    
    def get_context_hash(self, results: List[Dict]) -> str:
        """
        Generate hash of retrieved context for caching
        
        Args:
            results: Retrieved chunks
            
        Returns:
            Hash string
        """
        content = "".join([r['content'] for r in results])
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def rerank_results(
        self,
        query: str,
        results: List[Dict],
        method: str = "simple"
    ) -> List[Dict]:
        """
        Re-rank retrieval results
        
        Args:
            query: Original query
            results: Initial results
            method: Re-ranking method (simple, cross_encoder)
            
        Returns:
            Re-ranked results
        """
        if method == "simple":
            # Simple re-ranking: boost recent documents, penalize duplicates
            seen_docs = set()
            reranked = []
            
            for result in results:
                doc_id = result['document']['id']
                
                # Boost score for first occurrence
                if doc_id not in seen_docs:
                    result['similarity'] *= 1.1
                    seen_docs.add(doc_id)
                else:
                    result['similarity'] *= 0.9
                
                reranked.append(result)
            
            # Re-sort
            reranked.sort(key=lambda x: x['similarity'], reverse=True)
            return reranked
        
        # Add more sophisticated re-ranking methods here
        return results
    
    def get_stats(self) -> Dict:
        """Get retrieval statistics"""
        return {
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "faiss_index_size": faiss_store.index.ntotal if faiss_store.index else 0
        }


# Singleton instance
retriever = RetrievalService()


if __name__ == "__main__":
    # Test retrieval service
    print("Testing Retrieval Service...")
    
    # This requires database and FAISS to be set up
    # Just show the interface
    
    stats = retriever.get_stats()
    print(f"Retrieval Stats: {stats}")
    
    print("\n✅ Retrieval Service interface ready")
