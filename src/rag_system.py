"""
SmartRetriever Pro - Production RAG System

Main orchestrator that ties together:
- Document processing
- Vector indexing
- Retrieval
- Generation
- Caching
- Logging
"""
import time
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

from config import config
from database import db_manager, faiss_store, redis_cache
from core import chunker, embedder, retriever, generator
from storage import file_manager


class SmartRetrieverPro:
    """
    Production-ready RAG system
    
    Features:
    - Multi-layer caching (Redis)
    - Vector search (FAISS)
    - Persistent storage (PostgreSQL)
    - Smart chunking
    - Comprehensive logging
    """
    
    def __init__(self):
        """Initialize SmartRetriever Pro"""
        logger.info("="*70)
        logger.info("🧠 Initializing SmartRetriever Pro...")
        logger.info("="*70)
        
        # Initialize components
        self.db = db_manager
        self.faiss = faiss_store
        self.cache = redis_cache
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.files = file_manager
        
        # Try to load existing FAISS index
        self.faiss.load()
        
        logger.info("="*70)
        logger.info("✅ SmartRetriever Pro initialized!")
        logger.info("="*70)
    
    def add_document(
        self,
        filepath: str,
        filename: str = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Add a document to the knowledge base
        
        Complete pipeline:
        1. Save file
        2. Extract text
        3. Chunk document
        4. Generate embeddings
        5. Store in PostgreSQL
        6. Index in FAISS
        
        Args:
            filepath: Path to document file
            filename: Optional filename override
            metadata: Optional document metadata
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        filename = filename or Path(filepath).name
        logger.info(f"\n{'='*70}")
        logger.info(f"📄 Processing document: {filename}")
        logger.info(f"{'='*70}")
        
        try:
            # Step 1: Save file
            logger.info("Step 1: Saving file...")
            stored_path, file_hash = self.files.save_file_from_path(filepath,filename)
            file_info = self.files.get_file_info(stored_path)
            
            # Step 2: Read text content
            logger.info("Step 2: Extracting text...")
            text_content = self.files.read_text_file(stored_path)
            
            if not text_content.strip():
                raise ValueError("Document is empty")
            
            # Step 3: Add document to database
            logger.info("Step 3: Adding to database...")
            # doc = self.db.add_document(
            #     filename=filename,
            #     filepath=stored_path,
            #     content=text_content,
            #     content_type=file_info['extension'],
            #     file_size=file_info['size_bytes'],
            #     metadata=metadata or {},
            # )

            # document_id = doc.id

            doc_id, doc_hash = self.db.add_document(
                            filename=filename,
                            filepath=stored_path,
                            content=text_content,
                            content_type=file_info['extension'],
                            file_size=file_info['size_bytes'],
                            metadata=metadata or {},
                        )
            
            # 🚫 If document already exists, skip re-processing
            existing_chunks = self.db.get_chunks_by_document(doc_id)
            if existing_chunks:
                logger.warning(
                    f"⚠️ Document {filename} already ingested "
                    f"({len(existing_chunks)} chunks). Skipping reprocessing."
                )
                return {
                    "success": True,
                    "document_id": doc_id,
                    "filename": filename,
                    "chunks_created": 0,
                    "processing_time": time.time() - start_time,
                    "skipped": True
                }

    
            # Step 4: Chunk document
            logger.info(f"Step 4: Chunking document (strategy: {config.CHUNKING_STRATEGY})...")
            chunks = self.chunker.chunk_document(text_content)
            logger.info(f"   Created {len(chunks)} chunks")
            
            # Step 5: Generate embeddings (batch)
            logger.info("Step 5: Generating embeddings...")
            chunk_texts = [c['text'] for c in chunks]
            embeddings = self.embedder.embed_batch(chunk_texts)
            
            # Step 6: Store chunks and add to FAISS
            logger.info("Step 6: Storing chunks and indexing...")
            chunk_ids = []
            faiss_ids = []
            
            existing_chunks = self.db.get_stats()["total_chunks"]
            next_faiss_id = existing_chunks

            for chunk_data, embedding in zip(chunks, embeddings):
                # Get next FAISS ID
                # faiss_id = self.faiss.index.ntotal
                faiss_id = next_faiss_id
                next_faiss_id += 1
                
                # Add chunk to database
                chunk_id = self.db.add_chunk(
                    # document_id=doc.id,
                    document_id=doc_id,
                    chunk_index=chunk_data['chunk_index'],
                    content=chunk_data['text'],
                    token_count=chunk_data['token_count'],
                    faiss_id=faiss_id,
                    metadata=chunk_data['metadata']
                )
                
                # Record embedding metadata
                self.db.add_embedding(
                    chunk_id=chunk_id,
                    faiss_index_id=faiss_id,
                    embedding_model=config.AZURE_OPENAI_EMBEDDING_MODEL,
                    dimension=config.AZURE_OPENAI_EMBEDDING_DIMENSION
                )
                
                chunk_ids.append(chunk_id)
                faiss_ids.append(faiss_id)
            
            # Add all embeddings to FAISS at once
            import numpy as np
            embeddings_array = np.array(embeddings)
            self.faiss.add_vectors(embeddings_array, chunk_ids)
            
            # Save FAISS index
            self.faiss.save()
            
            elapsed = time.time() - start_time
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ Document processed successfully in {elapsed:.2f}s")
            logger.info(f"   Document ID: {doc_id}")
            logger.info(f"   Chunks created: {len(chunks)}")
            logger.info(f"   FAISS index size: {self.faiss.index.ntotal}")
            logger.info(f"{'='*70}\n")
            
            return {
                "success": True,
                "document_id": doc_id,
                "filename": filename,
                "chunks_created": len(chunks),
                "processing_time": elapsed,
                "file_hash": file_hash
            }
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            raise
    
    def add_documents_from_directory(
        self,
        directory_path: str,
        recursive: bool = True
    ) -> List[Dict]:
        """
        Add all documents from a directory
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search recursively
            
        Returns:
            List of processing results
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all text files
        pattern = "**/*" if recursive else "*"
        files = []
        for ext in config.ALLOWED_EXTENSIONS:
            files.extend(directory.glob(f"{pattern}{ext}"))
        
        logger.info(f"\n📂 Found {len(files)} documents in {directory_path}")
        
        results = []
        for filepath in files:
            try:
                result = self.add_document(str(filepath))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {e}")
                results.append({
                    "success": False,
                    "filename": filepath.name,
                    "error": str(e)
                })
        
        successful = sum(1 for r in results if r.get('success'))
        logger.info(f"\n✅ Processed {successful}/{len(files)} documents successfully")
        
        return results
    
    # def query(
    #     self,
    #     question: str,
    #     top_k: int = None,
    #     use_cache: bool = True
    # ) -> Dict:
    def query(
            self,
            question: str,
            top_k: int = None,
            threshold: float = None,
            use_cache: bool = True
        )-> Dict:
        """
        Query the RAG system
        
        Complete pipeline:
        1. Check cache
        2. Retrieve relevant chunks
        3. Generate response
        4. Log query
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            use_cache: Whether to use cache
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"❓ Query: {question}")
        logger.info(f"{'='*70}")
        
        try:
            # Step 1: Retrieve relevant chunks
            chunks = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                threshold=threshold,
                use_cache=use_cache
            )

            logger.info(f"Chunks in rag_system : {chunks}")
            
            if not chunks:
                logger.warning("No relevant chunks found")
                return {
                    "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                    "sources": [],
                    "chunks_retrieved": 0,
                    "response_time_ms": (time.time() - start_time) * 1000
                }


            # Step 2: Generate response
            context_hash = self.retriever.get_context_hash(chunks)
            response_data = self.generator.generate_response(
                query=question,
                context_chunks=chunks,
                use_cache=use_cache,
                context_hash=context_hash
            )

            logger.info(f"Response for query in rag_system : {response_data}")
            
            # Step 3: Log query
            chunk_ids = [c['chunk_id'] for c in chunks]
            elapsed_ms = (time.time() - start_time) * 1000
            
            self.db.log_query(
                query_text=question,
                retrieved_chunks=chunk_ids,
                response_time_ms=int(elapsed_ms),
                cache_hit=response_data.get('cached', False),
                metadata={
                    "top_k": len(chunks),
                    "model": response_data.get('model')
                }
            )
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ Query processed in {elapsed_ms:.2f}ms")
            logger.info(f"   Chunks retrieved: {len(chunks)}")
            logger.info(f"   Cached: {response_data.get('cached', False)}")
            logger.info(f"{'='*70}\n")
            
            return {
                "answer": response_data['answer'],
                "sources": response_data['sources'],
                "chunks_retrieved": len(chunks),
                "chunks": chunks,
                "response_time_ms": elapsed_ms,
                "cached": response_data.get('cached', False),
                "tokens_used": response_data.get('tokens_used', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            raise
    
    # def delete_document(self, doc_id: int) -> bool:
    #     """
    #     Delete a document and all its chunks
        
    #     Args:
    #         doc_id: Document ID
            
    #     Returns:
    #         True if deleted
    #     """
    #     logger.info(f"🗑️  Deleting document ID: {doc_id}")
        
    #     # Get chunks before deletion
    #     chunks = self.db.get_chunks_by_document(doc_id)
    #     faiss_ids = [c.faiss_id for c in chunks]
        
    #     # Delete from database (cascades to chunks and embeddings)
    #     success = self.db.delete_document(doc_id)
        
    #     if success:
    #         # Note: FAISS doesn't support deletion, would need to rebuild index
    #         # For now, we mark this as a limitation
    #         logger.warning("⚠️  FAISS index not updated (rebuild required)")
    #         logger.info(f"✅ Document {doc_id} deleted from database")
        
    #     return success
    # def delete_document(self, doc_id: int):
    #     """
    #     Delete document and its chunks from DB.
    #     FAISS is NOT auto-updated (explicit rebuild required).
    #     """

    #     # 🔹 Step 1: fetch chunk data safely (NO ORM after this)
    #     chunks = self.db.get_chunks_by_document(doc_id)

    #     faiss_ids = [
    #         chunk["faiss_id"] if isinstance(chunk, dict) else chunk.faiss_id
    #         for chunk in chunks
    #     ]

    #     # 🔹 Step 2: delete from DB
    #     self.db.delete_document(doc_id)

    #     # 🔹 Step 3: log (FAISS handled separately)
    #     self.logger.info(
    #         f"Deleted document {doc_id} with {len(faiss_ids)} chunks"
    #     )

    #     return {
    #         "success": True,
    #         "deleted_document_id": doc_id,
    #         "faiss_ids": faiss_ids
    #     }

    # def delete_document(self, doc_id: int):
    #     """
    #     Delete document and chunks from DB.
    #     FAISS is NOT auto-updated.
    #     """

    #     # Step 1: fetch chunk metadata safely
    #     chunks = self.db.get_chunks_by_document(doc_id)

    #     faiss_ids = [c["faiss_id"] for c in chunks]

    #     # Step 2: delete document (DB handles cascade)
    #     self.db.delete_document(doc_id)

    #     self.logger.info(
    #         f"Deleted document {doc_id} with {len(faiss_ids)} chunks"
    #     )

    #     return {
    #         "success": True,
    #         "deleted_document_id": doc_id,
    #         "faiss_ids": faiss_ids
    #     }

    def delete_document(self, doc_id: int) -> Dict:
        """
        Delete document and chunks from DB.
        FAISS is NOT auto-updated.
        """

        chunks = self.db.get_chunks_by_document(doc_id)
        faiss_ids = [c["faiss_id"] for c in chunks]

        success = self.db.delete_document(doc_id)

        logger.info(
            f"🗑️ Deleted document {doc_id} with {len(faiss_ids)} chunks"
        )

        return {
            "success": success,
            "deleted_document_id": doc_id,
            "faiss_ids": faiss_ids,
        }


    # def rebuild_faiss_index(self):
    #     """
    #     Rebuild FAISS index from database
        
    #     Use this after deleting documents
    #     """
    #     logger.info("🔄 Rebuilding FAISS index...")
        
    #     # Clear current index
    #     self.faiss.clear()
        
    #     # Get all chunks with embeddings
    #     with self.db.get_session() as session:
    #         from database.postgres import Chunk
    #         chunks = session.query(Chunk).all()
            
    #         if not chunks:
    #             logger.info("No chunks to index")
    #             return
            
    #         # Re-generate embeddings for all chunks
    #         chunk_texts = [c.content for c in chunks]
    #         chunk_ids = [c.id for c in chunks]
            
    #         logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    #         embeddings = self.embedder.embed_batch(chunk_texts, use_cache=True)
            
    #         # Add to FAISS
    #         import numpy as np
    #         embeddings_array = np.array(embeddings)
    #         self.faiss.add_vectors(embeddings_array, chunk_ids)
            
    #         # Update FAISS IDs in database
    #         for chunk, faiss_id in zip(chunks, range(len(chunks))):
    #             chunk.faiss_id = faiss_id
            
    #         session.commit()
        
    #     # Save index
    #     self.faiss.save()
        
    #     logger.info(f"✅ FAISS index rebuilt ({self.faiss.index.ntotal} vectors)")
    
    # def rebuild_faiss_index(self):
    #     logger.info("🔄 Rebuilding FAISS index...")

    #     self.faiss.clear()

    #     with self.db.get_session() as session:
    #         from database.postgres import Chunk

    #         chunks = (
    #             session.query(Chunk)
    #             .order_by(Chunk.id)
    #             .all()
    #         )

    #         if not chunks:
    #             logger.info("No chunks to index")
    #             return

    #         chunk_texts = [c.content for c in chunks]

    #         logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    #         embeddings = self.embedder.embed_batch(chunk_texts, use_cache=True)

    #         import numpy as np
    #         embeddings_array = np.array(embeddings)

    #         # 🔥 FAISS ID = index position
    #         self.faiss.add_vectors(embeddings_array)

    #         # 🔥 Update DB to match FAISS
    #         for idx, chunk in enumerate(chunks):
    #             chunk.faiss_id = idx

    #         session.commit()

    #     self.faiss.save()

    #     logger.info(f"✅ FAISS index rebuilt ({self.faiss.index.ntotal} vectors)")

    def rebuild_faiss_index(self):
        logger.info("🔄 Rebuilding FAISS index...")

        self.faiss.clear()

        with self.db.get_session() as session:
            from database.postgres import Chunk

            chunks = (
                session.query(Chunk)
                .order_by(Chunk.id)
                .all()
            )

            if not chunks:
                logger.info("No chunks to index")
                return

            chunk_texts = [c.content for c in chunks]

            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedder.embed_batch(chunk_texts, use_cache=True)

            import numpy as np
            embeddings_array = np.array(embeddings)

            # 🔑 FAISS IDs = index position
            faiss_ids = list(range(len(chunks)))

            self.faiss.add_vectors(embeddings_array, faiss_ids)

            # 🔑 Sync DB with FAISS
            for chunk, faiss_id in zip(chunks, faiss_ids):
                chunk.faiss_id = faiss_id

            session.commit()

        self.faiss.save()

        logger.info(f"✅ FAISS index rebuilt ({self.faiss.index.ntotal} vectors)")


    def get_stats(self) -> Dict:
        """
        Get comprehensive system statistics
        
        Returns:
            Dictionary with all stats
        """
        return {
            "database": self.db.get_stats(),
            "faiss": self.faiss.get_stats(),
            "cache": self.cache.get_stats(),
            "files": self.files.get_stats(),
            "queries": self.db.get_query_stats(limit=10)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.cache.clear_all()
        logger.info("✅ All caches cleared")


# Singleton instance
rag = SmartRetrieverPro()


if __name__ == "__main__":
    # Show system stats
    print("\n" + "="*70)
    print("SmartRetriever Pro - System Status")
    print("="*70)
    
    stats = rag.get_stats()
    
    print(f"\nDatabase:")
    print(f"  Documents: {stats['database']['total_documents']}")
    print(f"  Chunks: {stats['database']['total_chunks']}")
    print(f"  Embeddings: {stats['database']['total_embeddings']}")
    
    print(f"\nFAISS Index:")
    print(f"  Vectors: {stats['faiss']['total_vectors']}")
    print(f"  Dimension: {stats['faiss']['dimension']}")
    print(f"  Type: {stats['faiss']['index_type']}")
    
    print(f"\nCache:")
    print(f"  Status: {stats['cache'].get('status', 'unknown')}")
    
    print(f"\nFiles:")
    print(f"  Stored: {stats['files']['storage_files']}")
    print(f"  Total Size: {stats['files']['total_size_mb']:.2f} MB")
    
    print("\n" + "="*70)
