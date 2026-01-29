"""
PostgreSQL Database Models and Operations

Handles:
- Document metadata storage
- Chunk information
- Embedding metadata
- Query logs
"""
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, 
    TIMESTAMP, Boolean, JSON, ForeignKey, ARRAY, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from contextlib import contextmanager
from typing import List, Dict, Optional, Generator
import hashlib
from loguru import logger

from config import config

Base = declarative_base()


# ==================== Database Models ====================

class Document(Base):
    """
    Stores document metadata
    
    Each document can have multiple chunks
    """
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(Text, nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    content_type = Column(String(50))
    file_size = Column(Integer)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    doc_metadata = Column(JSON, default={})
    
    # Relationship to chunks
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}')>"


class Chunk(Base):
    """
    Stores individual chunks of documents
    
    Each chunk has:
    - Link to parent document
    - Content text
    - Position in document
    - FAISS vector index ID
    """
    __tablename__ = 'chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Position in document
    content = Column(Text, nullable=False)
    chunk_hash = Column(String(64), index=True)
    token_count = Column(Integer)
    faiss_id = Column(Integer, unique=True, index=True)  # ID in FAISS index
    created_at = Column(TIMESTAMP, server_default=func.now())
    chunk_metadata = Column(JSON, default={})
    
    # Relationship to document
    document = relationship("Document", back_populates="chunks")
    
    # Relationship to embedding
    embedding = relationship("Embedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_document_chunk', 'document_id', 'chunk_index'),
    )
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"


class Embedding(Base):
    """
    Stores embedding metadata
    
    Actual vectors stored in FAISS, this table just tracks metadata
    """
    __tablename__ = 'embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, ForeignKey('chunks.id', ondelete='CASCADE'), nullable=False, unique=True)
    faiss_index_id = Column(Integer, unique=True, nullable=False)
    embedding_model = Column(String(100), nullable=False)
    dimension = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationship to chunk
    chunk = relationship("Chunk", back_populates="embedding")
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, chunk_id={self.chunk_id}, model='{self.embedding_model}')>"


class QueryLog(Base):
    """
    Logs all queries for analytics and monitoring
    """
    __tablename__ = 'query_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    retrieved_chunks = Column(ARRAY(Integer), default=[])
    response_time_ms = Column(Integer)
    cache_hit = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    query_metadata = Column(JSON, default={})
    
    def __repr__(self):
        return f"<QueryLog(id={self.id}, query='{self.query_text[:30]}...')>"


# ==================== Database Manager ====================

class DatabaseManager:
    """
    Manages PostgreSQL database connections and operations
    
    Features:
    - Connection pooling
    - Context managers for sessions
    - CRUD operations for all models
    """
    
    def __init__(self):
        """Initialize database connection"""
        self.engine = create_engine(
            config.DATABASE_URL,
            pool_size=config.DB_POOL_SIZE,
            max_overflow=config.DB_MAX_OVERFLOW,
            echo=config.DEBUG,
            pool_pre_ping=True  # Verify connections before using
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # logger.info(f"✅ Database connected: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}")
        logger.info("✅ Database connected via DATABASE_URL")

    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("✅ Database tables created")
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("⚠️  Database tables dropped")
    
    @contextmanager
    def get_session(self) -> Generator:
        """
        Context manager for database sessions
        
        Usage:
            with db.get_session() as session:
                doc = session.query(Document).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    # ==================== Document Operations ====================
    
    def add_document(
        self,
        filename: str,
        filepath: str,
        content: str,
        content_type: str = "text/plain",
        file_size: int = 0,
        metadata: Dict = None
    ) -> Document:
        """
        Add a new document
        
        Returns Document object or existing document if hash matches
        """
        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        
        with self.get_session() as session:
            # Check if document already exists
            existing = session.query(Document).filter(Document.file_hash == file_hash).first()
            if existing:
                logger.info(f"Document already exists: {filename} (hash: {file_hash[:8]}...)")
                # return existing
                return existing.id, existing.file_hash
            
            
            # Create new document
            doc = Document(
                filename=filename,
                filepath=filepath,
                file_hash=file_hash,
                content_type=content_type,
                file_size=file_size,
                doc_metadata=metadata or {}
            )
            session.add(doc)
            session.flush()  # Get the ID
            
            logger.info(f"✅ Document added: {filename} (ID: {doc.id})")
            return doc.id, doc.file_hash
    
    def get_document(self, document_id: int) -> Optional[Document]:
        """Get document by ID"""
        with self.get_session() as session:
            return session.query(Document).filter(Document.id == document_id).first()
    
    # def get_all_documents(self) -> List[Document]:
    #     """Get all documents"""
    #     with self.get_session() as session:
    #         return session.query(Document).all()

    # def get_all_documents(self):
    #     with self.get_session() as session:
    #         docs = session.query(Document).all()
    #         return [
    #             {
    #                 "id": d.id,
    #                 "filename": d.filename,
    #                 "filepath": d.filepath,
    #                 "content_type": d.content_type,
    #                 "created_at": d.created_at
    #             }
    #             for d in docs
    #         ]

    # def get_all_documents(self):
    #     with self.session_scope() as session:
    #         docs = session.query(Document).all()

    #         return [
    #             {
    #                 "id": d.id,
    #                 "filename": d.filename,
    #                 "chunks_count": len(d.chunks)
    #             }
    #             for d in docs
    #         ]

    def get_all_documents(self) -> List[Dict]:
        session = self.SessionLocal()
        try:
            docs = session.query(Document).all()
            return [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "filepath": d.filepath,
                    "chunks_count": session.query(Chunk)
                        .filter(Chunk.document_id == d.id)
                        .count(),
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in docs
            ]
        finally:
            session.close()

    # def delete_document(self, document_id: int) -> bool:
    #     """Delete document and all its chunks"""
    #     with self.get_session() as session:
    #         doc = session.query(Document).filter(Document.id == document_id).first()
    #         if doc:
    #             session.delete(doc)
    #             logger.info(f"✅ Document deleted: {doc.filename} (ID: {document_id})")
    #             return True
    #         return False
    
    # def delete_document(self, document_id: int):
    #     session = self.Session()
    #     try:
    #         doc = session.query(Document).filter(Document.id == document_id).first()
    #         if doc:
    #             session.delete(doc)
    #             session.commit()
    #     finally:
    #         session.close()

    def delete_document(self, document_id: int) -> bool:
        session = self.SessionLocal()
        try:
            doc = session.query(Document).filter(Document.id == document_id).first()
            if not doc:
                return False

            session.delete(doc)
            session.commit()

            logger.info(f"✅ Document deleted: {doc.filename} (ID: {document_id})")
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ==================== Chunk Operations ====================
    
    def add_chunk(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        token_count: int,
        faiss_id: int,
        metadata: Dict = None
    ) -> Chunk:
        """Add a chunk to a document"""
        chunk_hash = hashlib.sha256(content.encode()).hexdigest()
        
        with self.get_session() as session:
            chunk = Chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                chunk_hash=chunk_hash,
                token_count=token_count,
                faiss_id=faiss_id,
                chunk_metadata=metadata or {}
            )
            session.add(chunk)
            session.flush()
            return chunk.id
    
    # def get_chunks_by_document(self, document_id: int) -> List[Chunk]:
    #     """Get all chunks for a document"""
    #     with self.get_session() as session:
    #         return session.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).all()
    

    # def get_chunks_by_document(self, document_id: int):
    #     with self.session_scope() as session:
    #         rows = (
    #             session.query(Chunk)
    #             .filter(Chunk.document_id == document_id)
    #             .all()
    #         )

    #         return [
    #             {
    #                 "id": c.id,
    #                 "faiss_id": c.faiss_id,
    #                 "chunk_index": c.chunk_index,
    #                 "token_count": c.token_count,
    #             }
    #             for c in rows
    #         ]

    # def get_chunks_by_document(self, document_id: int):
    #     session = self.Session()
    #     try:
    #         rows = (
    #             session.query(Chunk)
    #             .filter(Chunk.document_id == document_id)
    #             .all()
    #         )

    #         # 🚨 convert ORM → dict IMMEDIATELY
    #         return [
    #             {
    #                 "id": c.id,
    #                 "faiss_id": c.faiss_id,
    #                 "chunk_index": c.chunk_index,
    #                 "token_count": c.token_count,
    #             }
    #             for c in rows
    #         ]

    #     finally:
    #         session.close()

    def get_chunks_by_document(self, document_id: int) -> List[Dict]:
        session = self.SessionLocal()
        try:
            rows = session.query(Chunk).filter(
                Chunk.document_id == document_id
            ).all()

            return [
                {
                    "id": c.id,
                    "faiss_id": c.faiss_id,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                }
                for c in rows
            ]
        finally:
            session.close()


    def get_chunk_by_faiss_id(self, faiss_id: int) -> Optional[Chunk]:
        """Get chunk by its FAISS index ID"""
        with self.get_session() as session:
            return session.query(Chunk).filter(Chunk.faiss_id == faiss_id).first()
    
    # def get_chunks_by_faiss_ids(self, faiss_ids: List[int]) -> List[Chunk]:
    #     """Get multiple chunks by FAISS IDs"""
    #     with self.get_session() as session:
    #         return session.query(Chunk).filter(Chunk.faiss_id.in_(faiss_ids)).all()
    

    def get_chunks_by_faiss_ids(self, faiss_ids: List[int]) -> List[Dict]:
        with self.get_session() as session:
            chunks = (
                session.query(Chunk)
                .join(Document)
                .filter(Chunk.faiss_id.in_(faiss_ids))
                .all()
            )

            # ✅ Convert ORM → plain dicts BEFORE session closes
            results = []
            for chunk in chunks:
                results.append({
                    "id": chunk.id,
                    "faiss_id": chunk.faiss_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "document": {
                        "id": chunk.document.id,
                        "filename": chunk.document.filename,
                    },
                    "chunk_metadata": chunk.chunk_metadata,
                })

            return results



    # ==================== Embedding Operations ====================
    
    def add_embedding(
        self,
        chunk_id: int,
        faiss_index_id: int,
        embedding_model: str,
        dimension: int
    ) -> Embedding:
        """Record embedding metadata"""
        with self.get_session() as session:
            embedding = Embedding(
                chunk_id=chunk_id,
                faiss_index_id=faiss_index_id,
                embedding_model=embedding_model,
                dimension=dimension
            )
            session.add(embedding)
            session.flush()
            return embedding
    
    # ==================== Query Log Operations ====================
    
    def log_query(
        self,
        query_text: str,
        retrieved_chunks: List[int],
        response_time_ms: int,
        cache_hit: bool = False,
        metadata: Dict = None
    ):
        """Log a query for analytics"""
        if not config.ENABLE_QUERY_LOGGING:
            return
        
        with self.get_session() as session:
            log = QueryLog(
                query_text=query_text,
                retrieved_chunks=retrieved_chunks,
                response_time_ms=response_time_ms,
                cache_hit=cache_hit,
                query_metadata=metadata or {}
            )
            session.add(log)
    
    def get_query_stats(self, limit: int = 100) -> Dict:
        """Get query statistics"""
        with self.get_session() as session:
            total_queries = session.query(QueryLog).count()
            cache_hits = session.query(QueryLog).filter(QueryLog.cache_hit == True).count()
            
            recent_queries = session.query(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit).all()
            
            avg_response_time = session.query(func.avg(QueryLog.response_time_ms)).scalar()
            
            return {
                "total_queries": total_queries,
                "cache_hits": cache_hits,
                "cache_hit_rate": cache_hits / total_queries if total_queries > 0 else 0,
                "avg_response_time_ms": float(avg_response_time) if avg_response_time else 0,
                "recent_queries": [
                    {
                        "query": q.query_text,
                        "response_time": q.response_time_ms,
                        "cache_hit": q.cache_hit,
                        "timestamp": q.created_at.isoformat()
                    }
                    for q in recent_queries
                ]
            }
    
    # ==================== Utility Operations ====================
    
    def get_stats(self) -> Dict:
        """Get overall database statistics"""
        with self.get_session() as session:
            total_docs = session.query(Document).count()
            total_chunks = session.query(Chunk).count()
            total_embeddings = session.query(Embedding).count()
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_embeddings": total_embeddings,
                "avg_chunks_per_doc": total_chunks / total_docs if total_docs > 0 else 0
            }


# Singleton instance
db_manager = DatabaseManager()


if __name__ == "__main__":
    # Test database connection
    db_manager.create_tables()
    stats = db_manager.get_stats()
    print(f"Database Stats: {stats}")
