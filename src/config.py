# """
# Production Configuration for SmartRetriever Pro
# Supports Azure OpenAI, PostgreSQL, Redis, and FAISS
# """
# import os
# from typing import Optional
# from pydantic_settings import BaseSettings
# from pydantic import Field
# from dotenv import load_dotenv

# load_dotenv(override=True)


# class Config(BaseSettings):
#     """
#     Production-ready configuration using Pydantic for validation
    
#     Environment variables can be set in .env file or system environment
#     """
    
#     # ==================== Azure OpenAI Settings ====================
#     AZURE_OPENAI_ENDPOINT: str = Field(..., description="Azure OpenAI endpoint URL")
#     AZURE_OPENAI_API_KEY: str = Field(..., description="Azure OpenAI API key")
#     AZURE_OPENAI_API_VERSION: str = Field(default="2024-02-15-preview")
    
#     # Chat/Generation model
#     AZURE_OPENAI_DEPLOYMENT: str = Field(..., description="GPT deployment name")
    
#     # Embedding model
#     AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = Field(..., description="Embedding deployment name")
#     AZURE_OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-ada-002")
#     AZURE_OPENAI_EMBEDDING_DIMENSION: int = Field(default=1536, description="Embedding vector dimension")
    
#     # ==================== PostgreSQL Settings ====================
#     POSTGRES_HOST: str = Field(default="localhost")
#     POSTGRES_PORT: int = Field(default=5432)
#     POSTGRES_USER: str = Field(default="raguser")
#     POSTGRES_PASSWORD: str = Field(default="ragpass")
#     POSTGRES_DB: str = Field(default="smartretriever")
    
#     @property
#     def DATABASE_URL(self) -> str:
#         """Construct PostgreSQL connection URL"""
#         return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
#     # ==================== Redis Settings ====================
#     REDIS_HOST: str = Field(default="localhost")
#     REDIS_PORT: int = Field(default=6379)
#     REDIS_DB: int = Field(default=0)
#     REDIS_PASSWORD: Optional[str] = Field(default=None)
    
#     # Cache TTL (Time To Live) in seconds
#     CACHE_TTL_EMBEDDING: int = Field(default=86400, description="24 hours")
#     CACHE_TTL_RETRIEVAL: int = Field(default=3600, description="1 hour")
#     CACHE_TTL_RESPONSE: int = Field(default=1800, description="30 minutes")
    
#     @property
#     def REDIS_URL(self) -> str:
#         """Construct Redis connection URL"""
#         if self.REDIS_PASSWORD:
#             return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
#         return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
#     # ==================== FAISS Settings ====================
#     FAISS_INDEX_PATH: str = Field(default="data/faiss_index")
#     FAISS_INDEX_TYPE: str = Field(default="Flat", description="Flat, IVF, or HNSW")
    
#     # For IVF index (faster search, slight accuracy trade-off)
#     FAISS_NLIST: int = Field(default=100, description="Number of clusters for IVF")
#     FAISS_NPROBE: int = Field(default=10, description="Number of clusters to search")
    
#     # ==================== File Storage Settings ====================
#     STORAGE_PATH: str = Field(default="data/storage")
#     UPLOAD_PATH: str = Field(default="data/uploads")
#     PROCESSED_PATH: str = Field(default="data/processed")
    
#     MAX_FILE_SIZE_MB: int = Field(default=100)
#     ALLOWED_EXTENSIONS: list = Field(default=[".txt", ".pdf", ".md", ".docx"])
    
#     # ==================== Chunking Settings ====================
#     CHUNK_SIZE: int = Field(default=500, description="Characters per chunk")
#     CHUNK_OVERLAP: int = Field(default=50, description="Overlap between chunks")
#     CHUNKING_STRATEGY: str = Field(default="semantic", description="semantic, fixed, or recursive")
    
#     # Token limits (for safety)
#     MAX_TOKENS_PER_CHUNK: int = Field(default=512)
    
#     # ==================== RAG Settings ====================
#     TOP_K_RESULTS: int = Field(default=5, description="Documents to retrieve")
#     SIMILARITY_THRESHOLD: float = Field(default=0.3, description="Minimum similarity score")
    
#     # Re-ranking
#     ENABLE_RERANKING: bool = Field(default=False)
#     RERANK_TOP_N: int = Field(default=20, description="Candidates before re-ranking")
    
#     # ==================== Generation Settings ====================
#     TEMPERATURE: float = Field(default=0.7)
#     MAX_TOKENS: int = Field(default=800)
    
#     # ==================== Performance Settings ====================
#     BATCH_SIZE_EMBEDDING: int = Field(default=10, description="Batch size for embedding")
#     DB_POOL_SIZE: int = Field(default=10, description="PostgreSQL connection pool size")
#     DB_MAX_OVERFLOW: int = Field(default=20)
    
#     # ==================== Monitoring Settings ====================
#     ENABLE_METRICS: bool = Field(default=True)
#     ENABLE_QUERY_LOGGING: bool = Field(default=True)
#     LOG_LEVEL: str = Field(default="INFO", description="DEBUG, INFO, WARNING, ERROR")
    
#     # ==================== Development Settings ====================
#     DEBUG: bool = Field(default=False)
#     TESTING: bool = Field(default=False)
    
#     class Config:
#         env_file = ".env"
#         case_sensitive = True


# # Global config instance
# config = Config()


# def validate_config():
#     """
#     Validate configuration and print status
#     """
#     print("=" * 70)
#     print("🔧 SmartRetriever Pro - Configuration")
#     print("=" * 70)
    
#     print(f"\n✅ Azure OpenAI:")
#     print(f"   Endpoint: {config.AZURE_OPENAI_ENDPOINT[:30]}...")
#     print(f"   Deployment: {config.AZURE_OPENAI_DEPLOYMENT}")
#     print(f"   Embedding: {config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
    
#     print(f"\n✅ PostgreSQL:")
#     print(f"   Host: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}")
#     print(f"   Database: {config.POSTGRES_DB}")
    
#     print(f"\n✅ Redis:")
#     print(f"   Host: {config.REDIS_HOST}:{config.REDIS_PORT}")
    
#     print(f"\n✅ FAISS:")
#     print(f"   Index Type: {config.FAISS_INDEX_TYPE}")
#     print(f"   Index Path: {config.FAISS_INDEX_PATH}")
    
#     print(f"\n✅ RAG Settings:")
#     print(f"   Chunk Size: {config.CHUNK_SIZE} chars")
#     print(f"   Top-K: {config.TOP_K_RESULTS}")
#     print(f"   Chunking: {config.CHUNKING_STRATEGY}")
    
#     print("\n" + "=" * 70)


# if __name__ == "__main__":
#     validate_config()



"""
Production Configuration for SmartRetriever Pro
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict



class Config(BaseSettings):
    # model_config = SettingsConfigDict(
    #     env_file=".env",
    #     extra="allow"
    # )
    ALLOWED_EXTENSIONS: list[str] = [".txt", ".md", ".pdf"]
    # ==================== Azure OpenAI ====================
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str = "2024-12-01-preview"

    AZURE_OPENAI_DEPLOYMENT: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str
    AZURE_OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    AZURE_OPENAI_EMBEDDING_DIMENSION: int = 1536

    # ==================== Database ====================
    DATABASE_URL: str

    # ==================== Redis ====================
    REDIS_HOST: str = "redis"
    # REDIS_URL : str
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    @property
    def REDIS_URL(self) -> str:
        """Construct Redis connection URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    CACHE_TTL_EMBEDDING: int = 86400
    CACHE_TTL_RETRIEVAL: int = 3600
    CACHE_TTL_RESPONSE: int = 1800

    ENABLE_RERANKING: bool = False
    RERANK_TOP_N: int = 20

    # ==================== FAISS ====================
    FAISS_INDEX_PATH: str = "data/faiss_index"
    FAISS_INDEX_TYPE: str = "Flat"
    FAISS_NLIST: int = 100
    FAISS_NPROBE: int = 10

    # ==================== Storage ====================
    STORAGE_PATH: str = "data/storage"
    UPLOAD_PATH: str = "data/uploads"
    PROCESSED_PATH: str = "data/processed"
    MAX_FILE_SIZE_MB: int = 100

    # ==================== Chunking ====================
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    CHUNKING_STRATEGY: str = "semantic"
    MAX_TOKENS_PER_CHUNK: int = 512

    # ==================== RAG ====================
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    # ==================== Generation ====================
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 800

    # ==================== Performance ====================
    BATCH_SIZE_EMBEDDING: int = 10
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # ==================== Monitoring ====================
    ENABLE_METRICS: bool = True
    ENABLE_QUERY_LOGGING: bool = True
    LOG_LEVEL: str = "INFO"

    # ==================== Dev ====================
    DEBUG: bool = False
    TESTING: bool = False

    

    class Config:
        env_file = ".env"
        # env_file = ".env.local" if os.getenv("ENV") == "local" else ".env.docker"
        case_sensitive = True
        extra="allow"


config = Config()
