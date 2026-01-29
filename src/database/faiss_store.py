# """
# FAISS Vector Store

# Handles:
# - Vector indexing with FAISS
# - Fast similarity search
# - Index persistence
# - Multiple index types (Flat, IVF, HNSW)
# """
# import os
# import numpy as np
# import faiss
# from typing import List, Tuple, Optional
# from loguru import logger
# import pickle
# from pathlib import Path

# from config import config


# class FAISSVectorStore:
#     """
#     FAISS-based vector store for efficient similarity search
    
#     Supports:
#     - Flat index (exact search, slower for large datasets)
#     - IVF index (approximate search, faster)
#     - HNSW index (hierarchical navigable small world, very fast)
#     """
    
#     def __init__(self, dimension: int = None):
#         """
#         Initialize FAISS vector store
        
#         Args:
#             dimension: Vector dimension (default from config)
#         """
#         self.dimension = dimension or config.AZURE_OPENAI_EMBEDDING_DIMENSION
#         self.index = None
#         self.id_map = []  # Maps FAISS index to database chunk IDs
#         self.index_type = config.FAISS_INDEX_TYPE
        
#         # Initialize index
#         self._create_index()
        
#         logger.info(f"✅ FAISS Vector Store initialized (dimension={self.dimension}, type={self.index_type})")
    
#     def _create_index(self):
#         """Create FAISS index based on configuration"""
#         if self.index_type == "Flat":
#             # Flat index: Exact search, slower for large datasets
#             self.index = faiss.IndexFlatL2(self.dimension)
#             logger.info("Created Flat index (exact search)")
            
#         elif self.index_type == "IVF":
#             # IVF index: Approximate search with clustering
#             # Good for 10k-10M vectors
#             quantizer = faiss.IndexFlatL2(self.dimension)
#             self.index = faiss.IndexIVFFlat(
#                 quantizer,
#                 self.dimension,
#                 config.FAISS_NLIST  # Number of clusters
#             )
#             logger.info(f"Created IVF index (nlist={config.FAISS_NLIST})")
            
#         elif self.index_type == "HNSW":
#             # HNSW index: Hierarchical navigable small world
#             # Very fast, good for large datasets
#             M = 32  # Number of connections per layer
#             self.index = faiss.IndexHNSWFlat(self.dimension, M)
#             logger.info(f"Created HNSW index (M={M})")
            
#         else:
#             raise ValueError(f"Unknown index type: {self.index_type}")
    
#     def train_index(self, vectors: np.ndarray):
#         """
#         Train the index (required for IVF)
        
#         Args:
#             vectors: Training vectors (shape: [n_vectors, dimension])
#         """
#         if self.index_type == "IVF":
#             if not self.index.is_trained:
#                 logger.info(f"Training IVF index with {len(vectors)} vectors...")
#                 self.index.train(vectors.astype('float32'))
#                 logger.info("✅ IVF index trained")
#         else:
#             logger.info(f"{self.index_type} index doesn't require training")
    
#     def add_vectors(self, vectors: np.ndarray, ids: List[int]):
#         """
#         Add vectors to the index
        
#         Args:
#             vectors: Embedding vectors (shape: [n_vectors, dimension])
#             ids: Corresponding chunk IDs from database
#         """
#         if vectors.shape[1] != self.dimension:
#             raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
        
#         # Ensure vectors are float32 (FAISS requirement)
#         vectors = vectors.astype('float32')
        
#         # Train index if needed
#         if self.index_type == "IVF" and not self.index.is_trained:
#             self.train_index(vectors)
        
#         # Add vectors
#         start_id = len(self.id_map)
#         self.index.add(vectors)
#         self.id_map.extend(ids)
        
#         logger.info(f"✅ Added {len(vectors)} vectors to FAISS index (total: {self.index.ntotal})")
    
#     def search(
#         self,
#         query_vector: np.ndarray,
#         k: int = 5,
#         nprobe: int = None
#     ) -> Tuple[List[int], List[float]]:
#         """
#         Search for similar vectors
        
#         Args:
#             query_vector: Query embedding (shape: [dimension])
#             k: Number of results to return
#             nprobe: Number of clusters to search (for IVF only)
            
#         Returns:
#             Tuple of (chunk_ids, distances)
#         """
#         if self.index.ntotal == 0:
#             logger.warning("FAISS index is empty")
#             return [], []
        
#         # Reshape query vector
#         query_vector = query_vector.reshape(1, -1).astype('float32')
        
#         # Set nprobe for IVF index
#         if self.index_type == "IVF" and nprobe:
#             self.index.nprobe = nprobe or config.FAISS_NPROBE
        
#         # Search
#         distances, indices = self.index.search(query_vector, k)
        
#         # Convert indices to chunk IDs
#         chunk_ids = [self.id_map[idx] for idx in indices[0] if idx < len(self.id_map)]
#         distances_list = distances[0].tolist()
        
#         # Convert L2 distances to similarity scores (0-1 range)
#         # similarity = 1 / (1 + distance)
#         similarities = [1 / (1 + d) for d in distances_list]
        
#         logger.debug(f"Search found {len(chunk_ids)} results")
        
#         return chunk_ids, similarities
    
#     def save(self, path: str = None):
#         """
#         Save index to disk
        
#         Args:
#             path: Save path (default: config.FAISS_INDEX_PATH)
#         """
#         path = path or config.FAISS_INDEX_PATH
#         os.makedirs(os.path.dirname(path), exist_ok=True)
        
#         # Save FAISS index
#         index_file = f"{path}/index.faiss"
#         faiss.write_index(self.index, index_file)
        
#         # Save ID map
#         id_map_file = f"{path}/id_map.pkl"
#         with open(id_map_file, 'wb') as f:
#             pickle.dump(self.id_map, f)
        
#         logger.info(f"✅ FAISS index saved to {path}")
    
#     def load(self, path: str = None) -> bool:
#         """
#         Load index from disk
        
#         Args:
#             path: Load path (default: config.FAISS_INDEX_PATH)
            
#         Returns:
#             True if loaded successfully, False otherwise
#         """
#         path = path or config.FAISS_INDEX_PATH
        
#         index_file = f"{path}/index.faiss"
#         id_map_file = f"{path}/id_map.pkl"
        
#         if not os.path.exists(index_file) or not os.path.exists(id_map_file):
#             logger.warning(f"FAISS index not found at {path}")
#             return False
        
#         try:
#             # Load FAISS index
#             self.index = faiss.read_index(index_file)
            
#             # Load ID map
#             with open(id_map_file, 'rb') as f:
#                 self.id_map = pickle.load(f)
            
#             logger.info(f"✅ FAISS index loaded from {path} ({self.index.ntotal} vectors)")
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to load FAISS index: {e}")
#             return False
    
#     def clear(self):
#         """Clear the index"""
#         self._create_index()
#         self.id_map = []
#         logger.info("✅ FAISS index cleared")
    
#     def get_stats(self) -> dict:
#         """Get index statistics"""
#         return {
#             "total_vectors": self.index.ntotal,
#             "dimension": self.dimension,
#             "index_type": self.index_type,
#             "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
#         }


# # Singleton instance
# faiss_store = FAISSVectorStore()


# if __name__ == "__main__":
#     # Test FAISS store
#     print("Testing FAISS Vector Store...")
    
#     # Create some dummy vectors
#     test_vectors = np.random.rand(100, config.AZURE_OPENAI_EMBEDDING_DIMENSION).astype('float32')
#     test_ids = list(range(100))
    
#     # Add vectors
#     faiss_store.add_vectors(test_vectors, test_ids)
    
#     # Search
#     query_vector = np.random.rand(config.AZURE_OPENAI_EMBEDDING_DIMENSION).astype('float32')
#     chunk_ids, similarities = faiss_store.search(query_vector, k=5)
    
#     print(f"Found {len(chunk_ids)} similar vectors")
#     for cid, sim in zip(chunk_ids, similarities):
#         print(f"  Chunk ID: {cid}, Similarity: {sim:.3f}")
    
#     # Save and load
#     faiss_store.save()
#     faiss_store.load()
    
#     print("\n✅ FAISS Vector Store test completed")



"""
FAISS Vector Store

Handles:
- Vector indexing with FAISS
- Fast similarity search
- Index persistence
- Multiple index types (Flat, IVF, HNSW)
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from loguru import logger

from config import config


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search
    """

    def __init__(self, dimension: int = None):
        self.dimension = dimension or config.AZURE_OPENAI_EMBEDDING_DIMENSION
        self.index_type = config.FAISS_INDEX_TYPE

        self.index: faiss.Index = None
        self.id_map: List[int] = []

        self._create_index()
        logger.info(
            f"✅ FAISS Vector Store initialized "
            f"(dimension={self.dimension}, type={self.index_type})"
        )

    # ------------------------------------------------------------------
    # Index creation
    # ------------------------------------------------------------------

    def _create_index(self):
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created Flat index (exact search)")

        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                config.FAISS_NLIST
            )
            logger.info(f"Created IVF index (nlist={config.FAISS_NLIST})")

        elif self.index_type == "HNSW":
            M = 32
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            logger.info(f"Created HNSW index (M={M})")

        else:
            raise ValueError(f"Unknown FAISS index type: {self.index_type}")

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def train_index(self, vectors: np.ndarray):
        if self.index_type == "IVF" and not self.index.is_trained:
            logger.info(f"Training IVF index with {len(vectors)} vectors...")
            self.index.train(vectors.astype("float32"))
            logger.info("✅ IVF index trained")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        vectors = vectors.astype("float32")

        if self.index_type == "IVF" and not self.index.is_trained:
            self.train_index(vectors)

        self.index.add(vectors)
        self.id_map.extend(ids)

        logger.info(
            f"✅ Added {len(vectors)} vectors to FAISS index "
            f"(total: {self.index.ntotal})"
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        nprobe: int = None
    ) -> Tuple[List[int], List[float]]:

        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return [], []

        query_vector = query_vector.reshape(1, -1).astype("float32")

        if self.index_type == "IVF":
            self.index.nprobe = nprobe or config.FAISS_NPROBE

        distances, indices = self.index.search(query_vector, k)

        chunk_ids = [
            self.id_map[i] for i in indices[0]
            if 0 <= i < len(self.id_map)
        ]

        similarities = [1 / (1 + d) for d in distances[0].tolist()]
        return chunk_ids, similarities

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = None):
        """
        Save FAISS index and ID map to disk.
        `path` is a DIRECTORY.
        """
        path = Path(path or config.FAISS_INDEX_PATH)
        path.mkdir(parents=True, exist_ok=True)

        index_file = path / "index.faiss"
        id_map_file = path / "id_map.pkl"

        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, saving empty index")

        faiss.write_index(self.index, str(index_file))

        with open(id_map_file, "wb") as f:
            pickle.dump(self.id_map, f)

        logger.info(f"💾 FAISS index saved to {path}")

    def load(self, path: str = None) -> bool:
        """
        Load FAISS index and ID map from disk.
        """
        path = Path(path or config.FAISS_INDEX_PATH)

        index_file = path / "index.faiss"
        id_map_file = path / "id_map.pkl"

        if not index_file.exists() or not id_map_file.exists():
            logger.warning(f"FAISS index not found at {path}")
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            with open(id_map_file, "rb") as f:
                self.id_map = pickle.load(f)

            logger.info(
                f"✅ FAISS index loaded from {path} "
                f"({self.index.ntotal} vectors)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.clear()
            return False

    def clear(self):
        self._create_index()
        self.id_map = []
        logger.info("✅ FAISS index cleared")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": getattr(self.index, "is_trained", True),
        }


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------

faiss_store = FAISSVectorStore()
