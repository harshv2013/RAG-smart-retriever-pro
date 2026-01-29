# FAISS Indexing for Production RAG Systems

## From exact search to billion-scale vector retrieval: A comprehensive guide to Facebook's vector search library

![FAISS Architecture](cover-image-placeholder)

---

## Introduction

When your RAG system has 1,000 documents, any vector database works fine. But at 100,000 documents? Your queries take seconds. At 1 million? Your system crashes.

**FAISS (Facebook AI Similarity Search)** is how companies like Facebook, Uber, and Spotify handle billions of vectors. It's the most battle-tested vector search library, providing 10-100x faster search than alternatives while running on your own infrastructure.

After building a production RAG system that handles millions of vectors with sub-50ms query times, here's everything I learned about FAISS—from basic indexing to advanced optimization techniques.

**What you'll learn:**
- Why FAISS beats managed vector databases for most use cases
- Four index types: when to use each
- Production deployment patterns
- Performance optimization techniques
- Scaling from thousands to millions of vectors
- Real-world failure modes and solutions

---

## Table of Contents

1. [Why FAISS?](#why-faiss)
2. [Core Concepts](#core-concepts)
3. [Index Types Deep Dive](#index-types)
4. [Choosing the Right Index](#choosing-index)
5. [Production Implementation](#production-implementation)
6. [Performance Optimization](#performance-optimization)
7. [Scaling Strategies](#scaling-strategies)
8. [Advanced Techniques](#advanced-techniques)
9. [Troubleshooting Guide](#troubleshooting)
10. [Benchmarks & Comparisons](#benchmarks)

---

## Why FAISS? {#why-faiss}

### The Vector Search Problem

You have 1 million document chunks, each represented as a 1536-dimensional vector (OpenAI embedding). A user queries: "What was Q4 revenue?"

**Naive approach:**
```python
def search(query_vector, all_vectors):
    similarities = []
    for vec in all_vectors:  # 1 million iterations!
        sim = cosine_similarity(query_vector, vec)
        similarities.append(sim)
    return top_k(similarities, k=5)

# Time: 1M × 1536 multiplications = ~2-3 seconds
```

**2-3 seconds per query is unacceptable** for production.

### FAISS Solution

```python
import faiss

# Build index (once)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(vectors)
index.add(vectors)

# Search (fast!)
distances, indices = index.search(query_vector, k=5)

# Time: ~10-50ms (50-300x faster!)
```

### FAISS vs Alternatives

| Feature | FAISS | Pinecone | Qdrant | Weaviate | Milvus |
|---------|-------|----------|--------|----------|--------|
| **Speed (1M vecs)** | 10-50ms | 50-100ms | 40-80ms | 60-120ms | 30-70ms |
| **Cost** | Infrastructure only | $70/month + usage | Infrastructure | Infrastructure | Infrastructure |
| **Scalability** | Billions | 100M+ | 100M+ | 100M+ | Billions |
| **GPU Support** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Local Dev** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Deployment** | DIY | Managed | DIY | DIY | DIY |

**When to choose FAISS:**
- Want full control over infrastructure
- Need GPU acceleration
- Handling >1M vectors efficiently
- Budget-conscious (no per-query costs)
- Data privacy requirements

**When NOT to choose FAISS:**
- Want fully managed solution
- Small team, limited ops expertise
- Need built-in CRUD operations
- Prefer hosted services

---

## Core Concepts {#core-concepts}

### Vector Similarity

FAISS primarily uses **L2 (Euclidean) distance**:

```python
# L2 distance between two vectors
def l2_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

# Smaller distance = more similar
# distance = 0 means identical vectors
```

**Converting to similarity score:**
```python
def distance_to_similarity(distance):
    """Convert L2 distance to 0-1 similarity score"""
    return 1 / (1 + distance)

# distance=0   → similarity=1.0 (identical)
# distance=1   → similarity=0.5
# distance=10  → similarity=0.09
```

**Why L2 instead of Cosine?**
- L2 is faster to compute
- OpenAI embeddings are normalized, so L2 ≈ Cosine
- Can convert: `cosine_sim = 1 - (l2_dist² / 2)`

### Index Components

Every FAISS index has:

**1. The Vector Store**
Where actual vectors live in memory/disk

**2. The Search Algorithm**
How to find nearest neighbors quickly

**3. Quantization (optional)**
Compress vectors to save memory

**4. Training Data (for some indexes)**
Sample vectors to learn data distribution

### Index Types Overview

```
Flat Index (Exact)
├─ Fast for <100K vectors
├─ Perfect accuracy
└─ No training needed

IVF Index (Approximate)
├─ Fast for 100K-10M vectors
├─ 95-99% accuracy
└─ Requires training

HNSW Index (Graph-based)
├─ Very fast for 1M+ vectors
├─ 95-99% accuracy
└─ No training needed

Product Quantization (Compressed)
├─ 4-8x memory savings
├─ Slight accuracy loss
└─ Essential for >10M vectors
```

---

## Index Types Deep Dive {#index-types}

### 1. Flat Index (IndexFlatL2)

**The simplest and most accurate index.**

```python
import faiss
import numpy as np

# Create index
dimension = 1536  # OpenAI embedding size
index = faiss.IndexFlatL2(dimension)

# Add vectors
vectors = np.random.rand(10000, dimension).astype('float32')
index.add(vectors)

# Search
query = np.random.rand(1, dimension).astype('float32')
distances, indices = index.search(query, k=5)

print(f"Top 5 indices: {indices[0]}")
print(f"Distances: {distances[0]}")
```

**How it works:**
```
1. Store all vectors in memory
2. On search: compute distance to ALL vectors
3. Return top-k smallest distances
```

**Characteristics:**
- **Accuracy:** 100% (exact search)
- **Speed:** O(n) - linear in dataset size
- **Memory:** O(n×d) - stores all vectors
- **Training:** Not required
- **Build time:** Instant (just add vectors)

**Performance:**
```
10K vectors:    5-10ms per query   ✓ Great
100K vectors:   50-100ms per query  ⚠️ Okay
1M vectors:     500ms+ per query    ❌ Too slow
```

**When to use:**
- Dataset < 100K vectors
- Need perfect accuracy
- Quick prototyping
- Sufficient memory available

**Production Implementation:**
```python
class FlatIndexManager:
    """Production Flat index with safety checks"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map = []  # Map FAISS idx to DB chunk ID
        
    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors with ID tracking"""
        # Validate
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch")
        
        # Ensure float32
        vectors = vectors.astype('float32')
        
        # Add to index
        self.index.add(vectors)
        self.id_map.extend(ids)
        
        logger.info(f"Added {len(vectors)} vectors (total: {self.index.ntotal})")
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """Search with ID mapping"""
        query = query.reshape(1, -1).astype('float32')
        
        distances, indices = self.index.search(query, k)
        
        # Map back to database IDs
        chunk_ids = [self.id_map[idx] for idx in indices[0]]
        similarities = [1 / (1 + d) for d in distances[0]]
        
        return chunk_ids, similarities
```

---

### 2. IVF Index (IndexIVFFlat)

**The production workhorse for medium-to-large datasets.**

**Concept:** Inverted File Index (like a search engine)

```
1. Cluster vectors into N groups (voronoi cells)
2. Store which vectors belong to each cluster
3. On search: 
   - Find nearest cluster centers
   - Search only vectors in those clusters
   - Much faster! (search M vectors instead of 1M)
```

**Visual:**
```
All Vectors (1M)
     │
     ├─ Cluster 1 (10K vectors) ─┐
     ├─ Cluster 2 (10K vectors) ─┤
     ├─ Cluster 3 (10K vectors) ─┤ nlist = 100 clusters
     ├─ ...                      │
     └─ Cluster 100 (10K vectors)┘
     
Query → Find 10 nearest clusters → Search only 100K vectors (10x speedup!)
```

**Implementation:**
```python
class IVFIndexManager:
    """Production IVF index"""
    
    def __init__(
        self,
        dimension: int = 1536,
        nlist: int = 100,  # Number of clusters
        nprobe: int = 10   # Clusters to search
    ):
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Create quantizer (for clustering)
        quantizer = faiss.IndexFlatL2(dimension)
        
        # Create IVF index
        self.index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            nlist
        )
        
        self.id_map = []
        
    def train(self, training_vectors: np.ndarray):
        """Train index on sample vectors"""
        if self.index.is_trained:
            logger.info("Index already trained")
            return
        
        # Need at least 39*nlist training vectors
        min_training = 39 * self.nlist
        if len(training_vectors) < min_training:
            logger.warning(
                f"Not enough training vectors: {len(training_vectors)} < {min_training}"
            )
        
        training_vectors = training_vectors.astype('float32')
        
        logger.info(f"Training IVF index with {len(training_vectors)} vectors...")
        self.index.train(training_vectors)
        logger.info("✓ Training complete")
    
    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors (after training)"""
        if not self.index.is_trained:
            raise ValueError("Index must be trained before adding vectors")
        
        vectors = vectors.astype('float32')
        self.index.add(vectors)
        self.id_map.extend(ids)
        
        logger.info(f"Added {len(vectors)} vectors (total: {self.index.ntotal})")
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """Search with nprobe setting"""
        query = query.reshape(1, -1).astype('float32')
        
        # Set nprobe (how many clusters to search)
        self.index.nprobe = self.nprobe
        
        distances, indices = self.index.search(query, k)
        
        chunk_ids = [self.id_map[idx] for idx in indices[0] 
                     if 0 <= idx < len(self.id_map)]
        similarities = [1 / (1 + d) for d in distances[0]]
        
        return chunk_ids, similarities
```

**Key Parameters:**

**nlist (number of clusters):**
```python
# Rule of thumb: nlist ≈ sqrt(N) to 4*sqrt(N)

100K vectors  → nlist = 316 to 1264
1M vectors    → nlist = 1000 to 4000
10M vectors   → nlist = 3162 to 12649

# Our production default: nlist = 4*sqrt(N)
nlist = int(4 * np.sqrt(num_vectors))
```

**nprobe (clusters to search):**
```python
# Trade-off: accuracy vs speed

nprobe = 1   → Very fast (5ms),  70-80% recall  ❌ Too inaccurate
nprobe = 5   → Fast (10ms),      85-90% recall  ⚠️ Acceptable
nprobe = 10  → Balanced (15ms),  95-97% recall  ✓ Good (default)
nprobe = 20  → Slower (30ms),    98-99% recall  ✓ High accuracy
nprobe = 50  → Slow (80ms),      99.5% recall   ⚠️ Overkill
```

**Performance:**
```
100K vectors, nlist=400, nprobe=10:   10-20ms    ✓ Excellent
1M vectors, nlist=1000, nprobe=10:    15-30ms    ✓ Excellent
10M vectors, nlist=4000, nprobe=10:   30-60ms    ✓ Good
100M vectors, nlist=10000, nprobe=10: 100-200ms  ⚠️ Consider HNSW
```

**When to use:**
- Dataset 100K - 10M vectors
- Can tolerate 95-99% accuracy
- Want good speed/accuracy balance
- Have training data available

---

### 3. HNSW Index (IndexHNSWFlat)

**Graph-based index for very fast search.**

**Concept:** Hierarchical Navigable Small World graphs

```
Build multi-layer graph of vectors:

Layer 2: [A]───────────[B]  (sparse, long edges)
         │              │
Layer 1: [A]──[C]──[D]──[B]  (medium density)
         │    │    │    │
Layer 0: [A]-[C]-[D]-[E]-[B]-[F]-[G]  (all vectors, dense)

Search: Start at top layer, navigate down
- Each layer is exponentially denser
- Top layer = express highway
- Bottom layer = local streets
```

**Implementation:**
```python
class HNSWIndexManager:
    """Production HNSW index"""
    
    def __init__(
        self,
        dimension: int = 1536,
        M: int = 32,  # Connections per layer
        efConstruction: int = 40,  # Build quality
        efSearch: int = 16  # Search quality
    ):
        self.dimension = dimension
        self.M = M
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(dimension, M)
        
        # Set build parameters
        self.index.hnsw.efConstruction = efConstruction
        
        # Set search parameters
        self.index.hnsw.efSearch = efSearch
        
        self.id_map = []
        
        logger.info(f"Created HNSW index (M={M}, efConstruction={efConstruction})")
    
    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors (no training needed!)"""
        vectors = vectors.astype('float32')
        
        self.index.add(vectors)
        self.id_map.extend(ids)
        
        logger.info(f"Added {len(vectors)} vectors (total: {self.index.ntotal})")
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """Search"""
        query = query.reshape(1, -1).astype('float32')
        
        distances, indices = self.index.search(query, k)
        
        chunk_ids = [self.id_map[idx] for idx in indices[0]]
        similarities = [1 / (1 + d) for d in distances[0]]
        
        return chunk_ids, similarities
```

**Key Parameters:**

**M (connections per layer):**
```python
M = 16  → Faster build, uses less memory, slightly lower accuracy
M = 32  → Balanced (default)
M = 64  → Slower build, uses more memory, higher accuracy

# Our production: M = 32 for most use cases
```

**efConstruction (build quality):**
```python
efConstruction = 40   → Fast build, okay quality
efConstruction = 200  → Medium build, good quality (default)
efConstruction = 500  → Slow build, excellent quality

# Trade-off: build time vs index quality
# Higher = better recall, but takes longer to build
```

**efSearch (search quality):**
```python
efSearch = 16   → Fast search (5ms), 90% recall
efSearch = 32   → Balanced (8ms), 95% recall
efSearch = 64   → Slower (15ms), 98% recall
efSearch = 128  → Slow (30ms), 99% recall

# Can be changed at query time!
# Our production: efSearch = 32
```

**Performance:**
```
1M vectors, M=32, efSearch=32:     5-10ms   ✓ Excellent
10M vectors, M=32, efSearch=32:    8-15ms   ✓ Excellent  
100M vectors, M=32, efSearch=32:   15-30ms  ✓ Excellent
1B vectors, M=32, efSearch=32:     50-100ms ✓ Good
```

**Pros:**
- ✅ Very fast search
- ✅ No training required
- ✅ Excellent scaling
- ✅ High accuracy

**Cons:**
- ❌ Higher memory usage (stores graph)
- ❌ Slower build time
- ❌ Can't remove vectors easily

**When to use:**
- Dataset 1M+ vectors
- Need sub-10ms search latency
- Don't want to deal with training
- Have sufficient memory (2-3x vector size)

---

### 4. Product Quantization (IndexIVFPQ)

**Compressed index for massive datasets.**

**Concept:** Compress vectors to save memory

```
Original vector (1536 dims):
[0.123, 0.456, 0.789, ...] → 6,144 bytes (1536 × 4 bytes)

Product Quantization:
Split into 8 subvectors of 192 dims each
Each subvector → 1 byte codebook index
Result: 8 bytes (768x compression!)

Memory: 6KB → 8 bytes (save 99.87% memory!)
```

**Implementation:**
```python
class PQIndexManager:
    """Production PQ index for massive datasets"""
    
    def __init__(
        self,
        dimension: int = 1536,
        nlist: int = 1000,
        M: int = 8,  # Number of subquantizers
        nbits: int = 8  # Bits per subquantizer
    ):
        self.dimension = dimension
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        
        # Create quantizer
        quantizer = faiss.IndexFlatL2(dimension)
        
        # Create IVF+PQ index
        self.index = faiss.IndexIVFPQ(
            quantizer,
            dimension,
            nlist,
            M,
            nbits
        )
        
        self.id_map = []
        
        logger.info(
            f"Created IVF+PQ index "
            f"(nlist={nlist}, M={M}, nbits={nbits})"
        )
    
    def train(self, training_vectors: np.ndarray):
        """Train (required for PQ)"""
        if self.index.is_trained:
            return
        
        training_vectors = training_vectors.astype('float32')
        
        logger.info(f"Training PQ index with {len(training_vectors)} vectors...")
        self.index.train(training_vectors)
        logger.info("✓ Training complete")
    
    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors"""
        if not self.index.is_trained:
            raise ValueError("Must train before adding")
        
        vectors = vectors.astype('float32')
        self.index.add(vectors)
        self.id_map.extend(ids)
        
        logger.info(f"Added {len(vectors)} vectors")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        nprobe: int = 10
    ) -> Tuple[List[int], List[float]]:
        """Search"""
        query = query.reshape(1, -1).astype('float32')
        
        self.index.nprobe = nprobe
        distances, indices = self.index.search(query, k)
        
        chunk_ids = [self.id_map[idx] for idx in indices[0]]
        similarities = [1 / (1 + d) for d in distances[0]]
        
        return chunk_ids, similarities
```

**Key Parameters:**

**M (number of subquantizers):**
```python
# dimension must be divisible by M

dimension = 1536
M = 8   → 192 dims per subquantizer → 8 bytes per vector
M = 16  → 96 dims per subquantizer → 16 bytes per vector
M = 32  → 48 dims per subquantizer → 32 bytes per vector

# Higher M = better accuracy, more memory
# Our production: M = 8 or 16
```

**Memory Savings:**
```
Original (Flat):       1M vectors × 6KB = 6GB
IVF:                   1M vectors × 6KB = 6GB (same)
IVF+PQ (M=8):         1M vectors × 8B = 8MB (750x reduction!)
IVF+PQ (M=16):        1M vectors × 16B = 16MB (375x reduction!)
```

**Accuracy Trade-off:**
```
Flat:       100% recall
IVF:        95-99% recall
IVF+PQ M=8: 90-95% recall  ⚠️ Acceptable for some use cases
IVF+PQ M=16: 93-97% recall ✓ Good balance
```

**When to use:**
- Dataset 10M+ vectors
- Memory is constrained
- Can tolerate 90-95% accuracy
- Want to fit in RAM/GPU memory

---

## Choosing the Right Index {#choosing-index}

### Decision Flow

```
How many vectors?

< 100K
└─ Use IndexFlatL2
   ✓ Simple, fast enough, perfect accuracy

100K - 1M
└─ Use IndexIVFFlat
   ✓ Fast, good accuracy, proven workhorse

1M - 10M
├─ Prioritize speed? → Use IndexHNSWFlat
│  ✓ Fastest search, no training
└─ Prioritize memory? → Use IndexIVFPQ
   ✓ 10-100x memory savings

10M - 100M
├─ Have GPU? → Use GPU IndexIVFFlat
│  ✓ 10x faster than CPU
└─ CPU only? → Use IndexIVFPQ
   ✓ Fit in RAM

100M+
└─ Use IndexIVFPQ + sharding
   ✓ Only way to handle this scale
```

### Production Recommendations

**Startup / MVP (< 100K docs):**
```python
index = faiss.IndexFlatL2(1536)
# Simple, no tuning needed
```

**Growing (100K - 1M docs):**
```python
nlist = int(4 * np.sqrt(num_vectors))
quantizer = faiss.IndexFlatL2(1536)
index = faiss.IndexIVFFlat(quantizer, 1536, nlist)
index.train(training_vectors)
# Balanced speed and accuracy
```

**Scale (1M - 10M docs):**
```python
# Option 1: HNSW (if memory allows)
index = faiss.IndexHNSWFlat(1536, 32)
index.hnsw.efConstruction = 200
# Fastest search

# Option 2: IVF+PQ (if memory constrained)
index = faiss.IndexIVFPQ(quantizer, 1536, nlist, 16, 8)
index.train(training_vectors)
# 375x memory reduction
```

**Enterprise (10M+ docs):**
```python
# IVF+PQ on GPU
res = faiss.StandardGpuResources()
index = faiss.IndexIVFPQ(quantizer, 1536, nlist, 16, 8)
index.train(training_vectors)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
# 10x faster than CPU
```

---

## Production Implementation {#production-implementation}

### Complete Production Class

```python
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger

class ProductionFAISSStore:
    """Production-ready FAISS vector store"""
    
    def __init__(
        self,
        dimension: int = 1536,
        index_type: str = "Flat",  # Flat, IVF, HNSW, PQ
        index_path: str = "data/faiss_index",
        **kwargs
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = Path(index_path)
        
        self.index: Optional[faiss.Index] = None
        self.id_map: List[int] = []  # Map FAISS idx to DB chunk ID
        
        # Create index
        self._create_index(**kwargs)
        
        logger.info(
            f"✓ FAISS store initialized "
            f"(type={index_type}, dim={dimension})"
        )
    
    def _create_index(self, **kwargs):
        """Create appropriate index type"""
        
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
            
        elif self.index_type == "IVF":
            nlist = kwargs.get('nlist', 100)
            self.nprobe = kwargs.get('nprobe', 10)
            
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                nlist
            )
            logger.info(f"Created IVF index (nlist={nlist})")
            
        elif self.index_type == "HNSW":
            M = kwargs.get('M', 32)
            efConstruction = kwargs.get('efConstruction', 200)
            self.efSearch = kwargs.get('efSearch', 32)
            
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            self.index.hnsw.efConstruction = efConstruction
            self.index.hnsw.efSearch = self.efSearch
            
            logger.info(f"Created HNSW index (M={M})")
            
        elif self.index_type == "PQ":
            nlist = kwargs.get('nlist', 1000)
            M = kwargs.get('M', 8)
            nbits = kwargs.get('nbits', 8)
            self.nprobe = kwargs.get('nprobe', 10)
            
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                nlist,
                M,
                nbits
            )
            logger.info(f"Created IVF+PQ index (M={M})")
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def train(self, training_vectors: np.ndarray):
        """Train index (if needed)"""
        
        # Check if training is needed
        if not hasattr(self.index, 'is_trained'):
            logger.info("Index doesn't require training")
            return
        
        if self.index.is_trained:
            logger.info("Index already trained")
            return
        
        # Validate training size
        if self.index_type == "IVF":
            min_training = 39 * self.index.nlist
        elif self.index_type == "PQ":
            min_training = 256 * self.index.pq.M
        else:
            min_training = 1000
        
        if len(training_vectors) < min_training:
            logger.warning(
                f"Training vectors ({len(training_vectors)}) "
                f"less than recommended ({min_training})"
            )
        
        # Train
        training_vectors = training_vectors.astype('float32')
        logger.info(f"Training index with {len(training_vectors)} vectors...")
        
        self.index.train(training_vectors)
        
        logger.info("✓ Training complete")
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[int]
    ):
        """Add vectors to index"""
        
        # Validate
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: "
                f"expected {self.dimension}, got {vectors.shape[1]}"
            )
        
        if len(vectors) != len(ids):
            raise ValueError("Vector count must match ID count")
        
        # Check if training needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            raise ValueError("Index must be trained before adding vectors")
        
        # Convert to float32
        vectors = vectors.astype('float32')
        
        # Add to index
        self.index.add(vectors)
        self.id_map.extend(ids)
        
        logger.info(
            f"✓ Added {len(vectors)} vectors "
            f"(total: {self.index.ntotal})"
        )
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        **kwargs
    ) -> Tuple[List[int], List[float]]:
        """Search for nearest neighbors"""
        
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return [], []
        
        # Reshape and convert
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        # Set search parameters
        if self.index_type == "IVF" or self.index_type == "PQ":
            nprobe = kwargs.get('nprobe', self.nprobe)
            self.index.nprobe = nprobe
        
        elif self.index_type == "HNSW":
            efSearch = kwargs.get('efSearch', self.efSearch)
            self.index.hnsw.efSearch = efSearch
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Map indices to chunk IDs
        chunk_ids = [
            self.id_map[idx]
            for idx in indices[0]
            if 0 <= idx < len(self.id_map)
        ]
        
        # Convert L2 distance to similarity
        similarities = [1 / (1 + d) for d in distances[0]]
        
        logger.debug(f"Search returned {len(chunk_ids)} results")
        
        return chunk_ids, similarities
    
    def save(self, path: Optional[str] = None):
        """Save index and ID map to disk"""
        
        path = Path(path or self.index_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save ID map
        id_map_file = path / "id_map.pkl"
        with open(id_map_file, 'wb') as f:
            pickle.dump(self.id_map, f)
        
        # Save metadata
        metadata_file = path / "metadata.pkl"
        metadata = {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal,
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✓ Index saved to {path}")
    
    def load(self, path: Optional[str] = None) -> bool:
        """Load index and ID map from disk"""
        
        path = Path(path or self.index_path)
        
        index_file = path / "index.faiss"
        id_map_file = path / "id_map.pkl"
        metadata_file = path / "metadata.pkl"
        
        # Check if files exist
        if not all(f.exists() for f in [index_file, id_map_file]):
            logger.warning(f"Index files not found at {path}")
            return False
        
        try:
            # Load index
            self.index = faiss.read_index(str(index_file))
            
            # Load ID map
            with open(id_map_file, 'rb') as f:
                self.id_map = pickle.load(f)
            
            # Load metadata (if exists)
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info(f"Loaded metadata: {metadata}")
            
            logger.info(
                f"✓ Index loaded from {path} "
                f"({self.index.ntotal} vectors)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def clear(self):
        """Clear index and ID map"""
        self._create_index()
        self.id_map = []
        logger.info("✓ Index cleared")
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": getattr(self.index, 'is_trained', True),
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if self.index_type == "Flat":
            # Full vectors: 4 bytes per dim
            return (self.index.ntotal * self.dimension * 4) / (1024 * 1024)
        
        elif self.index_type == "IVF":
            # Same as Flat + overhead
            return (self.index.ntotal * self.dimension * 4 * 1.1) / (1024 * 1024)
        
        elif self.index_type == "HNSW":
            # Vectors + graph (roughly 2-3x)
            return (self.index.ntotal * self.dimension * 4 * 2.5) / (1024 * 1024)
        
        elif self.index_type == "PQ":
            # Compressed: M bytes per vector
            M = 8  # Default
            return (self.index.ntotal * M) / (1024 * 1024)
        
        return 0.0

# Usage
store = ProductionFAISSStore(
    dimension=1536,
    index_type="IVF",
    nlist=100,
    nprobe=10
)

# Train (if needed)
store.train(training_vectors)

# Add vectors
store.add_vectors(vectors, chunk_ids)

# Search
results_ids, similarities = store.search(query_vector, k=5)

# Save
store.save()

# Load
store.load()
```

---

## Performance Optimization {#performance-optimization}

### 1. Batch Operations

**Bad:**
```python
# Sequential adds (slow!)
for vector, id in zip(vectors, ids):
    index.add(np.array([vector]))  # 1000 add operations

# Time: ~10 seconds for 1000 vectors
```

**Good:**
```python
# Batch add (fast!)
index.add(vectors)  # Single add operation

# Time: ~0.1 seconds for 1000 vectors (100x faster!)
```

### 2. Optimal nlist for IVF

```python
def calculate_optimal_nlist(num_vectors: int) -> int:
    """Calculate optimal nlist based on dataset size"""
    
    # Rule: 4*sqrt(N) to sqrt(N)
    base_nlist = int(4 * np.sqrt(num_vectors))
    
    # Constraints
    min_nlist = 100  # Minimum clusters
    max_nlist = 65536  # FAISS limit
    
    # Round to nearest power of 2 (optional, for cleaner numbers)
    nlist = 2 ** int(np.log2(base_nlist))
    
    return np.clip(nlist, min_nlist, max_nlist)

# Examples:
# 100K vectors → nlist = 1264 → 1024
# 1M vectors → nlist = 4000 → 4096
# 10M vectors → nlist = 12649 → 8192
```

### 3. Training Data Selection

```python
def select_training_data(
    all_vectors: np.ndarray,
    nlist: int
) -> np.ndarray:
    """Select good training data"""
    
    # Minimum: 39*nlist
    # Recommended: 256*nlist
    # Maximum: all vectors (but expensive)
    
    min_training = 39 * nlist
    recommended_training = 256 * nlist
    
    num_training = min(
        len(all_vectors),
        max(min_training, recommended_training)
    )
    
    # Random sampling
    indices = np.random.choice(
        len(all_vectors),
        size=num_training,
        replace=False
    )
    
    training_vectors = all_vectors[indices]
    
    logger.info(
        f"Selected {num_training} training vectors "
        f"(min: {min_training}, recommended: {recommended_training})"
    )
    
    return training_vectors
```

### 4. Query Batching

```python
def batch_search(
    index: faiss.Index,
    queries: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Search multiple queries at once"""
    
    # FAISS can handle multiple queries
    queries = queries.astype('float32')
    
    # Search all at once
    distances, indices = index.search(queries, k)
    
    return distances, indices

# Example
queries = np.random.rand(100, 1536).astype('float32')

# Bad: Loop (slow)
for query in queries:
    distances, indices = index.search(query.reshape(1, -1), k=5)

# Good: Batch (fast)
distances, indices = index.search(queries, k=5)
# Returns: (100, 5) arrays
```

---

## Troubleshooting Guide {#troubleshooting}

### Issue 1: Training Fails

**Symptom:**
```
RuntimeError: Error in void faiss::IndexIVFFlat::train(...)
Training set too small
```

**Solution:**
```python
# Need at least 39*nlist training vectors
min_training = 39 * nlist

if len(training_vectors) < min_training:
    # Option 1: Reduce nlist
    nlist = len(training_vectors) // 39
    
    # Option 2: Generate more training data
    # (duplicate if necessary, but not ideal)
    while len(training_vectors) < min_training:
        training_vectors = np.vstack([
            training_vectors,
            training_vectors
        ])
```

### Issue 2: Low Search Accuracy

**Symptom:**
IVF index returns wrong results 30-40% of the time.

**Solution:**
```python
# Increase nprobe
index.nprobe = 20  # from 10

# Or increase nlist
nlist = int(8 * np.sqrt(num_vectors))  # from 4*sqrt

# Or use HNSW instead
index = faiss.IndexHNSWFlat(dimension, M=32)
```

### Issue 3: Out of Memory

**Symptom:**
```
MemoryError: Unable to allocate ...
```

**Solution:**
```python
# Option 1: Use Product Quantization
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, M=8, nbits=8)
# Saves 750x memory!

# Option 2: Shard the index
shard1 = create_index(vectors[:1000000])
shard2 = create_index(vectors[1000000:2000000])
# Search both, merge results

# Option 3: Move to GPU (if available)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

### Issue 4: Slow Indexing

**Symptom:**
Adding vectors takes hours.

**Solution:**
```python
# Option 1: Batch adds
# Instead of:
for v, id in zip(vectors, ids):
    index.add(np.array([v]))

# Do:
index.add(vectors)  # Single call

# Option 2: Use Flat for small datasets
# If < 100K vectors, Flat is actually faster to build

# Option 3: Parallelize (for multiple indices)
from concurrent.futures import ThreadPoolExecutor

def build_index(vectors, ids):
    index = create_index()
    index.add(vectors)
    return index

with ThreadPoolExecutor() as executor:
    indices = executor.map(build_index, vector_shards, id_shards)
```

---

## Conclusion

FAISS is the backbone of production-scale vector search. Key takeaways:

1. **Start Simple:** Use Flat until you need IVF (~100K vectors)
2. **IVF is the Workhorse:** 100K-10M vectors, tune nlist and nprobe
3. **HNSW for Speed:** 1M+ vectors, no training needed
4. **PQ for Scale:** 10M+ vectors, massive memory savings
5. **Always Benchmark:** Test with your actual data and queries

**Production Checklist:**
- ✅ Chose appropriate index type for scale
- ✅ Tuned nlist/nprobe (IVF) or M/efSearch (HNSW)
- ✅ Collected sufficient training data
- ✅ Implemented save/load persistence
- ✅ Added comprehensive logging
- ✅ Set up performance monitoring
- ✅ Tested at expected scale
- ✅ Documented index configuration

**What's Next:**
- Read the companion articles on RAG Systems and Chunking Strategies
- Experiment with GPU FAISS for 10x speedup
- Explore hybrid search (vector + keyword)
- Consider distributed FAISS for billion-scale

---

## Resources

- **Main Article:** Building Production-Ready RAG Systems
- **Companion:** Advanced Chunking Strategies
- **FAISS Documentation:** https://github.com/facebookresearch/faiss/wiki
- **FAISS Paper:** https://arxiv.org/abs/1702.08734
- **Tutorial:** https://www.pinecone.io/learn/faiss-tutorial/

---

**Tags:** #FAISS #VectorSearch #RAG #MachineLearning #Embeddings #ANN #Production #Performance

*Found this helpful? Share it with others building vector search systems! 👏*
