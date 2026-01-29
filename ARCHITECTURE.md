# 🏗️ SmartRetriever Pro - System Architecture

**Comprehensive System Design Documentation**

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Database Schema](#database-schema)
5. [FAISS Integration](#faiss-integration)
6. [Caching Strategy](#caching-strategy)
7. [Scalability Considerations](#scalability-considerations)
8. [Performance Optimization](#performance-optimization)
9. [Security](#security)
10. [Monitoring & Observability](#monitoring--observability)

---

## System Overview

SmartRetriever Pro is a production-ready RAG system designed for scalability, performance, and reliability.

### Design Principles

1. **Modularity** - Each component is independent and replaceable
2. **Scalability** - Horizontal scaling supported at each layer
3. **Performance** - Multi-layer caching and batch processing
4. **Reliability** - Error handling, retries, and graceful degradation
5. **Observability** - Comprehensive logging and metrics

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                         │
│                   (Python API / Scripts)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────────┐          ┌──────────────────┐
│  RAG Orchestrator│          │  File Manager     │
│  (rag_system.py) │          │  (file_manager.py)│
└─────────┬────────┘          └──────────────────┘
          │
    ┌─────┴─────┬─────────┬──────────┐
    │           │         │          │
    ▼           ▼         ▼          ▼
┌─────────┐ ┌──────┐ ┌────────┐ ┌─────────┐
│Chunker  │ │Embed │ │Retrieve│ │Generator│
│         │ │der   │ │r       │ │         │
└─────────┘ └──────┘ └────────┘ └─────────┘
    │           │         │          │
    └─────┬─────┴─────┬───┴──────────┘
          │           │
    ┌─────▼─────┬─────▼──────┬────────────┐
    │           │            │            │
    ▼           ▼            ▼            ▼
┌──────────┐ ┌──────┐ ┌───────────┐ ┌───────┐
│PostgreSQL│ │FAISS │ │   Redis   │ │ Files │
│          │ │Vector│ │   Cache   │ │Storage│
│          │ │Store │ │           │ │       │
└──────────┘ └──────┘ └───────────┘ └───────┘
```

---

## Component Architecture

### 1. RAG Orchestrator (`rag_system.py`)

**Responsibilities:**
- Coordinate all components
- Manage document pipeline
- Handle query processing
- Provide unified API

**Key Methods:**
- `add_document()` - Document ingestion pipeline
- `query()` - Query processing pipeline
- `get_stats()` - System statistics
- `rebuild_faiss_index()` - Index management

**Design Pattern:** Facade Pattern
- Provides simple interface to complex subsystem
- Manages component dependencies
- Handles error propagation

### 2. Document Chunker (`core/chunker.py`)

**Responsibilities:**
- Split documents into chunks
- Preserve semantic boundaries
- Count tokens accurately
- Maintain metadata

**Chunking Strategies:**

**A. Semantic Chunking (Default)**
```
Document → Paragraphs → Combine small → Split large → Chunks
```
- Respects paragraph boundaries
- Combines small paragraphs
- Splits large paragraphs by sentences
- Best for: Natural text, articles, documentation

**B. Fixed Chunking**
```
Document → Fixed-size windows → Overlap → Chunks
```
- Fixed character windows
- Configurable overlap
- Simple and fast
- Best for: Code, structured data

**C. Recursive Chunking**
```
Document → Try separator[0] → Too large? → Try separator[1] → ...
```
- Hierarchical splitting
- Multiple fallback separators
- Preserves structure
- Best for: Complex documents

**Token Counting:**
- Uses tiktoken (GPT-4 tokenizer)
- Accurate token estimates
- Prevents context overflow

### 3. Embedding Service (`core/embedder.py`)

**Responsibilities:**
- Generate embeddings via Azure OpenAI
- Batch processing for efficiency
- Cache management
- Rate limiting

**Features:**
- **Batch Processing**: Up to 10 texts per API call
- **Caching**: Redis-based embedding cache (24h TTL)
- **Error Handling**: Retry logic with exponential backoff
- **Performance**: ~100 embeddings/minute

**Flow:**
```
Text → Check cache → Hit? Return cached
                  → Miss? Generate → Cache → Return
```

### 4. Retrieval Service (`core/retriever.py`)

**Responsibilities:**
- Orchestrate retrieval pipeline
- FAISS similarity search
- PostgreSQL metadata fetch
- Result ranking

**Pipeline:**
```
Query → Embed → FAISS Search → Fetch Metadata → Filter → Rank → Return
```

**Optimization:**
- Retrieves 2x top_k from FAISS
- Filters by similarity threshold
- Re-ranks by multiple factors
- Caches results (1h TTL)

### 5. Generation Service (`core/generator.py`)

**Responsibilities:**
- Build prompts with context
- Call Azure OpenAI GPT
- Cache responses
- Track token usage

**Prompt Structure:**
```
System: "You are a helpful assistant..."

User: """
Context Documents:
[Document 1: source.txt]
Content here...

Question: What is...?
"""
```

**Features:**
- Context-aware generation
- Source citation
- Response caching (30min TTL)
- Token tracking

### 6. File Manager (`storage/file_manager.py`)

**Responsibilities:**
- File upload and storage
- Deduplication (hash-based)
- File type validation
- Storage organization

**Storage Structure:**
```
data/
├── storage/
│   ├── ab/    # First 2 chars of hash
│   │   └── document.txt
│   └── cd/
├── uploads/   # Temporary uploads
└── processed/ # Processed files
```

**Deduplication:**
- SHA256 hash of file content
- Check before saving
- Return existing if duplicate
- Saves storage and processing

---

## Data Flow

### Document Ingestion Flow

```
1. File Upload
   └─→ file_manager.save_file()
       └─→ Calculate SHA256 hash
       └─→ Check if exists (dedup)
       └─→ Save to storage/

2. Text Extraction
   └─→ file_manager.read_text_file()
       └─→ Read content
       └─→ Handle encoding errors

3. Chunking
   └─→ chunker.chunk_document()
       └─→ Apply strategy (semantic/fixed/recursive)
       └─→ Count tokens
       └─→ Add metadata

4. Embedding Generation
   └─→ embedder.embed_batch()
       └─→ Check cache for each chunk
       └─→ Batch API calls
       └─→ Cache results

5. Database Storage
   └─→ db_manager.add_document()
   └─→ db_manager.add_chunk() (for each chunk)
   └─→ db_manager.add_embedding() (metadata)

6. FAISS Indexing
   └─→ faiss_store.add_vectors()
       └─→ Add to index
       └─→ Update ID map
       └─→ Save index

7. Completion
   └─→ Return result with stats
```

### Query Processing Flow

```
1. Cache Check
   └─→ redis_cache.get_retrieval(query)
       └─→ Hit? Return cached results
       └─→ Miss? Continue...

2. Query Embedding
   └─→ embedder.embed_query(query)
       └─→ Check cache
       └─→ Generate if needed
       └─→ Cache for 24h

3. FAISS Search
   └─→ faiss_store.search(embedding, k=top_k*2)
       └─→ Similarity search
       └─→ Return chunk IDs + scores

4. Metadata Fetch
   └─→ db_manager.get_chunks_by_faiss_ids()
       └─→ Batch fetch from PostgreSQL
       └─→ Include document info

5. Filtering & Ranking
   └─→ Filter by similarity threshold
   └─→ Sort by score
   └─→ Limit to top_k

6. Cache Results
   └─→ redis_cache.set_retrieval()
       └─→ Cache for 1h

7. Response Generation
   └─→ generator.generate_response()
       └─→ Build context
       └─→ Call GPT
       └─→ Cache response (30min)

8. Query Logging
   └─→ db_manager.log_query()
       └─→ Store for analytics

9. Return
   └─→ Answer + sources + metadata
```

---

## Database Schema

### Documents Table

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    filepath TEXT NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    content_type VARCHAR(50),
    file_size INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_documents_hash ON documents(file_hash);
```

**Purpose:** Store document metadata

**Key Fields:**
- `file_hash`: SHA256 for deduplication
- `metadata`: JSONB for flexible schema

### Chunks Table

```sql
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    chunk_hash VARCHAR(64),
    token_count INTEGER,
    faiss_id INTEGER UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB,
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_faiss ON chunks(faiss_id);
```

**Purpose:** Store chunk content and link to FAISS

**Key Fields:**
- `faiss_id`: Maps to FAISS vector index
- `chunk_index`: Position in original document
- `token_count`: For context window management

### Embeddings Table

```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    faiss_index_id INTEGER UNIQUE NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Purpose:** Track embedding metadata (vectors stored in FAISS)

### Query Logs Table

```sql
CREATE TABLE query_logs (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    retrieved_chunks INTEGER[],
    response_time_ms INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_query_logs_created ON query_logs(created_at);
```

**Purpose:** Analytics and monitoring

---

## FAISS Integration

### Index Types

**1. Flat Index (Default)**
```python
index = faiss.IndexFlatL2(dimension)
```
- **Pros**: Exact search, simple, no training needed
- **Cons**: O(n) search, slow for large datasets
- **Best for**: < 100k vectors

**2. IVF Index (Production)**
```python
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
```
- **Pros**: Fast approximate search, scalable
- **Cons**: Requires training, slight accuracy loss
- **Best for**: 100k - 10M vectors
- **Configuration**: 
  - `nlist`: Number of clusters (√n to 4√n)
  - `nprobe`: Clusters to search (10-20)

**3. HNSW Index (Large Scale)**
```python
index = faiss.IndexHNSWFlat(dimension, M)
```
- **Pros**: Very fast, high accuracy
- **Cons**: High memory usage
- **Best for**: 1M+ vectors
- **Configuration**:
  - `M`: Connections per layer (16-64)

### Index Management

**Persistence:**
```python
# Save
faiss.write_index(index, "index.faiss")
pickle.dump(id_map, open("id_map.pkl", "wb"))

# Load
index = faiss.read_index("index.faiss")
id_map = pickle.load(open("id_map.pkl", "rb"))
```

**Training (IVF only):**
```python
# Collect training vectors
training_vectors = embeddings[:min(len(embeddings), 10000)]

# Train index
index.train(training_vectors.astype('float32'))

# Now can add vectors
index.add(all_embeddings)
```

### Search Process

```python
# 1. Prepare query
query_vector = embedding.reshape(1, -1).astype('float32')

# 2. Search FAISS
distances, indices = index.search(query_vector, k)

# 3. Convert to chunk IDs
chunk_ids = [id_map[idx] for idx in indices[0]]

# 4. Convert L2 distance to similarity
similarities = [1 / (1 + d) for d in distances[0]]
```

---

## Caching Strategy

### Three-Layer Cache

**Layer 1: Embedding Cache (24h TTL)**
```
Purpose: Cache text embeddings
Key: SHA256(text)[:16]
Value: Serialized numpy array
TTL: 86400 seconds (24 hours)
Impact: 15-20x speedup on repeated texts
```

**Layer 2: Retrieval Cache (1h TTL)**
```
Purpose: Cache retrieval results
Key: SHA256(query)[:16]
Value: JSON of retrieved chunks
TTL: 3600 seconds (1 hour)
Impact: 10-15x speedup on popular queries
```

**Layer 3: Response Cache (30min TTL)**
```
Purpose: Cache generated responses
Key: SHA256(query + context_hash)[:16]
Value: Text response
TTL: 1800 seconds (30 minutes)
Impact: 20-30x speedup on exact repeats
```

### Cache Invalidation

**When to invalidate:**
- Document added/updated: Clear retrieval cache
- Index rebuilt: Clear all caches
- Model changed: Clear embedding cache

**Strategy:**
```python
# Pattern-based invalidation
redis.delete(redis.keys("emb:*"))  # All embeddings
redis.delete(redis.keys("ret:*"))  # All retrievals
```

---

## Scalability Considerations

### Horizontal Scaling

**Application Layer:**
- Stateless RAG service
- Load balancer (Nginx/HAProxy)
- Multiple application instances
- Shared PostgreSQL + Redis + FAISS

**Database Layer:**
- PostgreSQL: Read replicas for queries
- Redis: Redis Cluster for cache distribution
- FAISS: Sharded indexes by category/date

### Vertical Scaling

**For 1M+ vectors:**
- Use IVF or HNSW index
- GPU FAISS for 10x speedup
- Increase PostgreSQL connection pool
- More Redis memory

### Storage Optimization

**FAISS:**
- Use Product Quantization for 4-8x compression
- Trade: Slight accuracy loss for huge memory savings

**PostgreSQL:**
- Partition chunks table by date
- Archive old documents
- Regular VACUUM

**Redis:**
- Set appropriate TTLs
- Use eviction policy: allkeys-lru
- Monitor memory usage

---

## Performance Optimization

### Benchmarks

**Query Performance:**
```
Cold query (no cache):
- Embedding: 100-200ms
- FAISS search: 10-50ms
- PostgreSQL fetch: 20-50ms
- GPT generation: 1000-1500ms
- Total: ~1500-2000ms

Warm query (full cache):
- Redis fetch: 5-10ms
- Total: ~50-100ms
- Speedup: 15-20x
```

**Batch Processing:**
```
100 documents:
- Serial: ~10 minutes
- Batch (10 at a time): ~2-3 minutes
- Speedup: 3-4x
```

### Optimization Techniques

**1. Batch Embedding**
```python
# Instead of:
for text in texts:
    embedding = embed(text)  # 100 API calls

# Do:
embeddings = embed_batch(texts, batch_size=10)  # 10 API calls
```

**2. Connection Pooling**
```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,           # Keep 10 connections
    max_overflow=20,        # Allow 20 more if needed
    pool_pre_ping=True      # Verify connections
)
```

**3. Async I/O** (Future Enhancement)
```python
async def process_documents(documents):
    tasks = [process_doc(doc) for doc in documents]
    results = await asyncio.gather(*tasks)
```

---

## Security

### Data Protection

**Secrets Management:**
- Environment variables for credentials
- Never commit `.env` to git
- Use Azure Key Vault in production

**Database Security:**
- Encrypted connections (SSL)
- Strong passwords
- Limited user permissions
- Regular backups

**API Security:**
- Rate limiting per user
- API key rotation
- HTTPS only in production

### Access Control

**Document-Level:**
```python
# Add user_id to metadata
metadata = {
    "source": "file.txt",
    "user_id": user.id,
    "access_level": "private"
}

# Filter in retrieval
results = [r for r in results 
           if r.metadata['user_id'] == current_user.id]
```

**Query Logging:**
- Log all queries for audit
- Track user actions
- Detect anomalies

---

## Monitoring & Observability

### Logging

**Log Levels:**
```python
logger.debug()   # Detailed debugging
logger.info()    # General operations
logger.warning() # Potential issues
logger.error()   # Errors occurred
```

**Structured Logging:**
```python
logger.info(
    "Query processed",
    extra={
        "query": query,
        "chunks_retrieved": len(chunks),
        "response_time_ms": elapsed,
        "cache_hit": cache_hit
    }
)
```

### Metrics

**Key Metrics:**
1. **Query Performance**
   - Average response time
   - P50, P95, P99 latencies
   - Cache hit rate

2. **System Health**
   - PostgreSQL connections
   - Redis memory usage
   - FAISS index size

3. **Business Metrics**
   - Queries per day
   - Documents processed
   - User satisfaction (thumbs up/down)

### Alerts

**Critical Alerts:**
- Database connection failure
- Redis unavailable
- High error rate (> 5%)
- Slow queries (> 5s)

**Warning Alerts:**
- Low cache hit rate (< 50%)
- High memory usage (> 80%)
- Disk space low (< 20%)

---

## Future Enhancements

1. **Hybrid Search** - Combine semantic + keyword search
2. **Re-ranking** - Cross-encoder for better ranking
3. **Multi-tenancy** - Isolated data per user/org
4. **Real-time Indexing** - Streaming document ingestion
5. **A/B Testing** - Test different retrieval strategies
6. **Auto-scaling** - Dynamic resource allocation
7. **Multi-modal** - Support images, audio, video

---

## Conclusion

SmartRetriever Pro is designed as a production-ready RAG system with:
- **Scalability**: From 100 to 10M+ documents
- **Performance**: Sub-100ms cached queries
- **Reliability**: Error handling and graceful degradation
- **Observability**: Comprehensive logging and metrics

The modular architecture allows easy extension and customization for specific use cases.

---

**For questions or contributions, see README.md**
