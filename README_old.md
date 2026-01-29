# 🧠 SmartRetriever Pro

**Production-Ready RAG (Retrieval-Augmented Generation) System with Azure OpenAI**

A comprehensive, scalable RAG system featuring FAISS vector search, PostgreSQL storage, Redis caching, and smart document processing.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://www.postgresql.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green.svg)](https://github.com/facebookresearch/faiss)
[![Redis](https://img.shields.io/badge/Redis-Cache-red.svg)](https://redis.io/)

---

## 🌟 Features

### Core Capabilities
- ✅ **Vector Search** - FAISS-powered similarity search (Flat, IVF, HNSW indexes)
- ✅ **Persistent Storage** - PostgreSQL for documents, chunks, and metadata
- ✅ **Multi-Layer Caching** - Redis for embeddings, retrieval, and responses
- ✅ **Smart Chunking** - Semantic, fixed, and recursive chunking strategies
- ✅ **Batch Processing** - Efficient batch embedding generation
- ✅ **Production Ready** - Connection pooling, error handling, logging

### Advanced Features
- 🔄 **Deduplication** - Hash-based file and chunk deduplication
- 📊 **Monitoring** - Query logging and performance metrics
- 🎯 **Configurable** - Extensive configuration via environment variables
- 🐳 **Docker Support** - Complete Docker Compose setup
- 🔍 **Source Attribution** - Track which documents answered queries
- ⚡ **Performance** - Optimized for speed with caching and batch processing

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     User Query                           │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
         ┌──────────────────────┐
         │   Redis Cache        │ ◄─── Check cache
         │   (Multi-layer)      │
         └──────────┬───────────┘
                    │ Cache Miss
                    ▼
         ┌──────────────────────┐
         │  Query Embedding     │
         │  (Azure OpenAI)      │
         └──────────┬───────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │     FAISS Similarity Search       │
    │  (Fast approximate NN search)     │
    └───────────────┬───────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   PostgreSQL         │
         │   (Fetch metadata)   │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Response Generation │
         │  (Azure OpenAI GPT)  │
         └──────────┬───────────┘
                    │
                    ▼
              Final Answer
```

**Tech Stack:**
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **Database:** PostgreSQL 15
- **Cache:** Redis 7
- **Embeddings:** Azure OpenAI text-embedding-ada-002
- **Generation:** Azure OpenAI GPT-4/GPT-3.5
- **Language:** Python 3.11+

---

## ⚡ Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone <your-repo-url>
cd smart-retriever-pro

# Create .env file
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Start all services
docker-compose up -d

# Initialize database
docker-compose exec app python scripts/setup_database.py

# Load sample documents
docker-compose exec app python scripts/load_documents.py data/documents

# Start interactive test
docker-compose exec app python scripts/test_system.py
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Setup PostgreSQL and Redis
# (See Installation section for details)

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Initialize database
python scripts/setup_database.py

# Load documents
python scripts/load_documents.py data/documents

# Test system
python scripts/test_system.py
```

---

## 📥 Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Azure OpenAI account with deployments

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Setup PostgreSQL

**On Ubuntu/Debian:**
```bash
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres createuser raguser
sudo -u postgres createdb smartretriever
sudo -u postgres psql -c "ALTER USER raguser WITH PASSWORD 'ragpass';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE smartretriever TO raguser;"
```

**On macOS:**
```bash
brew install postgresql@15
brew services start postgresql@15
createuser raguser
createdb smartretriever
psql -c "ALTER USER raguser WITH PASSWORD 'ragpass';"
```

**On Windows:**
Download and install from [postgresql.org](https://www.postgresql.org/download/windows/)

### Step 3: Setup Redis

**On Ubuntu/Debian:**
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

**On macOS:**
```bash
brew install redis
brew services start redis
```

**On Windows:**
Download from [redis.io](https://redis.io/download)

### Step 4: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your Azure OpenAI credentials:
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

### Step 5: Initialize Database

```bash
python scripts/setup_database.py
```

---

## ⚙️ Configuration

All configuration is managed through environment variables in `.env`:

### Azure OpenAI
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### Database
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpass
POSTGRES_DB=smartretriever
```

### Redis
```env
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL_EMBEDDING=86400  # 24 hours
CACHE_TTL_RETRIEVAL=3600   # 1 hour
CACHE_TTL_RESPONSE=1800    # 30 minutes
```

### RAG Settings
```env
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNKING_STRATEGY=semantic  # semantic, fixed, or recursive
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3
```

---

## 🚀 Usage

### 1. Load Documents

**Single document:**
```bash
python scripts/load_documents.py path/to/document.txt
```

**Directory (recursive):**
```bash
python scripts/load_documents.py data/documents
```

**Programmatic:**
```python
from rag_system import rag

# Add single document
result = rag.add_document("path/to/document.txt")

# Add directory
results = rag.add_documents_from_directory("data/documents")
```

### 2. Query the System

**Interactive mode:**
```bash
python scripts/test_system.py
```

**Programmatic:**
```python
from rag_system import rag

# Simple query
result = rag.query("What is machine learning?")
print(result['answer'])
print(f"Sources: {result['sources']}")

# Custom parameters
result = rag.query(
    "Explain Python",
    top_k=10,
    use_cache=True
)
```

### 3. Get Statistics

```python
from rag_system import rag

stats = rag.get_stats()
print(f"Total documents: {stats['database']['total_documents']}")
print(f"Total chunks: {stats['database']['total_chunks']}")
print(f"FAISS vectors: {stats['faiss']['total_vectors']}")
print(f"Cache hit rate: {stats['queries']['cache_hit_rate']:.1%}")
```

### 4. Manage Cache

```python
from rag_system import rag

# Clear all cache
rag.clear_cache()

# Query without cache
result = rag.query("Question", use_cache=False)
```

---

## 📚 API Reference

### RAGSystem

#### `add_document(filepath, filename=None, metadata=None)`
Add a document to the knowledge base.

**Parameters:**
- `filepath` (str): Path to document file
- `filename` (str, optional): Override filename
- `metadata` (dict, optional): Additional metadata

**Returns:** Dict with processing results

#### `query(question, top_k=None, use_cache=True)`
Query the RAG system.

**Parameters:**
- `question` (str): User's question
- `top_k` (int, optional): Number of chunks to retrieve
- `use_cache` (bool): Whether to use cache

**Returns:** Dict with answer and metadata

#### `get_stats()`
Get system statistics.

**Returns:** Dict with comprehensive stats

#### `delete_document(document_id)`
Delete a document.

**Parameters:**
- `document_id` (int): Document ID to delete

**Returns:** bool

#### `rebuild_faiss_index()`
Rebuild FAISS index from database.

---

## ⚡ Performance

### Benchmarks

**Query Performance:**
- First query (cold): ~1500-2000ms
- Cached query: ~50-100ms
- Speedup: **15-20x faster**

**Batch Processing:**
- 100 documents: ~2-3 minutes
- 1000 documents: ~20-30 minutes
- Throughput: ~30-50 docs/minute

### Optimization Tips

1. **Use caching** - Enable Redis for 15-20x speedup
2. **Batch embeddings** - Process multiple documents together
3. **Choose right index** - IVF for 10k+ vectors, HNSW for 100k+
4. **Adjust chunk size** - Smaller chunks = better precision, larger = more context
5. **Connection pooling** - Configured automatically for PostgreSQL

---

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U raguser -d smartretriever -h localhost
```

### Redis Connection Issues
```bash
# Check Redis is running
redis-cli ping
# Should return: PONG

# Check Redis connection
redis-cli -h localhost -p 6379
```

### FAISS Index Issues
```bash
# Rebuild FAISS index
python -c "from rag_system import rag; rag.rebuild_faiss_index()"
```

### Azure OpenAI Issues
- Verify API key in `.env`
- Check deployment names match Azure Portal
- Ensure quota is not exceeded

---

## 📊 Project Structure

```
smart-retriever-pro/
├── src/
│   ├── config.py              # Configuration
│   ├── database/
│   │   ├── postgres.py        # PostgreSQL models
│   │   ├── faiss_store.py     # FAISS vector store
│   │   └── redis_cache.py     # Redis caching
│   ├── core/
│   │   ├── chunker.py         # Document chunking
│   │   ├── embedder.py        # Embedding service
│   │   ├── retriever.py       # Retrieval logic
│   │   └── generator.py       # Response generation
│   ├── storage/
│   │   └── file_manager.py    # File management
│   └── rag_system.py          # Main orchestrator
├── scripts/
│   ├── setup_database.py      # DB initialization
│   ├── load_documents.py      # Document loader
│   └── test_system.py         # Interactive testing
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **FAISS** by Facebook AI Research
- **Azure OpenAI** by Microsoft
- **PostgreSQL** community
- **Redis** community

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation:** See `ARCHITECTURE.md` for system design
- **Blog:** See `MEDIUM_ARTICLE.md` for detailed walkthrough

---

**Built with ❤️ for production RAG systems**
