# 🚀 SmartRetriever Pro - Step-by-Step Setup Guide

Follow these steps **in order** to get the production system running.

---

## ✅ Step 1: Verify Prerequisites

Check you have everything installed:

```bash
# Python 3.8+
python --version

# PostgreSQL
psql --version

# Redis
redis-cli --version

# Docker (optional but recommended)
docker --version
docker-compose --version
```

---

## ✅ Step 2: Start Infrastructure (Docker)

**Option A: Using Docker (Recommended)**

```bash
cd smart-retriever-pro/docker
docker-compose up -d

# Verify services are running
docker-compose ps

# Should see:
# - smartretriever-postgres (healthy)
# - smartretriever-redis (healthy)
```

**Option B: Manual Installation**

```bash
# Start PostgreSQL
pg_ctl -D /usr/local/var/postgres start

# Start Redis
redis-server --daemonize yes

# Create database
createdb smartretriever
```

---

## ✅ Step 3: Install Python Dependencies

```bash
cd smart-retriever-pro
pip install -r requirements.txt

# Verify installation
python -c "import faiss; import redis; import psycopg2; print('All dependencies installed!')"
```

---

## ✅ Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
nano .env  # or use your favorite editor
```

**Required settings to update:**

```bash
# Azure OpenAI (MUST UPDATE)
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_API_KEY=YOUR-API-KEY-HERE
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# PostgreSQL (if using Docker, defaults are fine)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=smartretriever
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpass

# Redis (if using Docker, defaults are fine)
REDIS_HOST=localhost
REDIS_PORT=6379
```

---

## ✅ Step 5: Initialize Database

```bash
python scripts/setup_database.py

# You should see:
# ✓ PostgreSQL connection established
# ✓ Database tables created/verified
# ✓ Database setup completed successfully
```

---

## ✅ Step 6: Run Tests

```bash
python scripts/test_system.py

# Select option 2 (Run All Tests) to verify everything works
```

This will:
1. ✅ Initialize all components
2. ✅ Create sample documents  
3. ✅ Test document processing
4. ✅ Test retrieval
5. ✅ Test generation
6. ✅ Test caching

---

## ✅ Step 7: Try Interactive Mode

```bash
python scripts/test_system.py

# Select option 7 (Interactive Mode)
# Ask questions about the sample documents
```

---

## 🎯 Quick Verification Checklist

Run these commands to verify each component:

### PostgreSQL
```bash
docker exec smartretriever-postgres psql -U raguser -d smartretriever -c "\dt"

# Should show tables: documents, chunks, embeddings, query_logs
```

### Redis
```bash
docker exec smartretriever-redis redis-cli ping

# Should return: PONG
```

### FAISS
```python
python -c "from src.database.faiss_store import faiss_store; print(faiss_store.get_stats())"

# Should show FAISS index info
```

### Full System
```python
python -c "from src.rag_system import rag_system; print(rag_system.get_stats())"

# Should show complete system statistics
```

---

## 🐛 Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'src'"

**Fix:**
```bash
# Make sure you're in the project root
cd smart-retriever-pro

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run Python with -m
python -m scripts.test_system
```

### Issue: "Connection to PostgreSQL failed"

**Fix:**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker logs smartretriever-postgres

# Restart if needed
docker-compose restart postgres
```

### Issue: "Redis connection failed"

**Fix:**
```bash
# Check if Redis is running
docker ps | grep redis

# Test connection
redis-cli -h localhost -p 6379 ping

# Restart if needed
docker-compose restart redis
```

### Issue: "Azure OpenAI authentication failed"

**Fix:**
```bash
# Verify your credentials
cat .env | grep AZURE_OPENAI

# Test API directly
curl -H "api-key: YOUR-API-KEY" \
     "YOUR-ENDPOINT/openai/deployments?api-version=2024-02-15-preview"
```

---

## 📊 Verify Installation Success

Run this verification script:

```python
# test_installation.py
from src.rag_system import rag_system
from src.config import config

print("="*60)
print("SmartRetriever Pro - Installation Verification")
print("="*60)

# Test 1: Configuration
print("\n1. Configuration...")
try:
    config.validate()
    print("   ✓ All config variables set")
except Exception as e:
    print(f"   ✗ Config error: {e}")
    exit(1)

# Test 2: Database
print("\n2. PostgreSQL...")
try:
    from src.database.postgres import db_manager
    stats = db_manager.get_stats()
    print(f"   ✓ Connected (Documents: {stats['total_documents']})")
except Exception as e:
    print(f"   ✗ Database error: {e}")
    exit(1)

# Test 3: FAISS
print("\n3. FAISS...")
try:
    from src.database.faiss_store import faiss_store
    stats = faiss_store.get_stats()
    print(f"   ✓ Initialized (Vectors: {stats['total_vectors']})")
except Exception as e:
    print(f"   ✗ FAISS error: {e}")
    exit(1)

# Test 4: Redis
print("\n4. Redis...")
try:
    from src.database.redis_cache import cache
    is_healthy = cache.health_check()
    print(f"   ✓ Connected (Healthy: {is_healthy})")
except Exception as e:
    print(f"   ✗ Redis error: {e}")

# Test 5: Azure OpenAI
print("\n5. Azure OpenAI...")
try:
    from src.core.embedder import embedder
    test_emb = embedder.embed_text("test")
    print(f"   ✓ API working (Embedding dim: {len(test_emb)})")
except Exception as e:
    print(f"   ✗ Azure OpenAI error: {e}")
    exit(1)

print("\n" + "="*60)
print("✅ All components verified successfully!")
print("="*60)
print("\nYou're ready to use SmartRetriever Pro!")
print("Run: python scripts/test_system.py")
```

Save as `test_installation.py` and run:

```bash
python test_installation.py
```

---

## 🎉 Success!

If all steps completed successfully, you now have:

- ✅ Production RAG system running
- ✅ PostgreSQL storing metadata
- ✅ FAISS for vector search
- ✅ Redis for caching
- ✅ All components integrated

**Next steps:**

1. **Add your own documents**: Place files in `data/documents/`
2. **Process them**: Use the test script or write your own code
3. **Query the system**: Ask questions and get AI-powered answers!

---

## 📚 What Each File Does

| File | Purpose |
|------|---------|
| `src/config.py` | Configuration management |
| `src/database/postgres.py` | PostgreSQL operations |
| `src/database/faiss_store.py` | FAISS vector search |
| `src/database/redis_cache.py` | Redis caching |
| `src/core/chunker.py` | Document chunking |
| `src/core/embedder.py` | Azure OpenAI embeddings |
| `src/storage/file_manager.py` | File storage |
| `src/rag_system.py` | Main RAG orchestrator |
| `scripts/setup_database.py` | Database initialization |
| `scripts/test_system.py` | Interactive testing |

---

## 🚀 Ready to Build!

You now have a **production-ready RAG system**. Time to:

- Add your documents
- Build applications
- Deploy to production
- Scale to millions of documents

Happy building! 🎉
