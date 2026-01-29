# SmartRetriever Pro - Quick Command Reference

## 🐳 Docker Commands

### Initial Setup
```bash
cd docker
docker compose down -v
docker compose build --no-cache app
docker compose up -d postgres redis
docker compose run --rm app python scripts/setup_database.py
docker compose up -d app
```

### Start Services
```bash
# Detached mode
docker compose up -d

# Attached mode (see logs)
docker compose up app
```

### Stop Services
```bash
docker compose down
```

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f app
docker compose logs -f postgres
docker compose logs -f redis
```

### Restart Services
```bash
docker compose restart app
```

### Execute Commands
```bash
docker compose exec app python scripts/test_system.py
docker compose exec app python scripts/load_documents.py data/documents
```

### Clean Up
```bash
# Stop and remove containers
docker compose down

# Stop and remove containers + volumes
docker compose down -v
```

---

## 💻 Local Commands

### Setup
```bash
# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Reset and initialize database
python scripts/setup_database.py --reset
python scripts/setup_database.py
```

### Clear Data
```bash
# Clear FAISS index and storage
rm -rf data/faiss_index/*
rm -rf data/storage/*
rm -rf data/uploads/*
rm -rf data/processed/*
```

### Run Application
```bash
# Test system
python scripts/test_system.py

# Start Streamlit UI
streamlit run ui/app.py
```

### Database Operations
```bash
# Initialize database
python scripts/setup_database.py

# Reset database (clears all data)
python scripts/setup_database.py --reset

# Load documents
python scripts/load_documents.py data/documents
```

---

## 🔧 Maintenance Commands

### Redis
```bash
# Local
redis-cli FLUSHALL

# Docker
docker compose exec redis redis-cli FLUSHALL
```

### PostgreSQL
```bash
# Local - Connect to database
psql -U raguser -d smartretriever

# Docker - Connect to database
docker compose exec postgres psql -U raguser smartretriever

# Create database (if needed)
createdb smartretriever  # Local
docker compose exec postgres createdb -U raguser smartretriever  # Docker
```

### System Reset (Complete Clean)
```bash
# Docker
cd docker
docker compose down -v
rm -rf ../data/faiss_index/*
rm -rf ../data/storage/*
rm -rf ../data/uploads/*
docker compose build --no-cache app
docker compose up -d postgres redis
docker compose run --rm app python scripts/setup_database.py
docker compose up -d app

# Local
rm -rf data/faiss_index/*
rm -rf data/storage/*
rm -rf data/uploads/*
rm -rf data/processed/*
python scripts/setup_database.py --reset
python scripts/setup_database.py
```

---

## 🐛 Debug Commands

### Check System Status
```bash
# Docker
docker compose ps

# Application health check
docker compose exec app python -c "from src.rag_system import rag; print(rag.get_stats())"
```

### Access Container Shell
```bash
docker compose exec app bash
docker compose exec postgres bash
docker compose exec redis bash
```

### Check Logs with Timestamps
```bash
docker compose logs -f --timestamps app
```

### Test Network Connectivity
```bash
docker compose exec app ping postgres
docker compose exec app ping redis
```

---

## 📊 System Information

### Get Statistics
```python
# In Python
from src.rag_system import rag
stats = rag.get_stats()
print(f"Documents: {stats['database']['total_documents']}")
print(f"Chunks: {stats['database']['total_chunks']}")
print(f"FAISS vectors: {stats['faiss']['total_vectors']}")
```

### Check FAISS Index
```bash
# Local
ls -lh data/faiss_index/

# Docker
docker compose exec app ls -lh data/faiss_index/
```

### Check Storage Size
```bash
# Local
du -sh data/storage/

# Docker
docker compose exec app du -sh data/storage/
```

---

## 🚀 Quick Start Workflows

### First Time Setup (Docker)
```bash
cd docker
docker compose down -v
docker compose build --no-cache app
docker compose up -d postgres redis
sleep 5
docker compose run --rm app python scripts/setup_database.py
docker compose up -d app
# Access UI at http://localhost:8501
```

### First Time Setup (Local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_database.py
streamlit run ui/app.py
# Access UI at http://localhost:8501
```

### Daily Development (Docker)
```bash
cd docker
docker compose up -d
# Access UI at http://localhost:8501
```

### Daily Development (Local)
```bash
source .venv/bin/activate
streamlit run ui/app.py
# Access UI at http://localhost:8501
```

### Add New Documents
```bash
# Docker
docker compose exec app python scripts/load_documents.py data/documents

# Local
python scripts/load_documents.py data/documents
```

### Test System
```bash
# Docker
docker compose exec app python scripts/test_system.py

# Local
python scripts/test_system.py
```

---

## ⚙️ Configuration Switch

### Switch from Local to Docker
1. Edit `src/config.py`
2. Comment out local configuration:
   ```python
   # DATABASE_URL = "postgresql://raguser:ragpass@localhost:5432/smartretriever"
   # REDIS_HOST = "localhost"
   ```
3. Uncomment Docker configuration:
   ```python
   DATABASE_URL = "postgresql://raguser:ragpass@postgres:5432/smartretriever"
   REDIS_HOST = "redis"
   ```

### Switch from Docker to Local
1. Edit `src/config.py`
2. Comment out Docker configuration
3. Uncomment local configuration
4. Ensure PostgreSQL and Redis are running locally:
   ```bash
   brew services start postgresql@15
   brew services start redis
   ```

---

## 📝 Useful Aliases (Add to .bashrc or .zshrc)

```bash
# Docker
alias dcu='docker compose up -d'
alias dcd='docker compose down'
alias dcl='docker compose logs -f'
alias dcr='docker compose restart'

# SmartRetriever
alias sr-up='cd docker && docker compose up -d'
alias sr-down='cd docker && docker compose down'
alias sr-logs='cd docker && docker compose logs -f app'
alias sr-test='docker compose exec app python scripts/test_system.py'
alias sr-ui='streamlit run ui/app.py'
```

---

## 🆘 Emergency Commands

### System Not Responding
```bash
# Kill all containers
docker compose down -v

# Restart from scratch
docker compose build --no-cache app
docker compose up -d
```

### Database Locked
```bash
# Docker
docker compose restart postgres

# Local
brew services restart postgresql@15
```

### Redis Issues
```bash
# Docker
docker compose restart redis

# Local
brew services restart redis
```

### Full Nuclear Reset
```bash
# WARNING: This deletes EVERYTHING
cd docker
docker compose down -v
docker volume prune -f
rm -rf ../data/faiss_index/*
rm -rf ../data/storage/*
rm -rf ../data/uploads/*
rm -rf ../data/processed/*
docker compose build --no-cache app
docker compose up -d postgres redis
sleep 10
docker compose run --rm app python scripts/setup_database.py
docker compose up -d app
```
