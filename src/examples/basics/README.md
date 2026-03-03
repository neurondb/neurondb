# NeuronDB Basics - Simple Examples for Beginners

Simple, copy-paste friendly examples to get started with NeuronDB.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up database connection** (optional, uses defaults for Docker setup):
   ```bash
   export DB_HOST=localhost
   export DB_PORT=5433
   export DB_NAME=neurondb
   export DB_USER=neurondb
   export DB_PASSWORD=neurondb
   ```
   
   **Note:** Default values match the Docker Compose setup. If using a different PostgreSQL installation, adjust accordingly.

3. **Run an example:**
   ```bash
   python 01_basic_vectors.py
   ```

## Examples

### 01_basic_vectors.py
**What you'll learn:**
- Create tables with vector columns
- Insert vectors
- Basic vector operations

**Run:**
```bash
python 01_basic_vectors.py
```

**Time:** ~5 seconds

---

### 02_simple_embeddings.py
**What you'll learn:**
- Generate embeddings from text using SentenceTransformers
- Store embeddings in NeuronDB
- Basic similarity search

**Run:**
```bash
python 02_simple_embeddings.py
```

**Time:** ~30 seconds (includes model download on first run)

---

### 03_similarity_search.py
**What you'll learn:**
- Different distance metrics (cosine, L2)
- Filtered searches
- Finding nearest neighbors

**Run:**
```bash
python 03_similarity_search.py
```

**Time:** ~30 seconds

---

### 04_basic_index.py
**What you'll learn:**
- Create HNSW indexes
- Understand index parameters
- See performance improvements

**Run:**
```bash
python 04_basic_index.py
```

**Time:** ~1 minute (includes index creation)

---

### 05_simple_rag.py
**What you'll learn:**
- Build a simple RAG system
- Retrieve relevant context
- Use context for responses

**Run:**
```bash
python 05_simple_rag.py
```

**Time:** ~30 seconds

**Note:** This example simulates LLM responses. For production, integrate with OpenAI, Anthropic, or other LLM APIs.

---

## Prerequisites

- **PostgreSQL 16+** with NeuronDB extension installed
- **Python 3.8+**
- **pip** for installing dependencies

## Database Setup

If you don't have NeuronDB set up yet:

**Using Docker (recommended):**
```bash
# From repository root
cd ../..
docker compose up -d neurondb

# Wait for service to be healthy
docker compose ps

# Create extension (if not already created)
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# Verify extension is installed
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT neurondb.version();"
```

**Or follow the installation guide:**
- See [NeuronDB Installation](../../NeuronDB/INSTALL.md)

## What Each Example Does

| Example | Lines of Code | Complexity | Key Concepts |
|---------|--------------|------------|--------------|
| 01_basic_vectors | ~80 | ⭐ | Vector types, basic operations |
| 02_simple_embeddings | ~90 | ⭐⭐ | Text embeddings, similarity |
| 03_similarity_search | ~120 | ⭐⭐ | Distance metrics, filtering |
| 04_basic_index | ~140 | ⭐⭐⭐ | Indexing, performance |
| 05_simple_rag | ~150 | ⭐⭐⭐ | RAG, context retrieval |

## Learning Path

1. **Start here:** `01_basic_vectors.py` - Understand vector basics
2. **Then:** `02_simple_embeddings.py` - Learn about embeddings
3. **Next:** `03_similarity_search.py` - Master similarity search
4. **Advanced:** `04_basic_index.py` - Optimize with indexes
5. **Build:** `05_simple_rag.py` - Create a RAG system

## Common Issues

### "Extension neurondb does not exist"
Make sure NeuronDB is installed:
```bash
cd ../../NeuronDB
make install
```

### "Connection refused"
Check if PostgreSQL is running:
```bash
# For Docker setup
docker compose ps neurondb

# Test connection
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT 1;"

# If using native PostgreSQL installation
psql -h localhost -U neurondb -d neurondb -c "SELECT 1;"
```

### "ModuleNotFoundError"
Install dependencies:
```bash
pip install -r requirements.txt
```

### Slow embedding generation
The first run downloads the model (~80MB). Subsequent runs are faster.

## Next Steps

After completing these examples:

1. **Explore advanced examples:**
   - [Semantic Search](../semantic-search-docs/) - Full document search system
   - [RAG Chatbot](../rag-chatbot-pdfs/) - Complete RAG with PDFs
   - [Data Loading](../data_loading/) - Load datasets from HuggingFace

2. **Read the documentation:**
   - [NeuronDB Docs](../../NeuronDB/docs/)
   - [Vector Search Guide](../../NeuronDB/docs/vector-search/)
   - [Quick Start](../../QUICKSTART.md)

3. **Build your own project:**
   - Start with one of these examples
   - Modify for your use case
   - Add your own data

## Tips

- **Start simple:** Begin with `01_basic_vectors.py` even if you're experienced
- **Read the code:** Each example is heavily commented
- **Experiment:** Modify the examples to see what happens
- **Check the output:** Each example prints what it's doing

## Support

- **Documentation:** [neurondb.ai/docs](https://neurondb.ai/docs)
- **GitHub Issues:** [github.com/neurondb/neurondb/issues](https://github.com/neurondb/neurondb/issues)

---

**Happy Learning! 🚀**




