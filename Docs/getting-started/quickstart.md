# ⚡ Quick Start Guide

<div align="center">

**Get started with NeuronDB in minutes!**

[![Quick Start](https://img.shields.io/badge/quick--start-5_min-green)](.)
[![Difficulty](https://img.shields.io/badge/difficulty-easy-brightgreen)](.)

</div>

---

> [!TIP]
> Start with the [Simple Start Guide](simple-start.md) for a beginner-friendly walkthrough with detailed explanations.
>
> For complete NeuronDB ecosystem setup, see the [root QUICKSTART.md](../../QUICKSTART.md).

---

## 🎯 Goal

**What you'll accomplish:**
- ✅ Install NeuronDB extension
- ✅ Load sample data
- ✅ Run your first vector search query
- ✅ Understand basic concepts

**Time required:** 5-10 minutes

---

## 📋 Prerequisites

Before you begin, make sure you have:

- [ ] **NeuronDB installed** - See [Installation Guide](installation.md) for setup instructions
- [ ] **PostgreSQL client** - `psql` (or any SQL client)
- [ ] **5-10 minutes** - For complete quickstart

<details>
<summary><strong>🔍 Verify Prerequisites</strong></summary>

```bash
# Check if psql is installed
psql --version

# Check if Docker is installed (if using Docker)
docker --version
docker compose version
```

</details>

---

## 📦 Step 1: Install NeuronDB

If you haven't installed NeuronDB yet, choose your method:

### Option A: Docker Compose (Recommended for Quick Start) 🐳

**Fastest way to get started:**

```bash
# From repository root
docker compose up -d neurondb

# Wait for service to be healthy (30-60 seconds)
docker compose ps neurondb
```

**Expected output:**
```
NAME                STATUS
neurondb-cpu        healthy
```

> [!NOTE]
> Docker starts a PostgreSQL container with NeuronDB pre-installed. The first run takes 2 to 5 minutes to download images.

### Option B: Native Installation 🔧

**For production or custom setups:**

Follow the detailed [Installation Guide](installation.md) for native PostgreSQL installation.

---

### ✅ Verify Installation

**Test NeuronDB installation:**

```bash
# With Docker Compose
docker compose exec neurondb psql -U neurondb -d neurondb -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# Or with native PostgreSQL
psql -d your_database -c "CREATE EXTENSION IF NOT EXISTS neurondb;"
```

**Check the version:**

```bash
# With Docker Compose
docker compose exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"

# Or with native PostgreSQL
psql -d your_database -c "SELECT neurondb.version();"
```

**Expected output:**
```
 version
---------
 2.0
(1 row)
```

> [!SUCCESS]
> If you see version `2.0` or similar, NeuronDB is installed and working correctly.

---

## 📊 Step 2: Load Quickstart Data Pack

The quickstart data pack provides **~500 sample documents** with pre-generated embeddings, ready for immediate use.

<details>
<summary><strong>📚 What's in the Data Pack?</strong></summary>

- **~500 documents** - Sample text documents
- **Pre-generated embeddings** - Vector representations (384 dimensions)
- **HNSW index** - Pre-built index for fast search
- **Ready to query** - No setup required

</details>

### Option 1: Using the CLI (Recommended) 🚀

**Easiest method - handles everything automatically:**

```bash
# From repository root
./scripts/neurondb-cli.sh quickstart
```

**What it does:**
1. Creates the `quickstart_documents` table
2. Loads ~500 sample documents
3. Creates HNSW index
4. Verifies data is loaded

### Option 2: Using the Loader Script 📝

**Manual control over the process:**

```bash
# From repository root
./examples/quickstart/load_quickstart.sh
```

### Option 3: Using psql Directly 💻

**For maximum control:**

```bash
# With Docker Compose
docker compose exec neurondb psql -U neurondb -d neurondb -f examples/quickstart/quickstart_data.sql

# Or with native PostgreSQL
psql -d your_database -f examples/quickstart/quickstart_data.sql
```

---

### ✅ Verify Data Loaded

**Check that data was loaded successfully:**

```bash
# Count documents
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT COUNT(*) FROM quickstart_documents;"
```

**Expected output:**
```
 count
-------
   500
(1 row)
```

**Check table structure:**

```sql
\d quickstart_documents
```

**Expected columns:**
- `id` - Document ID
- `title` - Document title
- `content` - Document content
- `embedding` - Vector embedding (384 dimensions)

> [!SUCCESS]
> **Perfect!** Your data is loaded and ready to query.

---

## 🔍 Step 3: Try SQL Recipes

The SQL recipe library provides **ready-to-run queries** for common operations.

### Example 1: Basic Similarity Search 🎯

**Find documents similar to a specific document:**

```sql
-- Find documents similar to document #1
SELECT 
    id,
    title,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;
```

**What this does:**
1. Gets embedding of document #1
2. Calculates cosine distance to all other documents
3. Returns top 10 most similar documents

**Expected output:**
```
 id  | title                    |     distance      
-----+--------------------------+-------------------
  42 | Related Document Title   | 0.123456789012345
  87 | Another Similar Doc      | 0.234567890123456
  ...
(10 rows)
```

> [!NOTE]
> **Understanding distance:** Lower distance = more similar. Cosine distance ranges from 0 (identical) to 2 (opposite).

---

### Example 2: Query with Text Embedding 🔤

**Search using a text query:**

```sql
-- Generate embedding for query text
WITH query AS (
  SELECT embed_text('machine learning algorithms', 'all-MiniLM-L6-v2') AS q_vec
)
-- Find similar documents
SELECT 
    id,
    title,
    embedding <=> q.q_vec AS distance
FROM quickstart_documents, query q
ORDER BY embedding <=> q.q_vec
LIMIT 10;
```

**What this does:**
1. Generates embedding for "machine learning algorithms"
2. Searches for documents with similar embeddings
3. Returns top 10 results

> [!TIP]
> **Embedding models:** The `all-MiniLM-L6-v2` model is fast and works well for general text. See [Embedding Models](../../NeuronDB/docs/embedding-models.md) for more options.

---

### Example 3: Hybrid Search (Vector + Full-Text) 🔗

**Combine vector similarity with PostgreSQL full-text search:**

```sql
-- Hybrid search: vector + full-text
WITH query AS (
  SELECT 
    embed_text('machine learning', 'all-MiniLM-L6-v2') AS q_vec,
    to_tsquery('english', 'machine & learning') AS q_tsquery
)
SELECT 
    id,
    title,
    content,
    -- Combined score: 70% vector, 30% full-text
    (embedding <=> q.q_vec) * 0.7 + 
    (ts_rank(to_tsvector('english', content), q.q_tsquery) * 0.3) AS combined_score
FROM quickstart_documents, query q
WHERE to_tsvector('english', content) @@ q.q_tsquery
ORDER BY combined_score DESC
LIMIT 10;
```

**What this does:**
1. Generates vector embedding for query
2. Creates full-text search query
3. Combines both scores (70% vector, 30% text)
4. Returns top 10 results

> [!NOTE]
> **Why hybrid search?** Vector search finds semantically similar content, while full-text search finds exact keyword matches. Combining both gives better results.

---

### Example 4: Filtered Search 🎛️

**Add metadata filters to vector search:**

```sql
-- Search with filters
WITH query AS (
  SELECT embed_text('technology', 'all-MiniLM-L6-v2') AS q_vec
)
SELECT 
    id,
    title,
    embedding <=> q.q_vec AS distance
FROM quickstart_documents, query q
WHERE id > 100  -- Example filter
  AND id < 200  -- Example filter
ORDER BY embedding <=> q.q_vec
LIMIT 10;
```

**What this does:**
1. Generates query embedding
2. Applies metadata filters (e.g., date range, category)
3. Searches only within filtered subset
4. Returns top 10 results

> [!TIP]
> **Filtering tips:** Apply filters BEFORE vector search for better performance. PostgreSQL will use indexes on filter columns.

---

## 📚 More SQL Recipes

<details>
<summary><strong>📖 Additional Recipes</strong></summary>

### Reranking

**Use MMR (Maximal Marginal Relevance) for diverse results:**

```sql
SELECT * FROM neurondb.mmr_rerank(
  'quickstart_documents', 'embedding', 
  (SELECT embed_text('query text', 'all-MiniLM-L6-v2')),
  10,  -- top k
  0.7   -- lambda (diversity vs relevance)
);
```

### Batch Embedding

**Generate embeddings for multiple texts at once:**

```sql
SELECT embed_text_batch(
  ARRAY['text1', 'text2', 'text3'],
  'all-MiniLM-L6-v2'
);
```

### RAG Context Retrieval

**Retrieve context for RAG pipelines:**

```sql
SELECT * FROM neurondb.retrieve_context(
  (SELECT embed_text('query', 'all-MiniLM-L6-v2')),
  'quickstart_documents', 'embedding',
  10,  -- top k
  NULL  -- optional filters
);
```

</details>

---

## 🎓 Understanding the Results

<details>
<summary><strong>📚 Key Concepts</strong></summary>

### What is an Embedding?

An embedding is a vector. It represents the semantic meaning of text. Similar texts have similar embeddings.

**Example:**
- "machine learning" → `[0.1, 0.2, 0.3, ...]` (384 numbers)
- "artificial intelligence" → `[0.12, 0.19, 0.31, ...]` (similar numbers)
- "banana" → `[0.9, 0.1, 0.2, ...]` (different numbers)

### What is Distance?

**Distance** measures how similar two vectors are:
- **Lower distance** = more similar
- **Higher distance** = less similar

**Distance metrics:**
- `<=>` - Cosine distance (0 = identical, 2 = opposite)
- `<->` - L2/Euclidean distance (0 = identical, ∞ = different)
- `<#>` - Inner product (higher = more similar)

### What is HNSW Index?

HNSW stands for Hierarchical Navigable Small World. It is an index. It makes vector search fast.
- Without index: O(n) - checks every vector
- With HNSW: O(log n) - checks only a few vectors

**Trade-off:** Slightly less accurate but much faster.

</details>

---

## 🚀 Next Steps

**Continue your journey:**

- [ ] 📐 Read [Architecture Guide](architecture.md) to understand components
- [ ] 🧪 Try more [SQL Recipes](../../examples/sql-recipes/)
- [ ] 📚 Explore [Complete Documentation](../../documentation.md)
- [ ] 🔍 Check [Troubleshooting Guide](troubleshooting.md) if needed
- [ ] 🤖 Try [NeuronAgent Examples](../../NeuronAgent/examples/) for agent workflows
- [ ] 🔌 Explore [NeuronMCP Integration](../../NeuronMCP/docs/) for MCP tools

---

## 💡 Tips for Success

<details>
<summary><strong>💡 Helpful Tips</strong></summary>

### Performance Tips

- **Use indexes** - HNSW indexes make search 100x faster
- **Filter first** - Apply WHERE clauses before vector search
- **Limit results** - Use LIMIT to avoid processing too many rows
- **Batch operations** - Use `embed_text_batch` for multiple embeddings

### Development Tips

- **Start simple** - Get basic search working first
- **Add complexity gradually** - Try hybrid search after basic search works
- **Use examples** - Copy working examples from recipes
- **Check logs** - Use `docker compose logs` to debug issues

### Learning Tips

- **Read the docs** - Comprehensive documentation available
- **Try examples** - Hands-on learning is best
- **Experiment** - Try different queries and see what happens
- **Ask questions** - Check troubleshooting or community

</details>

---

## ❓ Common Questions

<details>
<summary><strong>❓ Frequently Asked Questions</strong></summary>

### Q: Why is my search slow?

**A:** Make sure you have an HNSW index:
```sql
CREATE INDEX ON quickstart_documents USING hnsw (embedding vector_cosine_ops);
```

### Q: How do I change the embedding model?

**A:** Use a different model name in `embed_text()`:
```sql
SELECT embed_text('text', 'sentence-transformers/all-mpnet-base-v2');
```

### Q: How do I use my own data?

**A:** Yes! Create your own table and load your data:
```sql
CREATE TABLE my_docs (id SERIAL, content TEXT, embedding vector(384));
```

### Q: How do I generate embeddings for my data?

**A:** Use `embed_text()` or `embed_text_batch()`:
```sql
UPDATE my_docs SET embedding = embed_text(content, 'all-MiniLM-L6-v2');
```

</details>

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Simple Start Guide](simple-start.md)** | Beginner-friendly walkthrough |
| **[Architecture Guide](architecture.md)** | Understand components |
| **[Installation Guide](installation.md)** | Detailed installation options |
| **[SQL Recipes](../../examples/sql-recipes/)** | Ready-to-run SQL examples |
| **[Complete Documentation](../../documentation.md)** | Full documentation index |

---

<div align="center">

[⬆ Back to Top](#-quick-start-guide) · [📚 Main Documentation](../../documentation.md) · [🚀 Simple Start](simple-start.md)

</div>
