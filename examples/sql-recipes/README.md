# SQL Recipe Library

Ready-to-run SQL recipes for common NeuronDB vector search and RAG patterns. Each recipe is standalone, documented, and designed to work with the [quickstart data pack](../quickstart-data/).

## Quick Start

```bash
# 1. Set up quickstart data (if not already done)
./scripts/neurondb-quickstart-data.sh

# 2. Run a recipe
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -f examples/sql-recipes/vector-search/01_basic_similarity.sql
```

## Recipe Categories

### Vector Search

Patterns for finding similar vectors and documents.

| Recipe | Description | Use Case |
|--------|-------------|----------|
| [01_basic_similarity.sql](vector-search/01_basic_similarity.sql) | Basic cosine similarity search | Find similar documents to a query |
| [02_filtered_search.sql](vector-search/02_filtered_search.sql) | Combine similarity with metadata filters | Search within categories/tags |
| [03_multiple_metrics.sql](vector-search/03_multiple_metrics.sql) | Compare L2, cosine, and inner product | Choose the right distance metric |
| [04_performance_tuning.sql](vector-search/04_performance_tuning.sql) | Optimize query performance | Speed up vector searches |

### Hybrid Search

Combine vector similarity with full-text search for better results.

| Recipe | Description | Use Case |
|--------|-------------|----------|
| [01_text_and_vector.sql](hybrid-search/01_text_and_vector.sql) | Basic hybrid search | Get best of both search methods |
| [02_weighted_combination.sql](hybrid-search/02_weighted_combination.sql) | Weighted scoring | Tune vector vs text importance |
| [03_reciprocal_rank_fusion.sql](hybrid-search/03_reciprocal_rank_fusion.sql) | RRF algorithm | Combine multiple ranking methods |
| [04_faceted_search.sql](hybrid-search/04_faceted_search.sql) | Faceted filtering | E-commerce style filtering |

### Indexing

Create and optimize vector indexes for fast queries.

| Recipe | Description | Use Case |
|--------|-------------|----------|
| [01_create_hnsw.sql](indexing/01_create_hnsw.sql) | Create HNSW indexes | Fast approximate nearest neighbor search |
| [02_create_ivf.sql](indexing/02_create_ivf.sql) | Create IVF indexes | Memory-efficient search for large datasets |
| [03_tune_parameters.sql](indexing/03_tune_parameters.sql) | Tune index parameters | Balance speed vs accuracy |
| [04_index_maintenance.sql](indexing/04_index_maintenance.sql) | Index monitoring and maintenance | Keep indexes optimized |

### RAG Patterns

Complete patterns for building RAG (Retrieval-Augmented Generation) systems.

| Recipe | Description | Use Case |
|--------|-------------|----------|
| [01_document_chunking.sql](rag-patterns/01_document_chunking.sql) | Split documents into chunks | Prepare documents for RAG |
| [02_context_retrieval.sql](rag-patterns/02_context_retrieval.sql) | Retrieve relevant context | Find relevant chunks for queries |
| [03_reranking.sql](rag-patterns/03_reranking.sql) | Improve retrieval quality | Get better context chunks |
| [04_complete_pipeline.sql](rag-patterns/04_complete_pipeline.sql) | End-to-end RAG pipeline | Complete RAG implementation |

## Prerequisites

### Required Setup

1. **NeuronDB Extension**: Installed and enabled
   ```sql
   CREATE EXTENSION IF NOT EXISTS neurondb;
   ```

2. **Quickstart Data**: Sample data loaded
   ```bash
   ./scripts/neurondb-quickstart-data.sh
   ```

3. **Full-Text Search** (for hybrid search recipes):
   ```sql
   -- Created automatically by hybrid search recipes
   -- Or manually:
   ALTER TABLE quickstart_documents ADD COLUMN IF NOT EXISTS fts_vector tsvector;
   UPDATE quickstart_documents SET fts_vector = to_tsvector('english', title || ' ' || content);
   CREATE INDEX IF NOT EXISTS quickstart_documents_fts_idx ON quickstart_documents USING gin(fts_vector);
   ```

### Table Structure

Recipes assume these tables exist (from quickstart data pack):

- `quickstart_documents` - Main documents table with embeddings
- `document_chunks` - Optional, for RAG patterns (created by chunking recipe)

## Using Recipes

### Running a Single Recipe

```bash
# Direct psql execution
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
  -f examples/sql-recipes/vector-search/01_basic_similarity.sql

# Or using docker compose
docker compose exec neurondb psql -U neurondb -d neurondb \
  -f /path/to/recipes/vector-search/01_basic_similarity.sql
```

### Customizing Recipes

All recipes are designed to be:
- **Standalone**: Can run independently
- **Customizable**: Easy to modify for your use case
- **Commented**: Clear explanations of each step

**Common customizations:**

1. **Change table name**: Replace `quickstart_documents` with your table
2. **Adjust parameters**: Modify similarity thresholds, limits, etc.
3. **Add filters**: Include your own WHERE clauses
4. **Change embedding model**: Use different models for `embed_text()`

### Recipe Format

Each recipe includes:

- **Header**: Description, prerequisites, use cases
- **Comments**: Explain each step
- **Multiple examples**: Different variations of the pattern
- **Performance notes**: Tips for optimization

## Integration with Quickstart Data Pack

Recipes are designed to work seamlessly with the quickstart data pack:

1. **Load quickstart data**:
   ```bash
   ./scripts/neurondb-quickstart-data.sh
   ```

2. **Run any recipe**:
   ```bash
   psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
     -f examples/sql-recipes/vector-search/01_basic_similarity.sql
   ```

3. **Adapt for your data**: Modify recipes to use your own tables

## Common Patterns

### Pattern 1: Basic Similarity Search

```sql
WITH query_vector AS (
    SELECT embed_text('your query') AS query_emb
)
SELECT 
    id,
    title,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM your_table, query_vector
WHERE embedding IS NOT NULL
ORDER BY embedding <=> query_vector.query_emb
LIMIT 10;
```

### Pattern 2: Hybrid Search

```sql
WITH query_vector AS (
    SELECT embed_text('your query') AS query_emb
),
vector_results AS (
    SELECT id, 1 - (embedding <=> query_vector.query_emb) AS score
    FROM your_table, query_vector
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 20
),
text_results AS (
    SELECT id, ts_rank(fts_vector, plainto_tsquery('english', 'your query')) AS score
    FROM your_table
    WHERE fts_vector @@ plainto_tsquery('english', 'your query')
    LIMIT 20
)
-- Combine results...
```

### Pattern 3: RAG Context Retrieval

```sql
WITH query_vector AS (
    SELECT embed_text('your question') AS query_emb
)
SELECT 
    chunk_text,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM document_chunks, query_vector
ORDER BY embedding <=> query_vector.query_emb
LIMIT 5;
```

## Performance Tips

1. **Use indexes**: HNSW indexes dramatically speed up queries
   ```sql
   CREATE INDEX ON your_table USING hnsw (embedding vector_cosine_ops);
   ```

2. **Tune ef_search**: Balance speed vs accuracy
   ```sql
   SET hnsw.ef_search = 100;  -- Higher = better recall, slower
   ```

3. **Limit result sets**: Use LIMIT appropriately
   ```sql
   -- Only retrieve what you need
   LIMIT 10;  -- Not 1000
   ```

4. **Cache query vectors**: Pre-compute embeddings for repeated queries

5. **Monitor performance**: Use EXPLAIN ANALYZE to understand query plans

## Troubleshooting

### "relation does not exist"

Ensure quickstart data is loaded:
```bash
./scripts/neurondb-quickstart-data.sh
```

### "embed_text() function not found"

Check NeuronDB extension:
```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
SELECT neurondb.version();
```

### "index does not exist"

Some recipes create indexes. If missing:
```sql
CREATE INDEX ON quickstart_documents USING hnsw (embedding vector_cosine_ops);
```

### Slow queries

1. Check if indexes exist: `\d+ quickstart_documents`
2. Verify index usage: `EXPLAIN ANALYZE <your_query>`
3. Adjust `ef_search`: `SET hnsw.ef_search = 40;`

## Next Steps

After exploring these recipes:

1. **Build your own**: Adapt patterns for your use case
2. **Explore examples**: Check out `examples/` directory
3. **Read documentation**: See `Docs/` for detailed guides
4. **Join community**: Share your recipes and patterns

## Related Documentation

- [Quickstart Data Pack](../quickstart-data/README.md) - Sample dataset
- [CLI Helpers](../../scripts/README.md) - Command-line utilities
- [Vector Search Guide](../../NeuronDB/docs/vector-search/) - Deep dive
- [RAG Playbook](../../NeuronDB/docs/rag/) - Complete RAG guide

## Contributing Recipes

Found a useful pattern? Share it!

1. Create a new SQL file in appropriate category
2. Follow the recipe format (header, comments, examples)
3. Test with quickstart data
4. Submit a pull request

**Recipe template:**

```sql
-- ============================================================================
-- Recipe: Your Recipe Name
-- ============================================================================
-- Purpose: What it does
-- 
-- Prerequisites:
--   - What's needed to run this
--
-- Use Cases:
--   - When to use this pattern
--
-- Performance Notes:
--   - Optimization tips
-- ============================================================================

-- Example 1: Basic usage
SELECT ...

-- Example 2: Advanced usage
SELECT ...
```

---

**Last Updated:** 2026-01-08  
**Recipes Version:** 1.0.0  
**Compatible with:** NeuronDB 2.0+



