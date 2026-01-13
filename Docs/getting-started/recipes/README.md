# SQL Recipe Library

**Ready-to-run SQL queries for NeuronDB vector search and hybrid retrieval**

This recipe library provides copy-paste ready SQL queries for common NeuronDB operations. Each recipe file contains multiple queries with explanations, use cases, and complexity ratings.

## Quick Start

1. **Load quickstart data** (if you haven't already):
   ```bash
   ./examples/quickstart/load_quickstart.sh
   ```

2. **Try a recipe**:
   ```bash
   psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -f 01_vector_search.sql
   ```

3. **Or copy individual queries** from the recipe files into your SQL client.

## Recipe Files

| File | Description | Complexity | Use Cases |
|------|-------------|------------|-----------|
| **[01_vector_search.sql](01_vector_search.sql)** | Vector similarity search patterns | ⭐ - ⭐⭐⭐ | KNN search, distance metrics, similarity ranking |
| **[02_hybrid_search.sql](02_hybrid_search.sql)** | Vector + full-text search combinations | ⭐⭐ - ⭐⭐⭐ | Combined semantic and keyword search, RRF fusion |
| **[03_filtered_search.sql](03_filtered_search.sql)** | Vector search with SQL filters | ⭐ - ⭐⭐⭐ | Category filters, date ranges, conditional search |
| **[04_indexing.sql](04_indexing.sql)** | Index creation patterns | ⭐ - ⭐⭐⭐ | HNSW indexes, IVF indexes, parameter tuning |
| **[05_embedding_generation.sql](05_embedding_generation.sql)** | Embedding generation patterns | ⭐ - ⭐⭐⭐ | Text embeddings, batch generation, model selection |

## Recipe Categories

### 1. Vector Search (`01_vector_search.sql`)

Basic to advanced vector similarity search queries.

**Key Recipes:**
- Basic cosine similarity search
- L2/Euclidean distance search
- Inner product search
- Distance threshold filtering
- Multi-metric comparison

**When to Use:**
- Finding similar documents
- Recommendation systems
- Semantic search
- Similarity-based ranking

### 2. Hybrid Search (`02_hybrid_search.sql`)

Combine vector similarity with full-text search.

**Key Recipes:**
- Weighted combination (vector + FTS)
- Reciprocal Rank Fusion (RRF)
- Query text search
- Boosted fields
- Performance comparison

**When to Use:**
- Combining semantic and keyword search
- Improving search recall
- Handling both synonym and exact match queries
- Balancing relevance and precision

### 3. Filtered Search (`03_filtered_search.sql`)

Vector search with SQL WHERE clause filters.

**Key Recipes:**
- Category filtering
- Date range filtering
- Multiple filter conditions
- Top-K per category
- Complex filters

**When to Use:**
- Category-based recommendations
- Time-based search (recent items)
- User-specific filtering
- Excluding unwanted results
- Multi-criteria search

### 4. Indexing (`04_indexing.sql`)

Create and manage vector indexes.

**Key Recipes:**
- HNSW index creation (L2, cosine, inner product)
- IVF index creation
- Index parameter tuning
- Multiple indexes
- Performance optimization

**When to Use:**
- Setting up new tables
- Optimizing query performance
- Tuning for your workload
- Supporting multiple distance metrics

### 5. Embedding Generation (`05_embedding_generation.sql`)

Generate embeddings from text.

**Key Recipes:**
- Single embedding generation
- Batch generation
- Updating existing documents
- Query embedding generation
- Error handling

**When to Use:**
- Ingesting new documents
- Updating existing data
- Generating query vectors
- Batch processing

## Prerequisites

- **NeuronDB Extension**: Installed and enabled
- **Quickstart Data Pack**: Recommended for trying recipes (see [examples/quickstart/](../../../examples/quickstart/))
- **PostgreSQL Client**: `psql` or any SQL client
- **Table**: `quickstart_documents` (from quickstart pack) or your own table

## Usage Patterns

### Pattern 1: Run Entire Recipe File

```bash
# Run all recipes in a file
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -f 01_vector_search.sql

# Or with Docker Compose
docker compose exec neurondb psql -U neurondb -d neurondb -f /path/to/recipes/01_vector_search.sql
```

### Pattern 2: Copy Individual Queries

1. Open the recipe file
2. Find the recipe you need
3. Copy the SQL query
4. Paste into your SQL client
5. Modify as needed for your use case

### Pattern 3: Interactive Learning

1. Run a simple recipe first
2. Understand the output
3. Try modifying the query
4. Experiment with parameters
5. Move to more complex recipes

## Complexity Guide

- ⭐ **Beginner**: Simple queries, basic operations, easy to understand
- ⭐⭐ **Intermediate**: Moderate complexity, requires some SQL knowledge
- ⭐⭐⭐ **Advanced**: Complex queries, performance optimization, advanced patterns

## Common Use Cases

### Use Case: Build a Recommendation System

1. **Setup**: Load quickstart data or your own data
2. **Indexing**: Use `04_indexing.sql` to create HNSW index
3. **Search**: Use `01_vector_search.sql` for similarity search
4. **Filtering**: Use `03_filtered_search.sql` for user/category filters

### Use Case: Semantic Search with Keywords

1. **Setup**: Load data with embeddings
2. **Hybrid**: Use `02_hybrid_search.sql` for combined search
3. **Filtering**: Use `03_filtered_search.sql` for additional filters

### Use Case: Ingest New Documents

1. **Generation**: Use `05_embedding_generation.sql` to generate embeddings
2. **Indexing**: Use `04_indexing.sql` to create indexes
3. **Search**: Use `01_vector_search.sql` to query

## Tips for Using Recipes

### 1. Start Simple
Begin with ⭐ complexity recipes to understand the patterns.

### 2. Read Comments
Each recipe includes comments explaining what it does and when to use it.

### 3. Customize for Your Schema
Recipes use `quickstart_documents` table - adjust table/column names for your schema.

### 4. Understand Parameters
HNSW parameters (`m`, `ef_construction`), search parameters (`ef_search`), etc.

### 5. Test Performance
Use `EXPLAIN ANALYZE` to verify index usage and query performance.

### 6. Combine Recipes
Mix recipes from different files to build complex workflows.

## Adapting Recipes

### Change Table Name

Recipes use `quickstart_documents` - replace with your table:

```sql
-- Recipe uses:
FROM quickstart_documents

-- Change to:
FROM your_table_name
```

### Change Column Names

Adjust embedding column name if different:

```sql
-- Recipe uses:
embedding <=> ...

-- Change to:
your_embedding_column <=> ...
```

### Change Dimensions

Adjust vector dimensions to match your model:

```sql
-- Recipe uses:
embedding vector(384)

-- Change to:
embedding vector(1536)  -- For OpenAI ada-002
```

## Performance Tips

### 1. Use Indexes
All vector search queries benefit from HNSW indexes. See `04_indexing.sql`.

### 2. Tune Search Parameters
Adjust `hnsw.ef_search` (HNSW) or `ivfflat.probes` (IVF) for speed vs. recall.

### 3. Use Filters
Apply WHERE clauses before vector search to reduce search space.

### 4. Batch Operations
When generating embeddings, use batch processing (see `05_embedding_generation.sql`).

### 5. Monitor Performance
Use `EXPLAIN ANALYZE` to verify index usage and query plans.

## Troubleshooting

### "Table quickstart_documents does not exist"

**Solution**: Load the quickstart data pack first:
```bash
./examples/quickstart/load_quickstart.sh
```

Or adapt the recipe to use your own table.

### "Index not used in query"

**Solution**: 
1. Verify index exists: `\d table_name`
2. Check query uses correct operator (<=> for cosine, <-> for L2)
3. Increase `ef_search` if using HNSW
4. Use `EXPLAIN ANALYZE` to see query plan

### "Embedding generation fails"

**Solution**:
1. Check embedding model is configured
2. Verify API keys (if using external models)
3. Check network connectivity
4. See `05_embedding_generation.sql` for error handling examples

### "Query is slow"

**Solution**:
1. Create/verify indexes (see `04_indexing.sql`)
2. Adjust search parameters (`ef_search`, `probes`)
3. Add filters to reduce search space
4. Use `EXPLAIN ANALYZE` to identify bottlenecks

## Related Resources

- **[Quickstart Data Pack](../../../examples/quickstart/)** - Sample data for trying recipes
- **[Quickstart Guide](../quickstart.md)** - Complete quickstart guide
- **[CLI Helpers](../../../scripts/neurondb-cli.sh)** - Command-line tools for common tasks
- **[NeuronDB Documentation](../../../NeuronDB/docs/)** - Comprehensive documentation

## Contributing

Found a useful pattern not in the recipes? Consider:

1. Adding it to the appropriate recipe file
2. Following the existing format (comments, complexity rating, use case)
3. Testing it works with quickstart data
4. Submitting a pull request

## Feedback

- **Issues**: [GitHub Issues](https://github.com/neurondb/neurondb/issues)
- **Documentation**: [neurondb.ai/docs](https://neurondb.ai/docs)
- **Questions**: Check the troubleshooting section above

---

**Happy Querying! 🚀**



