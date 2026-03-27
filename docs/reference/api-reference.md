# API Reference

The NeuronDB **extension** exposes its functionality through **SQL**: functions, operators, types, and GUCs.

## SQL API

| Document | Description |
|----------|-------------|
| **[SQL API](sql-api.md)** | Functions, operators, and aggregates |
| **[Data types](data-types.md)** | Vector and related types |
| **[Configuration](configuration.md)** | GUC variables |
| **[Top functions](top_functions.md)** | Most commonly used functions |

## Usage

All interaction with NeuronDB is via PostgreSQL:

```sql
CREATE EXTENSION neurondb;

-- Vector search
SELECT * FROM my_table ORDER BY embedding <=> query_vector LIMIT 10;

-- Embeddings
SELECT embed_text('hello world', 'all-MiniLM-L6-v2');

-- RAG
SELECT neurondb.rag_query('my_collection', 'What is X?', 5);
```

See [Getting started](../getting-started/simple-start.md) and [Quick start](../getting-started/quickstart.md) for setup and examples.

---

[Reference](../reference/) · [Documentation](../readme.md)
