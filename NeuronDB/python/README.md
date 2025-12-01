# NeuronDB Python SDK

Developer-friendly Python interface for NeuronDB PostgreSQL extension.

## Installation

```bash
pip install neurondb
```

## Quick Start

```python
from neurondb import Client, Model, VectorStore, RAG

# Connect to database
client = Client("postgresql://user:pass@localhost/dbname")

# Train a model
model = client.train(
    algorithm="random_forest",
    table="training_data",
    target="label",
    features=["feature1", "feature2", "feature3"]
)

# Make predictions
predictions = model.predict("test_data")

# Vector search
store = VectorStore(client, table="documents", column="embedding")
results = store.search(query_vector, k=10)

# RAG pipeline
rag = RAG(client, table="documents", embedding_model="all-MiniLM-L6-v2")
answer = rag.generate("What is machine learning?")

# Cleanup
client.close()
```

## Features

- **Model Training**: Train ML models with simple Python API
- **Vector Search**: High-level vector similarity search
- **RAG Pipeline**: Complete RAG workflow in Python
- **Async Support**: Asynchronous operations with asyncio
- **Type Hints**: Full type annotations for better IDE support

## Documentation

See [NeuronDB Documentation](https://neurondb.ai/docs) for detailed API reference.

