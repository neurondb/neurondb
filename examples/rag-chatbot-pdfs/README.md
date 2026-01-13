# RAG Chatbot Over PDFs

Complete example demonstrating a RAG (Retrieval-Augmented Generation) chatbot that answers questions over PDF documents.

## Overview

This example shows how to:
- Extract text from PDF files
- Chunk documents appropriately
- Generate and store embeddings
- Retrieve relevant context
- Generate answers using LLM with retrieved context

## Quick Start

### Prerequisites

- PostgreSQL 16+ with NeuronDB extension
- Python 3.8+
- LLM API key (OpenAI, Anthropic, etc.)
- PDF files to process

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (defaults match Docker Compose setup)
export DB_HOST=localhost
export DB_PORT=5433        # Docker Compose default port
export DB_NAME=neurondb
export DB_USER=neurondb    # Docker Compose default user
export DB_PASSWORD=neurondb  # Docker Compose default password
export OPENAI_API_KEY=your_key_here

# Ingest PDFs
python ingest_pdfs.py --input-dir pdfs/

# Start chatbot
python chatbot.py
```

## Files

- `ingest_pdfs.py` - PDF extraction and ingestion pipeline
- `chatbot.py` - Interactive chatbot interface
- `retrieve_context.py` - Context retrieval logic
- `generate_answer.py` - LLM integration for answer generation
- `requirements.txt` - Python dependencies

## Usage

### Ingest PDFs

```bash
python ingest_pdfs.py --input-dir pdfs/ --chunk-size 500 --overlap 50
```

### Chat Interface

```bash
python chatbot.py
# Then type questions interactively
```

### Programmatic Usage

```python
from chatbot import RAGChatbot

chatbot = RAGChatbot()
answer = chatbot.ask("What is the main topic of the documents?")
print(answer)
```

## Configuration

Edit `config.yaml` to customize:
- Embedding model
- Chunk size and overlap
- LLM provider and model
- Retrieval parameters (k, similarity threshold)

## Related Documentation

- [RAG Playbook](../../NeuronDB/docs/rag/playbook.md) - Complete RAG guidance
- [Document Processing](../../NeuronDB/docs/rag/document-processing.md) - PDF processing
- [LLM Integration](../../NeuronDB/docs/rag/llm-integration.md) - LLM setup








