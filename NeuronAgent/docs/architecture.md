# NeuronAgent Architecture

## Overview

NeuronAgent is an AI agent runtime system that integrates with NeuronDB for LLM and vector operations.

## Components

### Database Layer (`internal/db/`)
- **Models**: Go structs representing database entities
- **Connection**: Connection pool management with health checks
- **Queries**: All SQL queries with prepared statements

### Agent Runtime (`internal/agent/`)
- **Runtime**: Main execution engine with state machine
- **Memory**: HNSW-based vector search for long-term memory
- **LLM**: Integration with NeuronDB LLM functions
- **Context**: Context loading and management
- **Prompt**: Prompt construction with templating

### Tools System (`internal/tools/`)
- **Registry**: Tool registration and discovery
- **Executor**: Tool execution with timeout
- **Validators**: JSON Schema validation
- **Handlers**: SQL, HTTP, Code, Shell tools

### API Layer (`internal/api/`)
- **Handlers**: REST API endpoints
- **WebSocket**: Streaming support
- **Middleware**: Auth, rate limiting, CORS, logging

### Authentication (`internal/auth/`)
- **API Keys**: Bcrypt hashing and validation
- **Rate Limiting**: Per-key rate limits
- **Roles**: RBAC support

### Background Jobs (`internal/jobs/`)
- **Queue**: PostgreSQL-based job queue (SKIP LOCKED)
- **Worker**: Worker pool with graceful shutdown
- **Processor**: Job type processors

## Data Flow

1. User sends message via API
2. Runtime loads agent and session
3. Context is loaded (recent messages + memory chunks)
   - Memory chunks retrieved via NeuronDB vector search (HNSW)
   - Embeddings generated via `neurondb_embed()`
4. Prompt is built
5. LLM generates response (via NeuronDB)
   - Uses `neurondb_llm_generate()` or `neurondb_llm_complete()`
   - Streaming via `neurondb_llm_generate_stream()`
6. Tool calls are parsed and executed if needed
   - NeuronDB tools: RAG, Hybrid Search, Reranking, Vector, ML, Analytics
7. Final response is generated
8. Messages and memory chunks are stored
   - New memories embedded via `neurondb_embed()`
   - Stored with vector embeddings for future retrieval

## NeuronDB Integration

NeuronAgent deeply integrates with NeuronDB PostgreSQL extension for:

### Embeddings
- **Function**: `neurondb_embed()`, `neurondb_embed_batch()`
- **Usage**: Memory chunk embeddings, query embeddings for search
- **Client**: `pkg/neurondb/EmbeddingClient`

### LLM Operations
- **Functions**: `neurondb_llm_generate()`, `neurondb_llm_complete()`, `neurondb_llm_generate_stream()`
- **Usage**: Agent responses, planning, reflection
- **Client**: `pkg/neurondb/LLMClient`
- **Features**: Streaming, tool calling, multimodal (image analysis)

### Vector Search
- **Operations**: Vector similarity (`<=>`), HNSW indexes
- **Usage**: Memory retrieval, semantic search
- **Client**: `pkg/neurondb/VectorClient`
- **Indexes**: HNSW for fast approximate nearest neighbor search

### RAG Operations
- **Functions**: `neurondb_chunk_text()`, `neurondb_generate_answer()`, `neurondb_rerank_results()`
- **Usage**: Document processing, context retrieval, answer generation
- **Client**: `pkg/neurondb/RAGClient`

### Additional NeuronDB Features
- **Hybrid Search**: Combines vector and full-text search
- **Reranking**: Improves search result quality
- **ML Operations**: Machine learning capabilities
- **Analytics**: Performance analytics and insights

### Integration Points

1. **Memory System** (`internal/agent/memory.go`)
   - Uses `EmbeddingClient` for generating embeddings
   - Uses vector similarity search for retrieval
   - Stores embeddings in `memory_chunks` table with vector columns

2. **Runtime** (`internal/agent/runtime.go`)
   - Uses `LLMClient` for all LLM operations
   - Uses `EmbeddingClient` for context loading
   - Integrates with NeuronDB tools via tool registry

3. **Tool Registry** (`internal/tools/registry.go`)
   - Registers NeuronDB tools: RAG, Hybrid Search, Reranking, Vector, ML, Analytics
   - Tools use respective NeuronDB clients

4. **Advanced RAG** (`internal/agent/advanced_rag.go`)
   - Uses `RAGClient`, `HybridSearchClient`, `RerankingClient`
   - Implements multi-vector RAG, temporal RAG, faceted RAG

## Security

- API key authentication with bcrypt hashing
- Rate limiting per API key
- Tool execution sandboxing
- SQL tool restricted to read-only queries
- HTTP tool with URL allowlist
- Code tool with directory restrictions
- Shell tool with command whitelist

