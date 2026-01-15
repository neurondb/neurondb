# NeuronMCP Features

NeuronMCP is a Model Context Protocol (MCP) server that provides access to NeuronDB capabilities through the MCP protocol.

## Core Features

### MCP Protocol Support
- **JSON-RPC 2.0**: Full JSON-RPC 2.0 protocol support
- **Stdio Communication**: Communication via stdin/stdout
- **Tool Discovery**: Dynamic tool discovery
- **Resource Catalog**: Comprehensive resource catalog
- **Claude Desktop Compatible**: Optimized for Claude Desktop

### Tool Registration Modes
- **PostgreSQL-Only Mode**: Default mode with 5 essential PostgreSQL tools
- **Category-Based Selection**: Select tools by category
- **Full Tool Set**: Access to all 100+ tools
- **Custom Tool Registration**: Register custom tools

## Tool Categories

### Vector Operations (8 tools)
- Vector similarity search with multiple distance metrics
- L2, cosine, and inner product search
- Vector arithmetic operations
- Vector distance calculations
- Unified vector similarity

### Vector Quantization (7 tools)
- Multiple quantization types (int8, fp16, binary, uint8, ternary, int4)
- Quantization analysis
- Quantize/dequantize operations

### Embeddings (8 tools)
- Text embedding generation
- Batch embedding generation
- Image embeddings
- Multimodal embeddings (text + image)
- Cached embeddings
- Model configuration management

### Hybrid Search (7 tools)
- Semantic + lexical search
- Reciprocal rank fusion (RRF)
- Semantic + keyword search
- Multi-vector search
- Faceted vector search
- Temporal vector search
- Diverse vector search

### Reranking (6 tools)
- Cross-encoder reranking
- LLM-powered reranking
- Cohere reranking
- ColBERT reranking
- Learning-to-rank reranking
- Ensemble reranking

### Machine Learning (8 tools)
- Model training (classification, regression, clustering)
- Single and batch predictions
- Model evaluation
- Model management
- Model export

**Supported Algorithms:**
- Classification: logistic, random_forest, svm, knn, decision_tree, naive_bayes
- Regression: linear_regression, ridge, lasso
- Clustering: kmeans, gmm, dbscan, hierarchical

### Analytics (7 tools)
- General data analysis
- Clustering analysis
- Dimensionality reduction (PCA)
- Outlier detection
- Quality metrics (Recall@K, Precision@K)
- Data drift detection
- Topic modeling

### Time Series (1 tool)
- Time series analysis
- ARIMA, forecasting, seasonal decomposition

### AutoML (1 tool)
- Automated ML pipeline
- Task type detection
- Constraint handling

### ONNX (4 tools)
- ONNX model import
- ONNX model export
- ONNX model info
- ONNX predictions

### Index Management (6 tools)
- HNSW index creation
- IVF index creation
- Index status monitoring
- Index dropping
- Auto-tuning for HNSW
- Auto-tuning for IVF

### RAG Operations (4 tools)
- Document processing
- Context retrieval
- Response generation
- Document chunking

### Workers & GPU (2 tools)
- Background worker management
- GPU information retrieval

### Vector Graph (1 tool)
- Graph operations (BFS, DFS, PageRank, community detection)
- Graph-based vector operations

### Vecmap Operations (1 tool)
- Sparse vector operations
- Multiple distance metrics
- Vector arithmetic

### Dataset Loading (1 tool)
- Load from HuggingFace
- Load from URL
- Load from GitHub
- Load from S3
- Load from local filesystem
- Auto-embedding
- Auto-index creation

### PostgreSQL (8 tools)
- Version information
- Server statistics
- Database listing
- Connection information
- Lock information
- Replication status
- Configuration settings
- Extension listing

## Resource Catalog

### Schema Resources
- Table listings
- Table schema details
- Column definitions
- Index listings
- Index details

### Model Resources
- Model listings
- Model metadata
- Model metrics
- Prediction history

### Index Resources
- Index listings
- Index statistics
- Index build status

### Configuration Resources
- Current configuration
- GPU configuration
- LLM provider configuration

### Worker Resources
- Worker listings
- Worker status
- Worker queue status

### Statistics Resources
- Overview statistics
- Performance metrics
- Usage statistics

## Advanced Features

### Server Capabilities
- **Version Negotiation**: Server version and capability negotiation
- **Tool Versioning**: Version information for tools
- **Model Versioning**: Version information for models
- **Feature Flags**: Feature flag support
- **Pagination**: Pagination support
- **Streaming**: Streaming support
- **Dry Run**: Dry run mode
- **Idempotency**: Idempotent operations
- **Audit Logging**: Audit logging
- **Scoped Auth**: Scoped authentication
- **Rate Limiting**: Rate limiting
- **Output Validation**: Output validation
- **Tool Versioning**: Tool version management
- **Deprecation**: Deprecation support
- **Composite Tools**: Composite tool support
- **Resource Catalog**: Resource catalog support

### Configuration
- **Environment Variables**: Configuration via environment variables
- **Config File**: JSON configuration file support
- **Feature Toggles**: Enable/disable features
- **Logging Configuration**: Configurable logging

### Performance
- **Connection Pooling**: Efficient database connection pooling
- **Query Optimization**: Optimized queries
- **Caching**: Response caching
- **Batch Operations**: Batch processing support

## Integration Features

### Claude Desktop
- **Optimized Compatibility**: Optimized for Claude Desktop
- **Tool Limit Handling**: Handles Claude Desktop's 5-tool limit
- **Category Selection**: Category-based tool selection

### MCP Clients
- **Standard MCP**: Works with any MCP-compatible client
- **JSON-RPC 2.0**: Standard JSON-RPC 2.0 protocol
- **Error Handling**: Comprehensive error handling

## Use Cases

### Vector Search
- Semantic search across documents
- Similarity search
- Multi-vector search

### RAG Applications
- Document processing
- Context retrieval
- Response generation

### Machine Learning
- Model training
- Predictions
- Model evaluation

### Data Analysis
- Statistical analysis
- Clustering
- Outlier detection

### Database Management
- Schema inspection
- Index management
- Performance monitoring

## Security Features

### Authentication
- **Database Authentication**: Uses PostgreSQL authentication
- **Connection Security**: Secure database connections

### Validation
- **Input Validation**: Comprehensive input validation
- **SQL Injection Protection**: Protection against SQL injection
- **Output Validation**: Output validation

## Operational Features

### Logging
- **Structured Logging**: Structured logging support
- **Configurable Levels**: Configurable log levels
- **Error Stack Traces**: Detailed error information

### Monitoring
- **Health Checks**: Health check support
- **Performance Metrics**: Performance tracking
- **Resource Usage**: Resource usage monitoring

### Error Handling
- **Graceful Errors**: Graceful error handling
- **Error Codes**: Standard error codes
- **Error Messages**: Clear error messages

