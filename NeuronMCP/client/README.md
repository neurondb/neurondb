# NeuronMCP CLI Client

A professional, modular Python CLI client for interacting with NeuronMCP servers. This client provides a command-line interface that supports the same configuration format as Claude Desktop and enables batch command execution.

## Features

- ✅ **100% Claude Desktop Compatible**: Uses the exact same configuration file format
- ✅ **Modular Architecture**: Clean separation of concerns with professional code structure
- ✅ **Batch Execution**: Execute commands from a file
- ✅ **Result Export**: Automatically saves results to JSON files
- ✅ **Comprehensive Tool Support**: Supports all NeuronMCP server capabilities
- ✅ **Verbose Mode**: Detailed output for debugging

## Installation

```bash
cd NeuronMCP/client
pip install -r requirements.txt
chmod +x neurondb_mcp_client.py
```

## Requirements

- Python 3.8+
- Access to NeuronMCP server binary

## Usage

### Basic Usage

```bash
# Execute a single command
./neurondb_mcp_client.py -c ../neuronmcp_server.json -e "list_tools"

# Execute commands from file
./neurondb_mcp_client.py -c ../neuronmcp_server.json -f commands.txt

# Save results to specific file
./neurondb_mcp_client.py -c ../neuronmcp_server.json -f commands.txt -o my_results.json

# Verbose mode
./neurondb_mcp_client.py -c ../neuronmcp_server.json -e "list_tools" -v
```

### Command Format

Commands can be specified in two formats:

1. **Simple command** (no arguments):
   ```
   list_tools
   ```

2. **Command with arguments**:
   ```
   tool_name:arg1=val1,arg2=val2,arg3=[1,2,3]
   ```

### Examples

```bash
# List all tools
./neurondb_mcp_client.py -c neuronmcp_server.json -e "list_tools"

# Vector search
./neurondb_mcp_client.py -c neuronmcp_server.json -e "vector_search:table=docs,vector_column=embedding,query_vector=[0.1,0.2,0.3],limit=10"

# Train a model
./neurondb_mcp_client.py -c neuronmcp_server.json -e "train_model:algorithm=linear_regression,table=data,feature_col=features,label_col=label"

# Execute batch commands
./neurondb_mcp_client.py -c neuronmcp_server.json -f commands.txt -o results.json
```

## Configuration File Format

The client uses the NeuronMCP server configuration file (`neuronmcp_server.json`), which follows the same format as Claude Desktop:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "/path/to/neurondb-mcp",
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "neurondb",
        "NEURONDB_USER": "nbduser",
        "NEURONDB_PASSWORD": "password"
      }
    }
  }
}
```

## Commands File Format

The commands file (`commands.txt`) contains one command per line. Lines starting with `#` are treated as comments and ignored.

Example:
```
# List tools
list_tools

# Vector search
vector_search:table=documents,vector_column=embedding,query_vector=[0.1,0.2,0.3],limit=10

# Train model
train_model:algorithm=linear_regression,table=data,feature_col=features,label_col=label
```

## Output Format

Results are saved in JSON format with the following structure:

```json
{
  "metadata": {
    "start_time": "2025-12-03T22:30:00",
    "end_time": "2025-12-03T22:30:05",
    "total_commands": 10,
    "successful_commands": 9,
    "failed_commands": 1
  },
  "results": [
    {
      "timestamp": "2025-12-03T22:30:01",
      "command": "list_tools",
      "result": {
        "tools": [...]
      }
    },
    ...
  ]
}
```

## Available Tools

The client supports all NeuronMCP server tools:

### Vector Operations
- `vector_search` - Vector similarity search
- `vector_search_l2` - L2 distance search
- `vector_search_cosine` - Cosine similarity search
- `vector_search_inner_product` - Inner product search
- `vector_similarity` - Calculate vector similarity
- `generate_embedding` - Generate text embedding
- `batch_embedding` - Batch generate embeddings
- `create_vector_index` - Create vector index

### ML Operations
- `train_model` - Train ML model
- `predict` - Make prediction
- `predict_batch` - Batch predictions
- `evaluate_model` - Evaluate model
- `list_models` - List all models
- `get_model_info` - Get model information
- `delete_model` - Delete model
- `export_model` - Export model

### Analytics
- `cluster_data` - Cluster data
- `detect_outliers` - Detect outliers
- `reduce_dimensionality` - Dimensionality reduction
- `analyze_data` - Analyze data

### RAG Operations
- `process_document` - Process document
- `chunk_document` - Chunk document
- `retrieve_context` - Retrieve context
- `generate_response` - Generate response

### Indexing
- `create_hnsw_index` - Create HNSW index
- `create_ivf_index` - Create IVF index
- `index_status` - Get index status
- `drop_index` - Drop index
- `tune_hnsw_index` - Tune HNSW index
- `tune_ivf_index` - Tune IVF index

### Hybrid Search
- `hybrid_search` - Hybrid vector + keyword search

## Architecture

The client is built with a modular architecture:

```
client/
├── neurondb_mcp_client.py    # Main CLI entry point
├── mcp_client/               # Client package
│   ├── __init__.py          # Package initialization
│   ├── client.py            # Main client class
│   ├── config.py            # Configuration loading
│   ├── protocol.py          # MCP protocol (JSON-RPC 2.0)
│   ├── transport.py         # Stdio transport layer
│   ├── commands.py          # Command parsing
│   └── output.py            # Output management
├── commands.txt             # Example commands file
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Error Handling

The client handles errors gracefully:
- Configuration errors are reported with clear messages
- Command parsing errors include the problematic command
- Tool execution errors are captured in the output file
- Network/transport errors are logged with details

## Development

The codebase follows professional Python practices:
- Type hints throughout
- Comprehensive docstrings
- Modular design with clear separation of concerns
- Error handling at all levels
- Clean, readable code

## License

See the main NeuronMCP LICENSE file.

