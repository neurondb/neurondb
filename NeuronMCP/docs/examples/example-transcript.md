# NeuronMCP Example Transcript

This document contains example transcripts of MCP client interactions with NeuronMCP server.

## Transcript 1: Vector Search Workflow

This transcript demonstrates a complete vector search workflow including embedding generation and similarity search.

### Client Initialization

```
Client -> Server: Initialize
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {}
    },
    "clientInfo": {
      "name": "example-client",
      "version": "1.0.0"
    }
  }
}

Server -> Client: Initialize Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {
        "listChanged": true
      },
      "resources": {
        "subscribe": true,
        "listChanged": true
      }
    },
    "serverInfo": {
      "name": "neurondb-mcp",
      "version": "1.0.0"
    }
  }
}

Client -> Server: Initialized Notification
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

### List Available Tools

```
Client -> Server: List Tools
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}

Server -> Client: Tools List Response
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "vector_search",
        "description": "Perform vector similarity search using various distance metrics",
        "inputSchema": {
          "type": "object",
          "properties": {
            "table": {"type": "string"},
            "vector_column": {"type": "string"},
            "query_vector": {"type": "array", "items": {"type": "number"}},
            "limit": {"type": "integer", "default": 10},
            "distance_metric": {"type": "string", "enum": ["l2", "cosine", "inner_product"]}
          },
          "required": ["table", "vector_column", "query_vector"]
        }
      },
      {
        "name": "generate_embedding",
        "description": "Generate text embedding using configured model",
        "inputSchema": {
          "type": "object",
          "properties": {
            "text": {"type": "string"},
            "model": {"type": "string"}
          },
          "required": ["text"]
        }
      }
    ]
  }
}
```

### Generate Embedding

```
Client -> Server: Generate Embedding
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "generate_embedding",
    "arguments": {
      "text": "What is machine learning?",
      "model": "text-embedding-ada-002"
    }
  }
}

Server -> Client: Embedding Result
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "[0.0123, -0.0456, 0.0789, ...]"
      }
    ],
    "isError": false
  }
}
```

### Perform Vector Search

```
Client -> Server: Vector Search
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "vector_search",
    "arguments": {
      "table": "documents",
      "vector_column": "embedding",
      "query_vector": [0.0123, -0.0456, 0.0789, ...],
      "limit": 5,
      "distance_metric": "cosine",
      "additional_columns": ["id", "title", "content"]
    }
  }
}

Server -> Client: Search Results
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"results\": [{\"id\": 1, \"title\": \"Introduction to ML\", \"content\": \"Machine learning is...\", \"distance\": 0.123}, {\"id\": 2, \"title\": \"Deep Learning Basics\", \"content\": \"Deep learning extends...\", \"distance\": 0.234}]}"
      }
    ],
    "isError": false
  }
}
```

## Transcript 2: ML Model Training

This transcript demonstrates training a machine learning model.

### List Available Resources

```
Client -> Server: List Resources
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "resources/list",
  "params": {}
}

Server -> Client: Resources List
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "resources": [
      {
        "uri": "neurondb://schema/tables",
        "name": "Tables",
        "description": "List all tables with vector columns",
        "mimeType": "application/json"
      },
      {
        "uri": "neurondb://models",
        "name": "Models",
        "description": "List all trained models",
        "mimeType": "application/json"
      }
    ]
  }
}
```

### Read Schema Resource

```
Client -> Server: Read Schema Resource
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "resources/read",
  "params": {
    "uri": "neurondb://schema/tables"
  }
}

Server -> Client: Schema Data
{
  "jsonrpc": "2.0",
  "id": 6,
  "result": {
    "contents": [
      {
        "uri": "neurondb://schema/tables",
        "mimeType": "application/json",
        "text": "{\"tables\": [{\"name\": \"documents\", \"schema\": \"public\", \"vector_columns\": [\"embedding\"]}, {\"name\": \"training_data\", \"schema\": \"public\", \"vector_columns\": [\"features\"]}]}"
      }
    ]
  }
}
```

### Train Model

```
Client -> Server: Train Model
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "train_model",
    "arguments": {
      "algorithm": "random_forest",
      "table": "training_data",
      "feature_col": "features",
      "label_col": "label",
      "params": {
        "n_estimators": 100,
        "max_depth": 10
      },
      "project": "classification_project"
    }
  }
}

Server -> Client: Training Started
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"model_id\": 42, \"status\": \"training\", \"message\": \"Model training started\"}"
      }
    ],
    "isError": false
  }
}
```

### Check Model Status

```
Client -> Server: Get Model Info
{
  "jsonrpc": "2.0",
  "id": 8,
  "method": "tools/call",
  "params": {
    "name": "get_model_info",
    "arguments": {
      "model_id": 42
    }
  }
}

Server -> Client: Model Information
{
  "jsonrpc": "2.0",
  "id": 8,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"model_id\": 42, \"algorithm\": \"random_forest\", \"status\": \"completed\", \"accuracy\": 0.95, \"training_time_ms\": 1234}"
      }
    ],
    "isError": false
  }
}
```

## Transcript 3: Hybrid Search with Reranking

This transcript demonstrates a hybrid search workflow with reranking.

### Hybrid Search

```
Client -> Server: Hybrid Search
{
  "jsonrpc": "2.0",
  "id": 9,
  "method": "tools/call",
  "params": {
    "name": "hybrid_search",
    "arguments": {
      "table": "documents",
      "query_vector": [0.1, 0.2, 0.3, ...],
      "query_text": "machine learning algorithms",
      "vector_column": "embedding",
      "text_column": "content",
      "vector_weight": 0.7,
      "limit": 20
    }
  }
}

Server -> Client: Hybrid Search Results
{
  "jsonrpc": "2.0",
  "id": 9,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"results\": [{\"id\": 1, \"score\": 0.85, \"content\": \"...\"}, {\"id\": 2, \"score\": 0.82, \"content\": \"...\"}]}"
      }
    ],
    "isError": false
  }
}
```

### Rerank Results

```
Client -> Server: Rerank Results
{
  "jsonrpc": "2.0",
  "id": 10,
  "method": "tools/call",
  "params": {
    "name": "rerank_cross_encoder",
    "arguments": {
      "query": "machine learning algorithms",
      "documents": ["Document 1 content...", "Document 2 content...", ...],
      "model": "ms-marco-MiniLM-L-6-v2",
      "top_k": 10
    }
  }
}

Server -> Client: Reranked Results
{
  "jsonrpc": "2.0",
  "id": 10,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"results\": [{\"index\": 0, \"score\": 0.92, \"text\": \"Document 1 content...\"}, {\"index\": 2, \"score\": 0.88, \"text\": \"Document 3 content...\"}]}"
      }
    ],
    "isError": false
  }
}
```

## Error Handling Example

### Invalid Tool Call

```
Client -> Server: Invalid Tool Call
{
  "jsonrpc": "2.0",
  "id": 11,
  "method": "tools/call",
  "params": {
    "name": "vector_search",
    "arguments": {
      "table": "documents"
      // Missing required parameter: vector_column
    }
  }
}

Server -> Client: Error Response
{
  "jsonrpc": "2.0",
  "id": 11,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "error": "VALIDATION_ERROR",
      "details": "Missing required parameter: vector_column"
    }
  }
}
```

## Notes

- All requests use JSON-RPC 2.0 protocol
- Request IDs should be unique per client session
- Vector arrays are truncated in examples with "..."
- Error responses follow JSON-RPC 2.0 error format
- Results are returned as text content containing JSON strings

## Related Documentation

- [Tool and Resource Catalog](./tool-resource-catalog.md) - Complete catalog reference
- [Client Examples](../client/README.md) - Python client usage examples
- [MCP Protocol](https://modelcontextprotocol.io/) - Official MCP specification

