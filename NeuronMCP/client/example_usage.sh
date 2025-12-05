#!/bin/bash
# Example usage script for NeuronMCP CLI Client

# Make sure the script is executable
chmod +x neurondb_mcp_client.py

# Example 1: List all available tools
echo "Example 1: Listing all tools..."
./neurondb_mcp_client.py -c ../neuronmcp_server.json -e "list_tools" -v

# Example 2: Execute a single vector search command
echo -e "\nExample 2: Vector search..."
./neurondb_mcp_client.py -c ../neuronmcp_server.json \
  -e "vector_search:table=documents,vector_column=embedding,query_vector=[0.1,0.2,0.3],limit=5" \
  -o example_vector_search.json

# Example 3: Execute commands from file
echo -e "\nExample 3: Batch execution from commands file..."
./neurondb_mcp_client.py -c ../neuronmcp_server.json \
  -f commands.txt \
  -o batch_results.json \
  -v

echo -e "\nDone! Check the output JSON files for results."

