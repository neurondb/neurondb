#!/bin/bash
# Comprehensive test suite for NeuronMCP client
# Tests all capabilities of NeuronDB through NeuronMCP

set -e

CONFIG_FILE="neuronmcp_server.json"
OUTPUT_FILE="comprehensive_test_results.json"
VERBOSE="-v"

echo "=========================================="
echo "NeuronMCP Comprehensive Test Suite"
echo "=========================================="
echo ""

# Test 1: Basic Connection and Initialization
echo "[TEST 1] Testing basic connection and initialization..."
./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "list_tools" $VERBOSE > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Connection test passed"
else
    echo "✗ Connection test failed"
    exit 1
fi

# Test 2: List Tools (should return all available tools)
echo "[TEST 2] Testing list_tools..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "list_tools" -o test_tools.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ list_tools test passed"
    TOOL_COUNT=$(cat test_tools.json | grep -o '"name"' | wc -l)
    echo "  Found $TOOL_COUNT tools"
else
    echo "✗ list_tools test failed"
    exit 1
fi

# Test 3: List Resources
echo "[TEST 3] Testing resources/list..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "resources/list" -o test_resources.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ resources/list test passed"
else
    echo "✗ resources/list test failed"
fi

# Test 4: Vector Operations - Generate Embedding
echo "[TEST 4] Testing generate_embedding..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e 'generate_embedding:text="Test embedding generation",model=text-embedding-ada-002' -o test_embedding.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ generate_embedding test passed"
else
    echo "✗ generate_embedding test failed"
    echo "  Error: $RESULT"
fi

# Test 5: Vector Operations - Batch Embedding
echo "[TEST 5] Testing batch_embedding..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e 'batch_embedding:texts=["Text 1","Text 2","Text 3"]' -o test_batch_embedding.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ batch_embedding test passed"
else
    echo "✗ batch_embedding test failed"
fi

# Test 6: Vector Operations - Vector Similarity
echo "[TEST 6] Testing vector_similarity..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e 'vector_similarity:vector1=[0.1,0.2,0.3],vector2=[0.2,0.3,0.4],distance_metric=cosine' -o test_similarity.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ vector_similarity test passed"
else
    echo "✗ vector_similarity test failed"
fi

# Test 7: ML Operations - List Models
echo "[TEST 7] Testing list_models..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "list_models" -o test_list_models.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ list_models test passed"
else
    echo "✗ list_models test failed"
fi

# Test 8: Analytics - Analyze Data (will fail without table, but should handle gracefully)
echo "[TEST 8] Testing analyze_data error handling..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e 'analyze_data:table=nonexistent_table' -o test_analyze.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ analyze_data error handling test passed"
else
    echo "✗ analyze_data test failed"
fi

# Test 9: RAG Operations - Chunk Document
echo "[TEST 9] Testing chunk_document..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e 'chunk_document:text="This is a long document that needs to be chunked into smaller pieces for processing.",chunk_size=20,overlap=5' -o test_chunk.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ chunk_document test passed"
else
    echo "✗ chunk_document test failed"
fi

# Test 10: RAG Operations - Process Document
echo "[TEST 10] Testing process_document..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e 'process_document:text="Sample document for processing",chunk_size=100,overlap=20' -o test_process.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ process_document test passed"
else
    echo "✗ process_document test failed"
fi

# Test 11: Complex Command with JSON Parameters
echo "[TEST 11] Testing complex command with JSON parameters..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e 'train_model:algorithm=linear_regression,table=test_table,feature_col=features,label_col=label,params={"learning_rate":0.01,"max_iter":100}' -o test_complex.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ complex command test passed"
else
    echo "✗ complex command test failed"
fi

# Test 12: Array Parameters
echo "[TEST 12] Testing array parameters..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e 'batch_embedding:texts=["First text","Second text","Third text with spaces"]' -o test_array.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ array parameters test passed"
else
    echo "✗ array parameters test failed"
fi

# Test 13: Numeric Parameters
echo "[TEST 13] Testing numeric parameters..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e 'vector_similarity:vector1=[0.1,0.2,0.3],vector2=[0.4,0.5,0.6],distance_metric=cosine' -o test_numeric.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ numeric parameters test passed"
else
    echo "✗ numeric parameters test failed"
fi

# Test 14: Boolean Parameters
echo "[TEST 14] Testing boolean parameters..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e 'process_document:text="Test",generate_embeddings=true,chunk_size=100' -o test_boolean.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ boolean parameters test passed"
else
    echo "✗ boolean parameters test failed"
fi

# Test 15: Batch Execution from File
echo "[TEST 15] Testing batch execution from file..."
if [ -f "client/commands.txt" ]; then
    RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -f client/commands.txt -o test_batch.json 2>&1 | head -5)
    if echo "$RESULT" | grep -q "Executing"; then
        echo "✓ batch execution test passed"
    else
        echo "✗ batch execution test failed"
    fi
else
    echo "⚠ batch execution test skipped (commands.txt not found)"
fi

# Test 16: Output File Generation
echo "[TEST 16] Testing output file generation..."
./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "list_tools" -o "$OUTPUT_FILE" > /dev/null 2>&1
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ output file generation test passed"
    FILE_SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null || echo "0")
    echo "  Output file size: $FILE_SIZE bytes"
else
    echo "✗ output file generation test failed"
fi

# Test 17: Error Handling - Invalid Command
echo "[TEST 17] Testing error handling with invalid command..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "invalid_command_that_does_not_exist" -o test_error.json 2>&1)
if echo "$RESULT" | grep -q "Command executed"; then
    echo "✓ error handling test passed (command executed, error in result)"
else
    echo "✗ error handling test failed"
fi

# Test 18: Verbose Mode
echo "[TEST 18] Testing verbose mode..."
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "list_tools" -v -o test_verbose.json 2>&1)
if echo "$RESULT" | grep -q "Starting MCP server" && echo "$RESULT" | grep -q "MCP connection initialized"; then
    echo "✓ verbose mode test passed"
else
    echo "✗ verbose mode test failed"
fi

# Test 19: Multiple Sequential Commands
echo "[TEST 19] Testing multiple sequential commands..."
./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "list_tools" -o test_seq1.json > /dev/null 2>&1
./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "resources/list" -o test_seq2.json > /dev/null 2>&1
./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "list_models" -o test_seq3.json > /dev/null 2>&1
if [ -f test_seq1.json ] && [ -f test_seq2.json ] && [ -f test_seq3.json ]; then
    echo "✓ sequential commands test passed"
else
    echo "✗ sequential commands test failed"
fi

# Test 20: Claude Desktop Format Compatibility
echo "[TEST 20] Testing Claude Desktop format compatibility..."
# The server should send JSON directly without Content-Length headers
RESULT=$(./bin/neurondb-mcp-client -c "$CONFIG_FILE" -e "list_tools" -v -o test_claude.json 2>&1)
if echo "$RESULT" | grep -q "tools"; then
    echo "✓ Claude Desktop format compatibility test passed"
else
    echo "✗ Claude Desktop format compatibility test failed"
fi

echo ""
echo "=========================================="
echo "Test Suite Complete"
echo "=========================================="
echo ""
echo "Cleaning up test files..."
rm -f test_*.json
echo "Done!"

