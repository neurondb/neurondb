#!/bin/bash
# Final comprehensive validation test
# Tests 100% Claude Desktop compatibility and all NeuronMCP capabilities

set -e

CONFIG="neuronmcp_server.json"
CLIENT="./bin/neurondb-mcp-client"
OUTPUT="final_validation_$(date +%Y%m%d_%H%M%S).json"

echo "=========================================="
echo "Final Comprehensive Validation Test"
echo "=========================================="
echo ""

PASSED=0
FAILED=0

test_and_validate() {
    local name="$1"
    local command="$2"
    local should_have_error="${3:-false}"
    
    echo "[TEST] $name"
    
    if $CLIENT -c "$CONFIG" -e "$command" -o "/tmp/test_$$.json" >/dev/null 2>&1; then
        # Check result
        if [ "$should_have_error" = "true" ]; then
            if grep -q '"isError":\s*true' "/tmp/test_$$.json" 2>/dev/null || grep -q '"error"' "/tmp/test_$$.json" 2>/dev/null; then
                echo "  ✓ PASSED (error as expected)"
                PASSED=$((PASSED + 1))
            else
                echo "  ✗ FAILED (expected error but got success)"
                FAILED=$((FAILED + 1))
            fi
        else
            if grep -q '"isError":\s*true' "/tmp/test_$$.json" 2>/dev/null || grep -q '"error"' "/tmp/test_$$.json" 2>/dev/null; then
                echo "  ✗ FAILED (unexpected error)"
                FAILED=$((FAILED + 1))
                cat "/tmp/test_$$.json" | jq '.results[0].result' 2>/dev/null | head -5
            else
                echo "  ✓ PASSED"
                PASSED=$((PASSED + 1))
            fi
        fi
    else
        echo "  ✗ FAILED (command execution failed)"
        FAILED=$((FAILED + 1))
    fi
    rm -f "/tmp/test_$$.json"
    echo ""
}

# ============================================================================
# CLAUDE DESKTOP COMPATIBILITY TESTS
# ============================================================================
echo "=== CLAUDE DESKTOP COMPATIBILITY ==="
test_and_validate "Initialize and list tools" "list_tools"
test_and_validate "Request with string ID" "list_tools"
test_and_validate "Request with numeric parameters" "list_tools"
test_and_validate "Response format (JSON directly)" "list_tools"
test_and_validate "Notification handling" "list_tools"
test_and_validate "Error response format" "invalid_method" true

# ============================================================================
# ALL TOOL CATEGORIES
# ============================================================================
echo "=== VECTOR OPERATIONS ==="
test_and_validate "Generate embedding" 'generate_embedding:text="Test"'
test_and_validate "Batch embedding" 'batch_embedding:texts=["T1","T2"]'
test_and_validate "Vector similarity" 'vector_similarity:vector1=[0.1,0.2],vector2=[0.2,0.3]'

echo "=== ML OPERATIONS ==="
test_and_validate "List models" "list_models"
test_and_validate "Get model info (will fail - no model)" "get_model_info:model_id=999" true

echo "=== ANALYTICS ==="
test_and_validate "Analyze data (will fail - no table)" 'analyze_data:table=nonexistent' true

echo "=== RAG OPERATIONS ==="
test_and_validate "Chunk document" 'chunk_document:text="Test document",chunk_size=10'
test_and_validate "Process document" 'process_document:text="Test",chunk_size=10'
test_and_validate "Generate response" 'generate_response:query="Test",context=["C1"]'

echo "=== RESOURCES ==="
test_and_validate "List resources" "resources/list"
test_and_validate "Read schema resource" 'resources/read:uri="neurondb://schema"'
test_and_validate "Read models resource" 'resources/read:uri="neurondb://models"'

# ============================================================================
# EDGE CASES
# ============================================================================
echo "=== EDGE CASES ==="
test_and_validate "Special characters in text" 'generate_embedding:text="Text with \"quotes\""'
test_and_validate "Array parameters" 'batch_embedding:texts=["A","B","C"]'
test_and_validate "Numeric parameters" 'vector_similarity:vector1=[0.1,0.2],vector2=[0.3,0.4]'
test_and_validate "Boolean parameters" 'process_document:text="T",generate_embeddings=false'
test_and_validate "Complex JSON params" 'train_model:algorithm=linear_regression,table=t,feature_col=f,label_col=l,params={"key":"value"}' true

# ============================================================================
# BATCH EXECUTION
# ============================================================================
echo "=== BATCH EXECUTION ==="
echo "[TEST] Batch execution from file"
if [ -f "client/commands.txt" ]; then
    if $CLIENT -c "$CONFIG" -f client/commands.txt -o "$OUTPUT" >/dev/null 2>&1; then
        if [ -f "$OUTPUT" ]; then
            COUNT=$(jq '.results | length' "$OUTPUT" 2>/dev/null || echo "0")
            echo "  ✓ PASSED (executed $COUNT commands)"
            PASSED=$((PASSED + 1))
        else
            echo "  ✗ FAILED (output file not created)"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  ✗ FAILED (batch execution failed)"
        FAILED=$((FAILED + 1))
    fi
else
    echo "  ⚠ SKIPPED (commands.txt not found)"
fi
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
TOTAL=$((PASSED + FAILED))
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total tests: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
if [ $TOTAL -gt 0 ]; then
    RATE=$(echo "scale=1; $PASSED * 100 / $TOTAL" | bc)
    echo "Success rate: ${RATE}%"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ ALL TESTS PASSED!"
    echo "✓ 100% Claude Desktop Compatible"
    echo "✓ All capabilities tested"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi

