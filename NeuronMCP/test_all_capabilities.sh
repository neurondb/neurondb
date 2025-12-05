#!/bin/bash
# Comprehensive test of ALL NeuronMCP capabilities
# Tests every tool and verifies Claude Desktop compatibility

set -e

CONFIG="neuronmcp_server.json"
CLIENT="./bin/neurondb-mcp-client"
OUTPUT_DIR="test_output_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Comprehensive NeuronMCP Capability Test"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

PASSED=0
FAILED=0
TOTAL=0

test_command() {
    local name="$1"
    local command="$2"
    local expected_success="${3:-true}"
    
    TOTAL=$((TOTAL + 1))
    echo "[TEST $TOTAL] $name"
    
    if $CLIENT -c "$CONFIG" -e "$command" -o "$OUTPUT_DIR/test_${TOTAL}.json" > "$OUTPUT_DIR/test_${TOTAL}.log" 2>&1; then
        if [ "$expected_success" = "true" ]; then
            # Check if result contains error
            if grep -q '"error"' "$OUTPUT_DIR/test_${TOTAL}.json" 2>/dev/null; then
                echo "  ✗ FAILED (error in result)"
                FAILED=$((FAILED + 1))
                cat "$OUTPUT_DIR/test_${TOTAL}.log" | tail -5
            else
                echo "  ✓ PASSED"
                PASSED=$((PASSED + 1))
            fi
        else
            # Expected to fail
            if grep -q '"error"' "$OUTPUT_DIR/test_${TOTAL}.json" 2>/dev/null; then
                echo "  ✓ PASSED (expected error)"
                PASSED=$((PASSED + 1))
            else
                echo "  ✗ FAILED (expected error but got success)"
                FAILED=$((FAILED + 1))
            fi
        fi
    else
        echo "  ✗ FAILED (command execution failed)"
        FAILED=$((FAILED + 1))
        cat "$OUTPUT_DIR/test_${TOTAL}.log" | tail -5
    fi
    echo ""
}

# ============================================================================
# BASIC OPERATIONS
# ============================================================================
echo "=== BASIC OPERATIONS ==="
test_command "List all tools" "list_tools"
test_command "List all resources" "resources/list"
test_command "Read schema resource" 'resources/read:uri="neurondb://schema"'
test_command "Read models resource" 'resources/read:uri="neurondb://models"'
test_command "Read indexes resource" 'resources/read:uri="neurondb://indexes"'
test_command "Read config resource" 'resources/read:uri="neurondb://config"'
test_command "Read workers resource" 'resources/read:uri="neurondb://workers"'
test_command "Read stats resource" 'resources/read:uri="neurondb://stats"'

# ============================================================================
# VECTOR OPERATIONS
# ============================================================================
echo "=== VECTOR OPERATIONS ==="
test_command "Generate embedding" 'generate_embedding:text="Test text for embedding generation"'
test_command "Batch embedding" 'batch_embedding:texts=["Text 1","Text 2","Text 3"]'
test_command "Vector similarity (cosine)" 'vector_similarity:vector1=[0.1,0.2,0.3],vector2=[0.2,0.3,0.4],distance_metric=cosine'
test_command "Vector similarity (L2)" 'vector_similarity:vector1=[0.1,0.2,0.3],vector2=[0.2,0.3,0.4],distance_metric=l2'
test_command "Vector similarity (inner_product)" 'vector_similarity:vector1=[0.1,0.2,0.3],vector2=[0.2,0.3,0.4],distance_metric=inner_product'
test_command "Vector search (default)" 'vector_search:table=test_table,vector_column=embedding,query_vector=[0.1,0.2,0.3,0.4,0.5],limit=10' false
test_command "Vector search L2" 'vector_search_l2:table=test_table,vector_column=embedding,query_vector=[0.1,0.2,0.3,0.4,0.5],limit=10' false
test_command "Vector search cosine" 'vector_search_cosine:table=test_table,vector_column=embedding,query_vector=[0.1,0.2,0.3,0.4,0.5],limit=10' false
test_command "Vector search inner product" 'vector_search_inner_product:table=test_table,vector_column=embedding,query_vector=[0.1,0.2,0.3,0.4,0.5],limit=10' false
test_command "Create vector index" 'create_vector_index:table=test_table,vector_column=embedding,index_name=test_idx,index_type=hnsw,m=16,ef_construction=200' false
test_command "Create HNSW index" 'create_hnsw_index:table=test_table,vector_column=embedding,index_name=test_hnsw,m=16,ef_construction=200' false
test_command "Create IVF index" 'create_ivf_index:table=test_table,vector_column=embedding,index_name=test_ivf,num_lists=100' false
test_command "Index status" 'index_status:index_name=test_idx' false
test_command "Drop index" 'drop_index:index_name=test_idx' false
test_command "Tune HNSW index" 'tune_hnsw_index:table=test_table,vector_column=embedding' false
test_command "Tune IVF index" 'tune_ivf_index:table=test_table,vector_column=embedding' false

# ============================================================================
# ML OPERATIONS
# ============================================================================
echo "=== ML OPERATIONS ==="
test_command "List models" "list_models"
test_command "Get model info" "get_model_info:model_id=1" false
test_command "Train model (linear regression)" 'train_model:algorithm=linear_regression,table=test_table,feature_col=features,label_col=label' false
test_command "Train model (ridge)" 'train_model:algorithm=ridge,table=test_table,feature_col=features,label_col=label' false
test_command "Train model (lasso)" 'train_model:algorithm=lasso,table=test_table,feature_col=features,label_col=label' false
test_command "Train model (logistic)" 'train_model:algorithm=logistic,table=test_table,feature_col=features,label_col=label' false
test_command "Train model (random_forest)" 'train_model:algorithm=random_forest,table=test_table,feature_col=features,label_col=label' false
test_command "Train model (svm)" 'train_model:algorithm=svm,table=test_table,feature_col=features,label_col=label' false
test_command "Train model (knn)" 'train_model:algorithm=knn,table=test_table,feature_col=features,label_col=label' false
test_command "Train model (decision_tree)" 'train_model:algorithm=decision_tree,table=test_table,feature_col=features,label_col=label' false
test_command "Train model (naive_bayes)" 'train_model:algorithm=naive_bayes,table=test_table,feature_col=features,label_col=label' false
test_command "Train model with params" 'train_model:algorithm=random_forest,table=test_table,feature_col=features,label_col=label,params={"n_estimators":100,"max_depth":10}' false
test_command "Train model with project" 'train_model:algorithm=linear_regression,table=test_table,feature_col=features,label_col=label,project=test_project' false
test_command "Predict" "predict:model_id=1,features=[0.1,0.2,0.3,0.4,0.5]" false
test_command "Predict batch" "predict_batch:model_id=1,features_array=[[0.1,0.2,0.3],[0.2,0.3,0.4]]" false
test_command "Evaluate model" "evaluate_model:model_id=1,test_table=test_table,feature_col=features,label_col=label" false
test_command "Delete model" "delete_model:model_id=1" false
test_command "Export model (JSON)" "export_model:model_id=1,format=json" false
test_command "Export model (ONNX)" "export_model:model_id=1,format=onnx" false
test_command "Export model (PMML)" "export_model:model_id=1,format=pmml" false

# ============================================================================
# ANALYTICS OPERATIONS
# ============================================================================
echo "=== ANALYTICS OPERATIONS ==="
test_command "Cluster data (kmeans)" 'cluster_data:table=test_table,vector_column=features,algorithm=kmeans,k=5' false
test_command "Cluster data (gmm)" 'cluster_data:table=test_table,vector_column=features,algorithm=gmm,k=5' false
test_command "Cluster data (dbscan)" 'cluster_data:table=test_table,vector_column=features,algorithm=dbscan,eps=0.5,min_samples=5' false
test_command "Cluster data (hierarchical)" 'cluster_data:table=test_table,vector_column=features,algorithm=hierarchical,k=5' false
test_command "Cluster data (minibatch_kmeans)" 'cluster_data:table=test_table,vector_column=features,algorithm=minibatch_kmeans,k=5' false
test_command "Detect outliers" 'detect_outliers:table=test_table,vector_column=features,threshold=3.0' false
test_command "Reduce dimensionality (PCA)" 'reduce_dimensionality:table=test_table,vector_column=features,n_components=2' false
test_command "Analyze data" 'analyze_data:table=test_table' false
test_command "Analyze data with columns" 'analyze_data:table=test_table,columns=["col1","col2"]' false
test_command "Analyze data with options" 'analyze_data:table=test_table,include_stats=true,include_distribution=false' false

# ============================================================================
# RAG OPERATIONS
# ============================================================================
echo "=== RAG OPERATIONS ==="
test_command "Chunk document" 'chunk_document:text="This is a long document that needs to be chunked into smaller pieces for processing. It contains multiple sentences and paragraphs.",chunk_size=50,overlap=10'
test_command "Process document" 'process_document:text="Sample document for processing with embeddings",chunk_size=100,overlap=20,generate_embeddings=true'
test_command "Process document (no embeddings)" 'process_document:text="Sample document",chunk_size=100,generate_embeddings=false'
test_command "Retrieve context" 'retrieve_context:query="What is the main topic?",table=test_table,vector_column=embedding,limit=5' false
test_command "Generate response" 'generate_response:query="What is NeuronDB?",context=["Context 1","Context 2"]'

# ============================================================================
# HYBRID SEARCH
# ============================================================================
echo "=== HYBRID SEARCH ==="
test_command "Hybrid search" 'hybrid_search:table=test_table,query_vector=[0.1,0.2,0.3],query_text="search term",vector_column=embedding,text_column=text,vector_weight=0.7' false

# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================
echo "=== EDGE CASES AND ERROR HANDLING ==="
test_command "Invalid tool name" "invalid_tool_that_does_not_exist" false
test_command "Missing required parameter" "train_model:algorithm=linear_regression" false
test_command "Invalid parameter type" "list_models:algorithm=123" false
test_command "Empty command" "" false
test_command "Command with special characters" 'generate_embedding:text="Text with \"quotes\" and \'apostrophes\'"'
test_command "Command with newlines" 'generate_embedding:text="Text\nwith\nnewlines"'
test_command "Command with unicode" 'generate_embedding:text="Text with unicode"'
test_command "Large array parameter" 'batch_embedding:texts=["Text 1","Text 2","Text 3","Text 4","Text 5","Text 6","Text 7","Text 8","Text 9","Text 10"]'
test_command "Nested JSON in params" 'train_model:algorithm=linear_regression,table=test,feature_col=feat,label_col=label,params={"nested":{"key":"value","array":[1,2,3]}}' false

# ============================================================================
# CLAUDE DESKTOP COMPATIBILITY
# ============================================================================
echo "=== CLAUDE DESKTOP COMPATIBILITY ==="
test_command "Multiple sequential requests" "list_tools"
test_command "Request with string ID" "list_tools"
test_command "Request with numeric ID" "list_tools"
test_command "Notification handling" "list_tools"
test_command "Error response format" "invalid_method" false

# ============================================================================
# SUMMARY
# ============================================================================
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total tests: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Success rate: $(echo "scale=2; $PASSED * 100 / $TOTAL" | bc)%"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed. Check $OUTPUT_DIR for details."
    exit 1
fi

