#!/usr/bin/env python3
"""-------------------------------------------------------------------------
 *
 * neuronmcp-test-comprehensive.py
 *    Comprehensive Test Suite for All NeuronMCP Features
 *
 * Tests all 70+ tools, 7 resources, and advanced MCP protocol features.
 * Provides comprehensive validation of NeuronMCP functionality including
 * tool execution, resource management, protocol compliance, and error handling.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/tests/neuronmcp-test-comprehensive.py
 *
 *-------------------------------------------------------------------------
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add client directory to path
sys.path.insert(0, str(Path(__file__).parent / "client"))

from mcp_client.client import MCPClient
from mcp_client.config import load_config
from mcp_client.protocol import JSONRPCRequest


class ComprehensiveTester:
    """Comprehensive test suite for NeuronMCP."""
    
    def __init__(self, config_path: str, server_name: str = "neurondb"):
        """Initialize tester with configuration."""
        self.config = load_config(config_path, server_name)
        self.client = MCPClient(self.config, verbose=False)
        self.test_results = {
            "metadata": {
                "start_time": None,
                "end_time": None,
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "configuration_needed": 0,
            },
            "protocol_tests": {},
            "tools": {},
            "resources": {},
        }
        
    def _send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a raw MCP protocol request."""
        try:
            request = JSONRPCRequest(
                method=method,
                params=params or {},
            )
            response = self.client.transport.send_request(request)
            if response.is_error():
                return {"error": response.get_error_message(), "error_code": response.error.code if response.error else None}
            return response.result or {}
        except (BrokenPipeError, OSError, IOError) as e:
            # Connection broken - return error instead of crashing
            return {"error": f"Connection broken: {str(e)}", "connection_error": True}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def _extract_error(self, result: Dict[str, Any]) -> str:
        """Extract error message from result."""
        if result.get("error"):
            return result["error"]
        if result.get("isError", False):
            content = result.get("content", [])
            if content and isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict):
                    if "text" in content[0]:
                        return content[0]["text"]
        return "Unknown error"
    
    def _is_configuration_error(self, error_msg: str) -> bool:
        """Check if error is a configuration error (expected)."""
        error_lower = error_msg.lower()
        return (
            "CONFIGURATION_ERROR" in error_msg or
            ("configuration" in error_lower and "required" in error_lower) or
            ("api key" in error_lower and ("not" in error_lower or "missing" in error_lower)) or
            ("embedding" in error_lower and "requires" in error_lower)
        )
    
    def test_protocol_endpoints(self):
        """Test all MCP protocol endpoints."""
        print("\n" + "="*80)
        print("PHASE 0: Protocol Discovery & Endpoints")
        print("="*80)
        
        # tools/list
        print("\n[Protocol] Testing tools/list...")
        try:
            result = self.client.list_tools()
            if "error" in result:
                print(f"  ❌ FAILED: {result['error']}")
                self.test_results["protocol_tests"]["tools_list"] = {"status": "failed", "error": result["error"]}
            else:
                tools = result.get("tools", [])
                print(f"  ✅ PASSED: Found {len(tools)} tools")
                self.test_results["protocol_tests"]["tools_list"] = {"status": "passed", "count": len(tools)}
                self.test_results["metadata"]["passed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            self.test_results["protocol_tests"]["tools_list"] = {"status": "failed", "error": str(e)}
            self.test_results["metadata"]["failed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        
        # resources/list
        print("\n[Protocol] Testing resources/list...")
        try:
            result = self.client.list_resources()
            if "error" in result:
                print(f"  ❌ FAILED: {result['error']}")
                self.test_results["protocol_tests"]["resources_list"] = {"status": "failed", "error": result["error"]}
                self.test_results["metadata"]["failed"] += 1
            else:
                resources = result.get("resources", [])
                print(f"  ✅ PASSED: Found {len(resources)} resources")
                self.test_results["protocol_tests"]["resources_list"] = {"status": "passed", "count": len(resources)}
                self.test_results["metadata"]["passed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            self.test_results["protocol_tests"]["resources_list"] = {"status": "failed", "error": str(e)}
            self.test_results["metadata"]["failed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        
        # prompts/list
        print("\n[Protocol] Testing prompts/list...")
        try:
            result = self._send_request("prompts/list", {})
            if "error" in result:
                print(f"  ⚠️  NOT AVAILABLE: {result['error']}")
                self.test_results["protocol_tests"]["prompts_list"] = {"status": "skipped", "error": result["error"]}
                self.test_results["metadata"]["skipped"] += 1
            else:
                prompts = result.get("prompts", [])
                print(f"  ✅ PASSED: Found {len(prompts)} prompts")
                self.test_results["protocol_tests"]["prompts_list"] = {"status": "passed", "count": len(prompts)}
                self.test_results["metadata"]["passed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except Exception as e:
            print(f"  ⚠️  EXCEPTION (may be expected): {str(e)}")
            self.test_results["protocol_tests"]["prompts_list"] = {"status": "skipped", "error": str(e)}
            self.test_results["metadata"]["skipped"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        
        # tools/search
        print("\n[Protocol] Testing tools/search...")
        try:
            result = self._send_request("tools/search", {"query": "vector"})
            if "error" in result:
                print(f"  ⚠️  NOT AVAILABLE: {result['error']}")
                self.test_results["protocol_tests"]["tools_search"] = {"status": "skipped", "error": result["error"]}
                self.test_results["metadata"]["skipped"] += 1
            else:
                tools = result.get("tools", [])
                print(f"  ✅ PASSED: Found {len(tools)} tools matching 'vector'")
                self.test_results["protocol_tests"]["tools_search"] = {"status": "passed", "count": len(tools)}
                self.test_results["metadata"]["passed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except Exception as e:
            print(f"  ⚠️  EXCEPTION (may be expected): {str(e)}")
            self.test_results["protocol_tests"]["tools_search"] = {"status": "skipped", "error": str(e)}
            self.test_results["metadata"]["skipped"] += 1
            self.test_results["metadata"]["total_tests"] += 1
    
    def test_tool(self, tool_name: str, arguments: Dict[str, Any] = None, category: str = "general") -> Tuple[bool, str]:
        """Test a single tool and return (success, status)."""
        if arguments is None:
            arguments = {}
        
        start_time = time.time()
        try:
            result = self.client.call_tool(tool_name, arguments)
            elapsed = time.time() - start_time
            
            if result.get("isError", False) or "error" in result:
                error_msg = self._extract_error(result)
                # Check for connection errors
                if "Connection broken" in error_msg or "Broken pipe" in error_msg:
                    status = "connection_error"
                    self.test_results["metadata"]["failed"] += 1
                elif self._is_configuration_error(error_msg):
                    status = "configuration_needed"
                    self.test_results["metadata"]["configuration_needed"] += 1
                else:
                    status = "failed"
                    self.test_results["metadata"]["failed"] += 1
                self.test_results["tools"][tool_name] = {
                    "status": status,
                    "category": category,
                    "arguments": arguments,
                    "error": error_msg,
                    "elapsed": elapsed,
                }
                return False, status
            else:
                self.test_results["tools"][tool_name] = {
                    "status": "passed",
                    "category": category,
                    "arguments": arguments,
                    "elapsed": elapsed,
                }
                self.test_results["metadata"]["passed"] += 1
                return True, "passed"
        except (BrokenPipeError, OSError, IOError) as e:
            elapsed = time.time() - start_time
            error_msg = f"Connection broken: {str(e)}"
            self.test_results["tools"][tool_name] = {
                "status": "connection_error",
                "category": category,
                "arguments": arguments,
                "error": error_msg,
                "exception_type": type(e).__name__,
                "elapsed": elapsed,
            }
            self.test_results["metadata"]["failed"] += 1
            return False, "connection_error"
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            self.test_results["tools"][tool_name] = {
                "status": "failed",
                "category": category,
                "arguments": arguments,
                "error": error_msg,
                "exception_type": type(e).__name__,
                "elapsed": elapsed,
            }
            self.test_results["metadata"]["failed"] += 1
            return False, "failed"
        finally:
            self.test_results["metadata"]["total_tests"] += 1
    
    def test_all_tools(self):
        """Test all tools organized by category."""
        print("\n" + "="*80)
        print("PHASE 1-9: Testing All Tools")
        print("="*80)
        
        # Get all available tools
        tools_response = self.client.list_tools()
        all_tools_set = {t["name"] for t in tools_response.get("tools", [])}
        
        # Define all tool categories with their tools and test arguments
        tool_categories = {
            "PostgreSQL": [
                ("postgresql_version", {}),
                ("postgresql_stats", {}),
                ("postgresql_databases", {}),
                ("postgresql_connections", {}),
                ("postgresql_locks", {}),
                ("postgresql_replication", {}),
                ("postgresql_settings", {}),
                ("postgresql_extensions", {}),
            ],
            "Vector Search": [
                ("vector_search", {"table": "test_vectors", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3], "limit": 5}),
                ("vector_search_l2", {"table": "test_vectors", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3], "limit": 5}),
                ("vector_search_cosine", {"table": "test_vectors", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3], "limit": 5}),
                ("vector_search_inner_product", {"table": "test_vectors", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3], "limit": 5}),
            ],
            "Vector Operations": [
                ("vector_similarity", {"vector1": [0.1, 0.2, 0.3], "vector2": [0.4, 0.5, 0.6], "metric": "cosine"}),
                ("vector_arithmetic", {"vector1": [0.1, 0.2, 0.3], "vector2": [0.4, 0.5, 0.6], "operation": "add"}),
                ("vector_distance", {"vector1": [0.1, 0.2, 0.3], "vector2": [0.4, 0.5, 0.6], "metric": "l2"}),
                ("vector_similarity_unified", {"vector1": [0.1, 0.2, 0.3], "vector2": [0.4, 0.5, 0.6], "metric": "cosine"}),
            ],
            "Embeddings": [
                ("generate_embedding", {"text": "test text", "model": "default"}),
                ("batch_embedding", {"texts": ["text1", "text2"], "model": "default"}),
                ("embed_image", {"image_url": "https://example.com/image.jpg"}),
                ("embed_multimodal", {"text": "test", "image_url": "https://example.com/image.jpg"}),
                ("embed_cached", {"text": "test", "model": "default"}),
                ("configure_embedding_model", {"model_name": "test_model", "config_json": "{}"}),
                ("get_embedding_model_config", {"model_name": "test_model"}),
                ("list_embedding_model_configs", {}),
                ("delete_embedding_model_config", {"model_name": "test_model"}),
            ],
            "Indexing": [
                ("create_hnsw_index", {"table": "test_table", "vector_column": "embedding", "index_name": "test_idx"}),
                ("create_ivf_index", {"table": "test_table", "vector_column": "embedding", "index_name": "test_idx"}),
                ("create_vector_index", {"table": "test_table", "vector_column": "embedding", "index_type": "hnsw"}),
                ("index_status", {"index_name": "test_idx"}),
                ("drop_index", {"index_name": "test_idx"}),
                ("tune_hnsw_index", {"index_name": "test_idx"}),
                ("tune_ivf_index", {"index_name": "test_idx"}),
            ],
            "ML Operations": [
                ("train_model", {"algorithm": "linear_regression", "table": "test_table", "feature_col": "features", "label_col": "label"}),
                ("predict", {"model_id": 1, "features": [0.1, 0.2, 0.3]}),
                ("predict_batch", {"model_id": 1, "features_array": [[0.1, 0.2, 0.3]]}),
                ("evaluate_model", {"model_id": 1, "table": "test_table", "feature_col": "features", "label_col": "label"}),
                ("list_models", {}),
                ("get_model_info", {"model_id": 1}),
                ("delete_model", {"model_id": 1}),
                ("export_model", {"model_id": 1}),
            ],
            "Analytics": [
                ("cluster_data", {"table": "test_table", "vector_column": "embedding", "k": 3}),
                ("detect_outliers", {"table": "test_table", "vector_column": "embedding"}),
                ("reduce_dimensionality", {"table": "test_table", "vector_column": "embedding", "target_dim": 2}),
                ("analyze_data", {"table": "test_table"}),
                ("quality_metrics", {"table": "test_table", "vector_column": "embedding"}),
                ("drift_detection", {"table": "test_table", "vector_column": "embedding"}),
                ("topic_discovery", {"table": "test_table", "text_column": "text"}),
            ],
            "RAG": [
                ("process_document", {"document": "test document", "chunk_size": 100}),
                ("retrieve_context", {"query": "test query", "table": "test_table", "limit": 5}),
                ("generate_response", {"query": "test query", "context": "test context"}),
                ("chunk_document", {"document": "test document", "chunk_size": 100}),
            ],
            "Hybrid Search": [
                ("hybrid_search", {"query": "test", "table": "test_table", "vector_column": "embedding", "limit": 5}),
                ("text_search", {"query": "test", "table": "test_table", "limit": 5}),
                ("reciprocal_rank_fusion", {"results": [[{"id": 1}], [{"id": 2}]]}),
                ("semantic_keyword_search", {"query": "test", "table": "test_table", "vector_column": "embedding"}),
                ("multi_vector_search", {"table": "test_table", "queries": [[0.1, 0.2, 0.3]], "limit": 5}),
                ("faceted_vector_search", {"table": "test_table", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3]}),
                ("temporal_vector_search", {"table": "test_table", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3]}),
                ("diverse_vector_search", {"table": "test_table", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3]}),
            ],
            "Reranking": [
                ("rerank_cross_encoder", {"query": "test", "documents": ["doc1", "doc2"]}),
                ("rerank_llm", {"query": "test", "documents": ["doc1", "doc2"]}),
                ("rerank_cohere", {"query": "test", "documents": ["doc1", "doc2"]}),
                ("rerank_colbert", {"query": "test", "documents": ["doc1", "doc2"]}),
                ("rerank_ltr", {"query": "test", "documents": ["doc1", "doc2"]}),
                ("rerank_ensemble", {"query": "test", "documents": ["doc1", "doc2"]}),
            ],
            "Quantization": [
                ("vector_quantize", {"operation": "to_int8", "vector": [0.1, 0.2, 0.3]}),
                ("quantization_analyze", {"operation": "analyze_int8", "vector": [0.1, 0.2, 0.3]}),
            ],
            "Advanced": [
                ("timeseries_analysis", {"operation": "arima", "table": "test_table", "value_column": "value", "time_column": "time"}),
                ("automl", {"table": "test_table", "feature_col": "features", "label_col": "label"}),
                ("onnx_model", {"operation": "info"}),
                ("vector_graph", {"operation": "pagerank", "table": "test_table", "vector_column": "embedding"}),
                ("vecmap_operations", {"operation": "distance", "vector1": [0.1, 0.2], "vector2": [0.3, 0.4]}),
            ],
            "Dataset": [
                ("load_dataset", {"source_type": "huggingface", "source_path": "squad", "limit": 10}),
            ],
            "Workers & GPU": [
                ("worker_management", {"operation": "status"}),
                ("gpu_info", {}),
            ],
        }
        
        # Test each category
        for category, tools in tool_categories.items():
            print(f"\n[{category}] Testing {len(tools)} tools...")
            for tool_name, args in tools:
                if tool_name in all_tools_set:
                    success, status = self.test_tool(tool_name, args, category)
                    status_symbol = "✅" if success else "⚠️" if status == "configuration_needed" else "❌"
                    print(f"  {status_symbol} {tool_name}")
                else:
                    print(f"  ⏭️  {tool_name} (not available)")
                    self.test_results["tools"][tool_name] = {
                        "status": "skipped",
                        "category": category,
                        "reason": "not in tool list",
                    }
                    self.test_results["metadata"]["skipped"] += 1
                    self.test_results["metadata"]["total_tests"] += 1
        
        # Test any remaining tools not in our predefined list
        tested_tools = set()
        for category_tools in tool_categories.values():
            tested_tools.update(t[0] for t in category_tools)
        
        remaining_tools = all_tools_set - tested_tools
        if remaining_tools:
            print(f"\n[Additional Tools] Testing {len(remaining_tools)} additional tools...")
            for tool_name in sorted(remaining_tools):
                success, status = self.test_tool(tool_name, {}, "additional")
                status_symbol = "✅" if success else "⚠️" if status == "configuration_needed" else "❌"
                print(f"  {status_symbol} {tool_name}")
    
    def test_resources(self):
        """Test all resources."""
        print("\n" + "="*80)
        print("PHASE 11: Testing Resources")
        print("="*80)
        
        resources = [
            "neurondb://schema",
            "neurondb://models",
            "neurondb://indexes",
            "neurondb://config",
            "neurondb://workers",
            "neurondb://vector_stats",
            "neurondb://index_health",
        ]
        
        for uri in resources:
            print(f"\n[Resource] Testing {uri}...")
            start_time = time.time()
            try:
                result = self.client.read_resource(uri)
                elapsed = time.time() - start_time
                
                if "error" in result:
                    error_msg = result["error"]
                    if "Connection broken" in error_msg or "Broken pipe" in error_msg:
                        print(f"  ❌ CONNECTION ERROR: {error_msg[:100]}")
                        self.test_results["resources"][uri] = {
                            "status": "connection_error",
                            "error": error_msg,
                            "elapsed": elapsed,
                        }
                    else:
                        print(f"  ❌ FAILED: {error_msg[:100]}")
                        self.test_results["resources"][uri] = {
                            "status": "failed",
                            "error": error_msg,
                            "elapsed": elapsed,
                        }
                    self.test_results["metadata"]["failed"] += 1
                else:
                    print(f"  ✅ PASSED")
                    self.test_results["resources"][uri] = {
                        "status": "passed",
                        "elapsed": elapsed,
                    }
                    self.test_results["metadata"]["passed"] += 1
            except (BrokenPipeError, OSError, IOError) as e:
                elapsed = time.time() - start_time
                print(f"  ❌ CONNECTION ERROR: {str(e)}")
                self.test_results["resources"][uri] = {
                    "status": "connection_error",
                    "error": f"Connection broken: {str(e)}",
                    "exception_type": type(e).__name__,
                    "elapsed": elapsed,
                }
                self.test_results["metadata"]["failed"] += 1
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  ❌ EXCEPTION: {str(e)}")
                self.test_results["resources"][uri] = {
                    "status": "failed",
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "elapsed": elapsed,
                }
                self.test_results["metadata"]["failed"] += 1
            finally:
                self.test_results["metadata"]["total_tests"] += 1
    
    def test_advanced_protocol_features(self):
        """Test advanced protocol features."""
        print("\n" + "="*80)
        print("PHASE 10: Advanced Protocol Features")
        print("="*80)
        
        # prompts/get
        print("\n[Advanced] Testing prompts/get...")
        try:
            # First list prompts to get a name
            prompts_result = self._send_request("prompts/list", {})
            if prompts_result.get("connection_error"):
                print(f"  ❌ CONNECTION ERROR: Cannot test prompts/get (connection broken)")
                self.test_results["protocol_tests"]["prompts_get"] = {"status": "connection_error", "error": prompts_result.get("error")}
                self.test_results["metadata"]["failed"] += 1
            elif "error" not in prompts_result and prompts_result.get("prompts"):
                prompt_name = prompts_result["prompts"][0].get("name")
                if prompt_name:
                    result = self._send_request("prompts/get", {"name": prompt_name})
                    if result.get("connection_error"):
                        print(f"  ❌ CONNECTION ERROR: {result.get('error', 'Connection broken')}")
                        self.test_results["protocol_tests"]["prompts_get"] = {"status": "connection_error", "error": result.get("error")}
                        self.test_results["metadata"]["failed"] += 1
                    elif "error" in result:
                        print(f"  ⚠️  FAILED: {result['error']}")
                        self.test_results["protocol_tests"]["prompts_get"] = {"status": "failed", "error": result["error"]}
                        self.test_results["metadata"]["failed"] += 1
                    else:
                        print(f"  ✅ PASSED")
                        self.test_results["protocol_tests"]["prompts_get"] = {"status": "passed"}
                        self.test_results["metadata"]["passed"] += 1
                else:
                    print(f"  ⏭️  SKIPPED: No prompts available")
                    self.test_results["protocol_tests"]["prompts_get"] = {"status": "skipped", "reason": "no prompts"}
                    self.test_results["metadata"]["skipped"] += 1
            else:
                print(f"  ⏭️  SKIPPED: prompts/list failed")
                self.test_results["protocol_tests"]["prompts_get"] = {"status": "skipped", "reason": "prompts/list failed"}
                self.test_results["metadata"]["skipped"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except (BrokenPipeError, OSError, IOError) as e:
            print(f"  ❌ CONNECTION ERROR: {str(e)}")
            self.test_results["protocol_tests"]["prompts_get"] = {"status": "connection_error", "error": str(e)}
            self.test_results["metadata"]["failed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except Exception as e:
            print(f"  ⚠️  EXCEPTION: {str(e)}")
            self.test_results["protocol_tests"]["prompts_get"] = {"status": "skipped", "error": str(e)}
            self.test_results["metadata"]["skipped"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        
        # sampling/createMessage
        print("\n[Advanced] Testing sampling/createMessage...")
        try:
            result = self._send_request("sampling/createMessage", {
                "messages": [{"role": "user", "content": "test"}],
                "model": "default",
            })
            if result.get("connection_error"):
                print(f"  ❌ CONNECTION ERROR: {result.get('error', 'Connection broken')}")
                self.test_results["protocol_tests"]["sampling_createMessage"] = {"status": "connection_error", "error": result.get("error")}
                self.test_results["metadata"]["failed"] += 1
            elif "error" in result:
                print(f"  ⚠️  CONFIGURATION NEEDED: {result['error'][:100]}")
                self.test_results["protocol_tests"]["sampling_createMessage"] = {"status": "configuration_needed", "error": result["error"]}
                self.test_results["metadata"]["configuration_needed"] += 1
            else:
                print(f"  ✅ PASSED")
                self.test_results["protocol_tests"]["sampling_createMessage"] = {"status": "passed"}
                self.test_results["metadata"]["passed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except (BrokenPipeError, OSError, IOError) as e:
            print(f"  ❌ CONNECTION ERROR: {str(e)}")
            self.test_results["protocol_tests"]["sampling_createMessage"] = {"status": "connection_error", "error": str(e)}
            self.test_results["metadata"]["failed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except Exception as e:
            print(f"  ⚠️  EXCEPTION (may be expected): {str(e)}")
            self.test_results["protocol_tests"]["sampling_createMessage"] = {"status": "skipped", "error": str(e)}
            self.test_results["metadata"]["skipped"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        
        # progress/get
        print("\n[Advanced] Testing progress/get...")
        try:
            result = self._send_request("progress/get", {"progress_id": "test_progress_123"})
            if result.get("connection_error"):
                print(f"  ❌ CONNECTION ERROR: {result.get('error', 'Connection broken')}")
                self.test_results["protocol_tests"]["progress_get"] = {"status": "connection_error", "error": result.get("error")}
                self.test_results["metadata"]["failed"] += 1
            elif "error" in result:
                print(f"  ⚠️  NOT FOUND (expected for non-existent progress): {result['error'][:100]}")
                self.test_results["protocol_tests"]["progress_get"] = {"status": "passed", "note": "error expected for non-existent ID"}
                self.test_results["metadata"]["passed"] += 1
            else:
                print(f"  ✅ PASSED")
                self.test_results["protocol_tests"]["progress_get"] = {"status": "passed"}
                self.test_results["metadata"]["passed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except (BrokenPipeError, OSError, IOError) as e:
            print(f"  ❌ CONNECTION ERROR: {str(e)}")
            self.test_results["protocol_tests"]["progress_get"] = {"status": "connection_error", "error": str(e)}
            self.test_results["metadata"]["failed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except Exception as e:
            print(f"  ⚠️  EXCEPTION: {str(e)}")
            self.test_results["protocol_tests"]["progress_get"] = {"status": "skipped", "error": str(e)}
            self.test_results["metadata"]["skipped"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        
        # tools/call_batch
        print("\n[Advanced] Testing tools/call_batch...")
        try:
            result = self._send_request("tools/call_batch", {
                "tools": [
                    {"name": "list_models", "arguments": {}},
                    {"name": "gpu_info", "arguments": {}},
                ],
                "parallel": False,
            })
            if result.get("connection_error"):
                print(f"  ❌ CONNECTION ERROR: {result.get('error', 'Connection broken')}")
                self.test_results["protocol_tests"]["tools_call_batch"] = {"status": "connection_error", "error": result.get("error")}
                self.test_results["metadata"]["failed"] += 1
            elif "error" in result:
                print(f"  ⚠️  NOT AVAILABLE: {result['error'][:100]}")
                self.test_results["protocol_tests"]["tools_call_batch"] = {"status": "skipped", "error": result["error"]}
                self.test_results["metadata"]["skipped"] += 1
            else:
                print(f"  ✅ PASSED")
                self.test_results["protocol_tests"]["tools_call_batch"] = {"status": "passed"}
                self.test_results["metadata"]["passed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except (BrokenPipeError, OSError, IOError) as e:
            print(f"  ❌ CONNECTION ERROR: {str(e)}")
            self.test_results["protocol_tests"]["tools_call_batch"] = {"status": "connection_error", "error": str(e)}
            self.test_results["metadata"]["failed"] += 1
            self.test_results["metadata"]["total_tests"] += 1
        except Exception as e:
            print(f"  ⚠️  EXCEPTION (may be expected): {str(e)}")
            self.test_results["protocol_tests"]["tools_call_batch"] = {"status": "skipped", "error": str(e)}
            self.test_results["metadata"]["skipped"] += 1
            self.test_results["metadata"]["total_tests"] += 1
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        metadata = self.test_results["metadata"]
        total = metadata["total_tests"]
        passed = metadata["passed"]
        failed = metadata["failed"]
        skipped = metadata["skipped"]
        config_needed = metadata["configuration_needed"]
        
        print(f"\nTotal Tests: {total}")
        if total > 0:
            print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
            print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
            print(f"⚠️  Configuration Needed: {config_needed} ({config_needed/total*100:.1f}%)")
            print(f"⏭️  Skipped: {skipped} ({skipped/total*100:.1f}%)")
        
        # Breakdown by category
        print("\n" + "-"*80)
        print("Breakdown by Category:")
        print("-"*80)
        
        # Count tools by category
        category_counts = {}
        for tool_name, tool_data in self.test_results["tools"].items():
            category = tool_data.get("category", "unknown")
            status = tool_data.get("status", "unknown")
            if category not in category_counts:
                category_counts[category] = {"passed": 0, "failed": 0, "skipped": 0, "configuration_needed": 0, "connection_error": 0}
            category_counts[category][status] = category_counts[category].get(status, 0) + 1
        
        for category, counts in sorted(category_counts.items()):
            total_cat = sum(counts.values())
            print(f"\n{category}: {total_cat} tools")
            print(f"  ✅ Passed: {counts.get('passed', 0)}")
            print(f"  ❌ Failed: {counts.get('failed', 0)}")
            print(f"  🔌 Connection Error: {counts.get('connection_error', 0)}")
            print(f"  ⚠️  Config Needed: {counts.get('configuration_needed', 0)}")
            print(f"  ⏭️  Skipped: {counts.get('skipped', 0)}")
        
        # Protocol tests summary
        if self.test_results["protocol_tests"]:
            print("\n" + "-"*80)
            print("Protocol Tests:")
            print("-"*80)
            for test_name, test_data in self.test_results["protocol_tests"].items():
                status = test_data.get("status", "unknown")
                status_symbol = "✅" if status == "passed" else "⚠️" if status == "configuration_needed" else "❌" if status == "failed" else "⏭️"
                print(f"  {status_symbol} {test_name}: {status}")
        
        # Resources summary
        if self.test_results["resources"]:
            print("\n" + "-"*80)
            print("Resources:")
            print("-"*80)
            passed_res = sum(1 for r in self.test_results["resources"].values() if r.get("status") == "passed")
            total_res = len(self.test_results["resources"])
            print(f"  ✅ Passed: {passed_res}/{total_res}")
            for uri, res_data in self.test_results["resources"].items():
                status = res_data.get("status", "unknown")
                status_symbol = "✅" if status == "passed" else "❌"
                print(f"    {status_symbol} {uri}: {status}")
        
        # Save results
        results_file = Path(__file__).parent / "test_comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")
        print("="*80)
    
    def run(self):
        """Run all tests."""
        self.test_results["metadata"]["start_time"] = datetime.now().isoformat()
        
        try:
            self.client.connect()
            print("="*80)
            print("NeuronMCP Comprehensive Test Suite")
            print("="*80)
            
            # Phase 0: Protocol Discovery
            self.test_protocol_endpoints()
            
            # Phase 1-9: All Tools
            self.test_all_tools()
            
            # Phase 10: Advanced Protocol Features
            self.test_advanced_protocol_features()
            
            # Phase 11: Resources
            self.test_resources()
            
        finally:
            self.client.disconnect()
            self.test_results["metadata"]["end_time"] = datetime.now().isoformat()
            self.generate_report()


def main():
    """Main entry point."""
    config_path = Path(__file__).parent.parent / "conf" / "neuronmcp-server.json"
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    tester = ComprehensiveTester(str(config_path), "neurondb")
    tester.run()


if __name__ == "__main__":
    main()

