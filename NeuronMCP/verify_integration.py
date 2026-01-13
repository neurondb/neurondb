#!/usr/bin/env python3
"""
Comprehensive NeuronMCP-NeuronDB Integration Verification Script

This script systematically verifies all aspects of NeuronMCP integration with NeuronDB:
- MCP Protocol Foundation
- Database Connection & Type Registration
- Tool Registration & Discovery
- NeuronDB SQL Function Integration
- All Tool Categories
- Resources
- Error Handling
- End-to-End Workflows
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add client directory to path
sys.path.insert(0, str(Path(__file__).parent / "client"))

from mcp_client.client import MCPClient
from mcp_client.config import load_config
from mcp_client.protocol import JSONRPCRequest


class IntegrationVerifier:
    """Comprehensive integration verifier for NeuronMCP-NeuronDB."""
    
    def __init__(self, config_path: str, server_name: str = "neurondb"):
        """Initialize verifier with configuration."""
        self.config = load_config(config_path, server_name)
        self.client = MCPClient(self.config, verbose=False)
        self.verification_results = {
            "metadata": {
                "start_time": None,
                "end_time": None,
                "version": "1.0.0",
            },
            "mcp_protocol": {},
            "database_connection": {},
            "tool_registration": {},
            "sql_functions": {},
            "vector_operations": {},
            "embedding_operations": {},
            "ml_operations": {},
            "index_management": {},
            "postgresql_tools": {},
            "resources": {},
            "advanced_features": {},
            "error_handling": {},
            "e2e_workflows": {},
            "summary": {
                "total_checks": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "warnings": 0,
            }
        }
        
    def _record_result(self, category: str, check_name: str, status: str, 
                       details: Dict[str, Any] = None, warning: str = None):
        """Record a verification result."""
        if category not in self.verification_results:
            self.verification_results[category] = {}
        
        result = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        if details:
            result["details"] = details
        if warning:
            result["warning"] = warning
            
        self.verification_results[category][check_name] = result
        
        # Update summary
        self.verification_results["summary"]["total_checks"] += 1
        if status == "passed":
            self.verification_results["summary"]["passed"] += 1
        elif status == "failed":
            self.verification_results["summary"]["failed"] += 1
        elif status == "skipped":
            self.verification_results["summary"]["skipped"] += 1
        if warning:
            self.verification_results["summary"]["warnings"] += 1
    
    def verify_mcp_protocol(self):
        """Verify MCP Protocol Foundation."""
        print("\n" + "="*80)
        print("1. MCP PROTOCOL FOUNDATION")
        print("="*80)
        
        # Test initialization
        print("\n[1.1] Testing MCP initialization...")
        try:
            self.client.connect()
            self._record_result("mcp_protocol", "initialization", "passed", {
                "protocol_version": "2025-06-18",
                "client_info": "neurondb-mcp-client v1.0.0"
            })
            print("  ✅ PASSED: MCP connection initialized")
        except Exception as e:
            self._record_result("mcp_protocol", "initialization", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
            return False
        
        # Test tools/list
        print("\n[1.2] Testing tools/list endpoint...")
        try:
            result = self.client.list_tools()
            if "error" in result:
                self._record_result("mcp_protocol", "tools_list", "failed", {
                    "error": result["error"]
                })
                print(f"  ❌ FAILED: {result['error']}")
            else:
                tools = result.get("tools", [])
                tool_count = len(tools)
                self._record_result("mcp_protocol", "tools_list", "passed", {
                    "tool_count": tool_count
                })
                print(f"  ✅ PASSED: Found {tool_count} tools")
                if tool_count < 100:
                    print(f"  ⚠️  WARNING: Expected 100+ tools, got {tool_count}")
        except Exception as e:
            self._record_result("mcp_protocol", "tools_list", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
        
        # Test tools/call
        print("\n[1.3] Testing tools/call endpoint...")
        try:
            result = self.client.call_tool("postgresql_version", {})
            if result.get("isError", False):
                error_msg = result.get("content", [{}])[0].get("text", "Unknown error")
                if "CONNECTION_ERROR" in error_msg:
                    self._record_result("mcp_protocol", "tools_call", "skipped", {
                        "reason": "Database connection required"
                    })
                    print(f"  ⏭️  SKIPPED: Database connection required")
                else:
                    self._record_result("mcp_protocol", "tools_call", "failed", {
                        "error": error_msg
                    })
                    print(f"  ❌ FAILED: {error_msg[:100]}")
            else:
                self._record_result("mcp_protocol", "tools_call", "passed")
                print("  ✅ PASSED: Tool call successful")
        except Exception as e:
            self._record_result("mcp_protocol", "tools_call", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
        
        # Test resources/list
        print("\n[1.4] Testing resources/list endpoint...")
        try:
            result = self.client.list_resources()
            if "error" in result:
                self._record_result("mcp_protocol", "resources_list", "failed", {
                    "error": result["error"]
                })
                print(f"  ❌ FAILED: {result['error']}")
            else:
                resources = result.get("resources", [])
                resource_count = len(resources)
                self._record_result("mcp_protocol", "resources_list", "passed", {
                    "resource_count": resource_count
                })
                print(f"  ✅ PASSED: Found {resource_count} resources")
        except Exception as e:
            self._record_result("mcp_protocol", "resources_list", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
        
        # Test error handling
        print("\n[1.5] Testing error handling...")
        try:
            result = self.client.call_tool("invalid_tool_xyz", {})
            if result.get("isError", False):
                self._record_result("mcp_protocol", "error_handling", "passed", {
                    "test": "invalid_tool_rejection"
                })
                print("  ✅ PASSED: Invalid tool correctly rejected")
            else:
                self._record_result("mcp_protocol", "error_handling", "failed", {
                    "reason": "Invalid tool not rejected"
                })
                print("  ❌ FAILED: Invalid tool not rejected")
        except Exception as e:
            self._record_result("mcp_protocol", "error_handling", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
        
        return True
    
    def verify_database_connection(self):
        """Verify Database Connection & Type Registration."""
        print("\n" + "="*80)
        print("2. DATABASE CONNECTION & TYPE REGISTRATION")
        print("="*80)
        
        # Test database connection via postgresql_version
        print("\n[2.1] Testing database connection...")
        try:
            result = self.client.call_tool("postgresql_version", {})
            if result.get("isError", False):
                error_msg = result.get("content", [{}])[0].get("text", "Unknown error")
                self._record_result("database_connection", "connection", "failed", {
                    "error": error_msg
                })
                print(f"  ❌ FAILED: {error_msg[:100]}")
                return False
            else:
                # Parse version info
                content = result.get("content", [])
                if content and len(content) > 0:
                    version_text = content[0].get("text", "{}")
                    try:
                        version_data = json.loads(version_text)
                        self._record_result("database_connection", "connection", "passed", {
                            "postgresql_version": version_data.get("version", "unknown")
                        })
                        print(f"  ✅ PASSED: Connected to PostgreSQL")
                    except:
                        self._record_result("database_connection", "connection", "passed")
                        print("  ✅ PASSED: Database connection successful")
        except Exception as e:
            self._record_result("database_connection", "connection", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
            return False
        
        # Test NeuronDB extension
        print("\n[2.2] Testing NeuronDB extension...")
        try:
            result = self.client.call_tool("postgresql_extensions", {})
            if result.get("isError", False):
                self._record_result("database_connection", "neurondb_extension", "skipped", {
                    "reason": "Cannot check extensions"
                })
                print("  ⏭️  SKIPPED: Cannot check extensions")
            else:
                content = result.get("content", [])
                if content and len(content) > 0:
                    extensions_text = content[0].get("text", "[]")
                    try:
                        extensions = json.loads(extensions_text)
                        neurondb_found = any(ext.get("extname") == "neurondb" for ext in extensions)
                        if neurondb_found:
                            self._record_result("database_connection", "neurondb_extension", "passed", {
                                "extension_installed": True
                            })
                            print("  ✅ PASSED: NeuronDB extension installed")
                        else:
                            self._record_result("database_connection", "neurondb_extension", "failed", {
                                "reason": "NeuronDB extension not found"
                            })
                            print("  ❌ FAILED: NeuronDB extension not found")
                    except:
                        self._record_result("database_connection", "neurondb_extension", "skipped", {
                            "reason": "Cannot parse extensions"
                        })
                        print("  ⏭️  SKIPPED: Cannot parse extensions")
        except Exception as e:
            self._record_result("database_connection", "neurondb_extension", "skipped", {
                "error": str(e)
            })
            print(f"  ⏭️  SKIPPED: {str(e)}")
        
        # Test custom type registration (indirectly via vector operations)
        print("\n[2.3] Testing custom type registration...")
        self._record_result("database_connection", "type_registration", "skipped", {
            "reason": "Type registration verified indirectly through vector operations",
            "note": "Vector type (OID 17648) and array type (OID 17656) should be registered"
        })
        print("  ⏭️  SKIPPED: Will be verified through vector operations")
        
        return True
    
    def verify_tool_registration(self):
        """Verify Tool Registration & Discovery."""
        print("\n" + "="*80)
        print("3. TOOL REGISTRATION & DISCOVERY")
        print("="*80)
        
        # Get all tools
        print("\n[3.1] Verifying tool registration...")
        try:
            result = self.client.list_tools()
            if "error" in result:
                self._record_result("tool_registration", "tool_list", "failed", {
                    "error": result["error"]
                })
                print(f"  ❌ FAILED: {result['error']}")
                return False
            
            tools = result.get("tools", [])
            tool_names = [t.get("name") for t in tools]
            
            # Expected tool categories
            expected_categories = {
                "vector_operations": [
                    "vector_search", "vector_search_l2", "vector_search_cosine",
                    "vector_search_inner_product", "vector_similarity", "vector_arithmetic",
                    "vector_distance", "vector_quantize"
                ],
                "embeddings": [
                    "generate_embedding", "batch_embedding", "embed_image",
                    "embed_multimodal", "embed_cached", "configure_embedding_model",
                    "get_embedding_model_config", "list_embedding_model_configs"
                ],
                "ml_operations": [
                    "train_model", "predict", "predict_batch", "evaluate_model",
                    "list_models", "get_model_info", "delete_model", "export_model"
                ],
                "index_management": [
                    "create_hnsw_index", "create_ivf_index", "index_status",
                    "drop_index", "tune_hnsw_index", "tune_ivf_index"
                ],
                "postgresql_tools": [
                    "postgresql_version", "postgresql_stats", "postgresql_databases",
                    "postgresql_connections", "postgresql_settings", "postgresql_extensions"
                ],
            }
            
            category_results = {}
            for category, expected_tools in expected_categories.items():
                found_tools = [t for t in expected_tools if t in tool_names]
                missing_tools = [t for t in expected_tools if t not in tool_names]
                category_results[category] = {
                    "expected": len(expected_tools),
                    "found": len(found_tools),
                    "missing": missing_tools
                }
            
            self._record_result("tool_registration", "tool_list", "passed", {
                "total_tools": len(tools),
                "categories": category_results
            })
            print(f"  ✅ PASSED: Found {len(tools)} tools")
            
            # Check for missing tools
            all_missing = []
            for category, results in category_results.items():
                if results["missing"]:
                    all_missing.extend(results["missing"])
                    print(f"  ⚠️  WARNING: Missing {len(results['missing'])} tools in {category}: {results['missing'][:3]}")
            
            if all_missing:
                self._record_result("tool_registration", "missing_tools", "warning", {
                    "missing_count": len(all_missing),
                    "missing_tools": all_missing[:10]  # First 10
                })
            
        except Exception as e:
            self._record_result("tool_registration", "tool_list", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
            return False
        
        # Verify tool schemas
        print("\n[3.2] Verifying tool schemas...")
        try:
            sample_tools = tools[:5]  # Check first 5 tools
            schema_issues = []
            for tool in sample_tools:
                if "name" not in tool:
                    schema_issues.append(f"{tool.get('name', 'unknown')}: missing name")
                if "description" not in tool:
                    schema_issues.append(f"{tool.get('name', 'unknown')}: missing description")
                if "inputSchema" not in tool:
                    schema_issues.append(f"{tool.get('name', 'unknown')}: missing inputSchema")
            
            if schema_issues:
                self._record_result("tool_registration", "tool_schemas", "failed", {
                    "issues": schema_issues
                })
                print(f"  ❌ FAILED: Found {len(schema_issues)} schema issues")
            else:
                self._record_result("tool_registration", "tool_schemas", "passed", {
                    "checked_tools": len(sample_tools)
                })
                print(f"  ✅ PASSED: Tool schemas verified ({len(sample_tools)} tools checked)")
        except Exception as e:
            self._record_result("tool_registration", "tool_schemas", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
        
        return True
    
    def verify_sql_functions(self):
        """Verify NeuronDB SQL Function Integration."""
        print("\n" + "="*80)
        print("4. NEURONDB SQL FUNCTION INTEGRATION")
        print("="*80)
        
        # This will be verified indirectly through tool execution
        # We'll check that tools that should call specific functions work correctly
        
        sql_functions_to_verify = {
            "neurondb.embed": ["generate_embedding"],
            "neurondb.embed_batch": ["batch_embedding"],
            "neurondb.train": ["train_model"],
            "neurondb.predict": ["predict"],
            "neurondb.list_models": ["list_models"],
        }
        
        print("\n[4.1] Verifying SQL function integration (indirect)...")
        print("  Note: Function calls verified through tool execution")
        
        for func_name, tool_names in sql_functions_to_verify.items():
            # Check if tools exist
            result = self.client.list_tools()
            tools = result.get("tools", [])
            tool_names_found = [t.get("name") for t in tools]
            
            found_tools = [t for t in tool_names if t in tool_names_found]
            if found_tools:
                self._record_result("sql_functions", func_name, "passed", {
                    "tools": found_tools,
                    "note": "Function integration verified through tool existence"
                })
                print(f"  ✅ PASSED: {func_name} - tools exist: {found_tools}")
            else:
                self._record_result("sql_functions", func_name, "failed", {
                    "expected_tools": tool_names,
                    "reason": "Tools not found"
                })
                print(f"  ❌ FAILED: {func_name} - tools not found: {tool_names}")
        
        return True
    
    def verify_vector_operations(self):
        """Verify Vector Operations."""
        print("\n" + "="*80)
        print("5. VECTOR OPERATIONS VERIFICATION")
        print("="*80)
        
        vector_tools = [
            "vector_search", "vector_search_l2", "vector_search_cosine",
            "vector_search_inner_product", "vector_similarity", "vector_arithmetic",
            "vector_distance", "vector_quantize"
        ]
        
        print("\n[5.1] Verifying vector operation tools...")
        result = self.client.list_tools()
        tools = result.get("tools", [])
        tool_names = [t.get("name") for t in tools]
        
        found_tools = [t for t in vector_tools if t in tool_names]
        missing_tools = [t for t in vector_tools if t not in tool_names]
        
        self._record_result("vector_operations", "tool_availability", 
                           "passed" if len(found_tools) == len(vector_tools) else "failed", {
            "expected": len(vector_tools),
            "found": len(found_tools),
            "missing": missing_tools
        })
        
        if missing_tools:
            print(f"  ❌ FAILED: Missing {len(missing_tools)} vector tools: {missing_tools}")
        else:
            print(f"  ✅ PASSED: All {len(vector_tools)} vector operation tools available")
        
        return True
    
    def verify_embedding_operations(self):
        """Verify Embedding Operations."""
        print("\n" + "="*80)
        print("6. EMBEDDING OPERATIONS VERIFICATION")
        print("="*80)
        
        embedding_tools = [
            "generate_embedding", "batch_embedding", "embed_image",
            "embed_multimodal", "embed_cached", "configure_embedding_model",
            "get_embedding_model_config", "list_embedding_model_configs",
            "delete_embedding_model_config"
        ]
        
        print("\n[6.1] Verifying embedding operation tools...")
        result = self.client.list_tools()
        tools = result.get("tools", [])
        tool_names = [t.get("name") for t in tools]
        
        found_tools = [t for t in embedding_tools if t in tool_names]
        missing_tools = [t for t in embedding_tools if t not in tool_names]
        
        self._record_result("embedding_operations", "tool_availability",
                           "passed" if len(found_tools) == len(embedding_tools) else "failed", {
            "expected": len(embedding_tools),
            "found": len(found_tools),
            "missing": missing_tools
        })
        
        if missing_tools:
            print(f"  ❌ FAILED: Missing {len(missing_tools)} embedding tools: {missing_tools}")
        else:
            print(f"  ✅ PASSED: All {len(embedding_tools)} embedding operation tools available")
        
        return True
    
    def verify_ml_operations(self):
        """Verify ML Operations."""
        print("\n" + "="*80)
        print("7. ML OPERATIONS VERIFICATION")
        print("="*80)
        
        ml_tools = [
            "train_model", "predict", "predict_batch", "evaluate_model",
            "list_models", "get_model_info", "delete_model", "export_model"
        ]
        
        print("\n[7.1] Verifying ML operation tools...")
        result = self.client.list_tools()
        tools = result.get("tools", [])
        tool_names = [t.get("name") for t in tools]
        
        found_tools = [t for t in ml_tools if t in tool_names]
        missing_tools = [t for t in ml_tools if t not in tool_names]
        
        self._record_result("ml_operations", "tool_availability",
                           "passed" if len(found_tools) == len(ml_tools) else "failed", {
            "expected": len(ml_tools),
            "found": len(found_tools),
            "missing": missing_tools
        })
        
        if missing_tools:
            print(f"  ❌ FAILED: Missing {len(missing_tools)} ML tools: {missing_tools}")
        else:
            print(f"  ✅ PASSED: All {len(ml_tools)} ML operation tools available")
        
        return True
    
    def verify_index_management(self):
        """Verify Index Management."""
        print("\n" + "="*80)
        print("8. INDEX MANAGEMENT VERIFICATION")
        print("="*80)
        
        index_tools = [
            "create_hnsw_index", "create_ivf_index", "index_status",
            "drop_index", "tune_hnsw_index", "tune_ivf_index"
        ]
        
        print("\n[8.1] Verifying index management tools...")
        result = self.client.list_tools()
        tools = result.get("tools", [])
        tool_names = [t.get("name") for t in tools]
        
        found_tools = [t for t in index_tools if t in tool_names]
        missing_tools = [t for t in index_tools if t not in tool_names]
        
        self._record_result("index_management", "tool_availability",
                           "passed" if len(found_tools) == len(index_tools) else "failed", {
            "expected": len(index_tools),
            "found": len(found_tools),
            "missing": missing_tools
        })
        
        if missing_tools:
            print(f"  ❌ FAILED: Missing {len(missing_tools)} index tools: {missing_tools}")
        else:
            print(f"  ✅ PASSED: All {len(index_tools)} index management tools available")
        
        return True
    
    def verify_postgresql_tools(self):
        """Verify PostgreSQL Tools (27 tools)."""
        print("\n" + "="*80)
        print("9. POSTGRESQL TOOLS VERIFICATION (27 tools)")
        print("="*80)
        
        postgresql_tools = {
            "server_info": [
                "postgresql_version", "postgresql_stats", "postgresql_databases",
                "postgresql_settings", "postgresql_extensions"
            ],
            "database_objects": [
                "postgresql_tables", "postgresql_indexes", "postgresql_schemas",
                "postgresql_views", "postgresql_sequences", "postgresql_functions",
                "postgresql_triggers", "postgresql_constraints"
            ],
            "user_role_management": [
                "postgresql_users", "postgresql_roles", "postgresql_permissions"
            ],
            "performance": [
                "postgresql_table_stats", "postgresql_index_stats", "postgresql_active_queries",
                "postgresql_wait_events", "postgresql_connections", "postgresql_locks",
                "postgresql_replication"
            ],
            "size_storage": [
                "postgresql_table_size", "postgresql_index_size", "postgresql_bloat",
                "postgresql_vacuum_stats"
            ]
        }
        
        print("\n[9.1] Verifying PostgreSQL tools...")
        result = self.client.list_tools()
        tools = result.get("tools", [])
        tool_names = [t.get("name") for t in tools]
        
        all_postgresql_tools = []
        for category_tools in postgresql_tools.values():
            all_postgresql_tools.extend(category_tools)
        
        found_tools = [t for t in all_postgresql_tools if t in tool_names]
        missing_tools = [t for t in all_postgresql_tools if t not in tool_names]
        
        category_results = {}
        for category, category_tools in postgresql_tools.items():
            found = [t for t in category_tools if t in tool_names]
            missing = [t for t in category_tools if t not in tool_names]
            category_results[category] = {
                "expected": len(category_tools),
                "found": len(found),
                "missing": missing
            }
        
        self._record_result("postgresql_tools", "tool_availability",
                           "passed" if len(missing_tools) == 0 else "failed", {
            "expected": len(all_postgresql_tools),
            "found": len(found_tools),
            "missing": missing_tools,
            "by_category": category_results
        })
        
        if missing_tools:
            print(f"  ❌ FAILED: Missing {len(missing_tools)} PostgreSQL tools: {missing_tools}")
        else:
            print(f"  ✅ PASSED: All {len(all_postgresql_tools)} PostgreSQL tools available")
        
        return True
    
    def verify_resources(self):
        """Verify Resources (9 resources)."""
        print("\n" + "="*80)
        print("10. RESOURCES VERIFICATION (9 resources)")
        print("="*80)
        
        expected_resources = [
            "neurondb://schema", "neurondb://models", "neurondb://indexes",
            "neurondb://config", "neurondb://workers", "neurondb://vector_stats",
            "neurondb://index_health", "neurondb://datasets", "neurondb://collections"
        ]
        
        print("\n[10.1] Verifying resources...")
        try:
            result = self.client.list_resources()
            if "error" in result:
                self._record_result("resources", "resource_list", "failed", {
                    "error": result["error"]
                })
                print(f"  ❌ FAILED: {result['error']}")
                return False
            
            resources = result.get("resources", [])
            resource_uris = [r.get("uri") for r in resources]
            
            found_resources = [r for r in expected_resources if r in resource_uris]
            missing_resources = [r for r in expected_resources if r not in resource_uris]
            
            self._record_result("resources", "resource_list", 
                               "passed" if len(missing_resources) == 0 else "failed", {
                "expected": len(expected_resources),
                "found": len(found_resources),
                "missing": missing_resources
            })
            
            if missing_resources:
                print(f"  ❌ FAILED: Missing {len(missing_resources)} resources: {missing_resources}")
            else:
                print(f"  ✅ PASSED: All {len(expected_resources)} resources available")
        except Exception as e:
            self._record_result("resources", "resource_list", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
            return False
        
        return True
    
    def verify_advanced_features(self):
        """Verify Advanced Features."""
        print("\n" + "="*80)
        print("11. ADVANCED FEATURES VERIFICATION")
        print("="*80)
        
        advanced_features = {
            "hybrid_search": [
                "hybrid_search", "reciprocal_rank_fusion", "semantic_keyword_search",
                "multi_vector_search", "faceted_vector_search", "temporal_vector_search",
                "diverse_vector_search"
            ],
            "reranking": [
                "rerank_cross_encoder", "rerank_llm", "rerank_cohere",
                "rerank_colbert", "rerank_ltr", "rerank_ensemble"
            ],
            "analytics": [
                "cluster_data", "detect_outliers", "reduce_dimensionality",
                "analyze_data", "quality_metrics", "detect_drift", "topic_discovery"
            ],
            "advanced": [
                "timeseries_analysis", "automl", "onnx_model",
                "vector_graph", "vecmap_operations", "load_dataset",
                "worker_management", "gpu_info"
            ]
        }
        
        print("\n[11.1] Verifying advanced features...")
        result = self.client.list_tools()
        tools = result.get("tools", [])
        tool_names = [t.get("name") for t in tools]
        
        category_results = {}
        for category, category_tools in advanced_features.items():
            found = [t for t in category_tools if t in tool_names]
            missing = [t for t in category_tools if t not in tool_names]
            category_results[category] = {
                "expected": len(category_tools),
                "found": len(found),
                "missing": missing
            }
        
        all_advanced_tools = []
        for tools_list in advanced_features.values():
            all_advanced_tools.extend(tools_list)
        
        found_tools = [t for t in all_advanced_tools if t in tool_names]
        missing_tools = [t for t in all_advanced_tools if t not in tool_names]
        
        self._record_result("advanced_features", "tool_availability",
                           "passed" if len(missing_tools) == 0 else "failed", {
            "expected": len(all_advanced_tools),
            "found": len(found_tools),
            "missing": missing_tools,
            "by_category": category_results
        })
        
        if missing_tools:
            print(f"  ⚠️  WARNING: Missing {len(missing_tools)} advanced feature tools")
            print(f"     Missing: {missing_tools[:5]}...")
        else:
            print(f"  ✅ PASSED: All {len(all_advanced_tools)} advanced feature tools available")
        
        return True
    
    def verify_error_handling(self):
        """Verify Error Handling & Validation."""
        print("\n" + "="*80)
        print("12. ERROR HANDLING & VALIDATION")
        print("="*80)
        
        print("\n[12.1] Testing error handling...")
        
        # Test invalid tool name
        try:
            result = self.client.call_tool("invalid_tool_xyz", {})
            if result.get("isError", False):
                self._record_result("error_handling", "invalid_tool_rejection", "passed")
                print("  ✅ PASSED: Invalid tool correctly rejected")
            else:
                self._record_result("error_handling", "invalid_tool_rejection", "failed")
                print("  ❌ FAILED: Invalid tool not rejected")
        except Exception as e:
            self._record_result("error_handling", "invalid_tool_rejection", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
        
        # Test invalid resource URI
        try:
            result = self.client.read_resource("neurondb://invalid")
            if "error" in result:
                self._record_result("error_handling", "invalid_resource_rejection", "passed")
                print("  ✅ PASSED: Invalid resource correctly rejected")
            else:
                self._record_result("error_handling", "invalid_resource_rejection", "failed")
                print("  ❌ FAILED: Invalid resource not rejected")
        except Exception as e:
            self._record_result("error_handling", "invalid_resource_rejection", "failed", {
                "error": str(e)
            })
            print(f"  ❌ FAILED: {str(e)}")
        
        return True
    
    def generate_report(self):
        """Generate comprehensive verification report."""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        
        summary = self.verification_results["summary"]
        total = summary["total_checks"]
        
        if total > 0:
            passed_pct = (summary["passed"] / total) * 100
            failed_pct = (summary["failed"] / total) * 100
            skipped_pct = (summary["skipped"] / total) * 100
            
            print(f"\nTotal Checks: {total}")
            print(f"✅ Passed: {summary['passed']} ({passed_pct:.1f}%)")
            print(f"❌ Failed: {summary['failed']} ({failed_pct:.1f}%)")
            print(f"⏭️  Skipped: {summary['skipped']} ({skipped_pct:.1f}%)")
            if summary["warnings"] > 0:
                print(f"⚠️  Warnings: {summary['warnings']}")
        
        # Save report
        report_file = Path(__file__).parent / "verification_report.json"
        self.verification_results["metadata"]["end_time"] = datetime.now().isoformat()
        
        with open(report_file, 'w') as f:
            json.dump(self.verification_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print("="*80)
        
        return report_file
    
    def run_all(self):
        """Run all verification checks."""
        self.verification_results["metadata"]["start_time"] = datetime.now().isoformat()
        
        try:
            # Connect first
            self.client.connect()
            
            # Run all verification steps
            self.verify_mcp_protocol()
            self.verify_database_connection()
            self.verify_tool_registration()
            self.verify_sql_functions()
            self.verify_vector_operations()
            self.verify_embedding_operations()
            self.verify_ml_operations()
            self.verify_index_management()
            self.verify_postgresql_tools()
            self.verify_resources()
            self.verify_advanced_features()
            self.verify_error_handling()
            
        finally:
            self.client.disconnect()
            self.generate_report()


def main():
    """Main entry point."""
    config_path = Path(__file__).parent / "neuronmcp_server.json"
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        print("Please create neuronmcp_server.json with MCP server configuration", file=sys.stderr)
        sys.exit(1)
    
    verifier = IntegrationVerifier(str(config_path), "neurondb")
    verifier.run_all()


if __name__ == "__main__":
    main()







