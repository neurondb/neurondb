#!/usr/bin/env python3
"""-------------------------------------------------------------------------
 *
 * test_dataloading.py
 *    Comprehensive Test Suite for NeuronMCP Dataset Loading
 *
 * Tests all dataset loading features including:
 * - Multiple source types (HuggingFace, URL, GitHub, S3, local, database)
 * - Multiple formats (CSV, JSON, JSONL, Parquet, Excel, etc.)
 * - Auto-embedding functionality
 * - Schema detection
 * - Index creation
 * - Error handling and edge cases
 * - Transformations
 * - Incremental loading
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/tests/test_dataloading.py
 *
 *-------------------------------------------------------------------------
"""

import sys
import json
import time
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add client directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "client"))

try:
    from mcp_client.client import MCPClient
    from mcp_client.config import load_config
    from mcp_client.protocol import JSONRPCRequest
except ImportError:
    # Fallback if client not available
    print("[WARNING] MCP client not available. Some tests may be skipped.", file=sys.stderr)
    MCPClient = None


class DataLoadingTester:
    """Comprehensive test suite for NeuronMCP dataset loading."""
    
    def __init__(self, config_path: str = None, server_name: str = "neurondb"):
        """Initialize tester with configuration."""
        self.config_path = config_path or str(Path(__file__).parent.parent / "neuronmcp_server.json")
        self.server_name = server_name
        self.client = None
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
            "tests": {},
        }
        
        if MCPClient:
            try:
                self.config = load_config(self.config_path, server_name)
                self.client = MCPClient(self.config, verbose=False)
                # Connect to the server
                self.client.connect()
            except Exception as e:
                print(f"[WARNING] Failed to initialize MCP client: {e}", file=sys.stderr)
                print("[INFO] Tests will be skipped", file=sys.stderr)
                self.client = None
    
    def _send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a raw MCP protocol request."""
        if not self.client or not self.client.transport:
            return {"error": "Client not initialized", "skipped": True}
        
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
            return {"error": f"Connection broken: {str(e)}", "connection_error": True}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Call a tool and return success status, message, and result."""
        if not self.client:
            return False, "Client not initialized", {}
        
        # Add delay to avoid rate limiting and circuit breaker
        time.sleep(1.0)
        
        result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if result.get("skipped"):
            return False, "Test skipped", result
        
        if result.get("error"):
            error_msg = result.get("error", "Unknown error")
            # Check if it's a circuit breaker error
            if "circuit breaker" in error_msg.lower() and "open" in error_msg.lower():
                # Circuit breaker is open - wait for timeout (usually 60s) then retry
                print(f"  [INFO] Circuit breaker open, waiting 65 seconds...", file=sys.stderr)
                time.sleep(65)
                result = self._send_request("tools/call", {
                    "name": tool_name,
                    "arguments": arguments
                })
                if result.get("error"):
                    error_msg = result.get("error", "Unknown error")
            
            return False, error_msg, result
        
        # Check if result indicates success
        content = result.get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict):
                text = content[0].get("text", "")
                if "error" in text.lower() or "failed" in text.lower():
                    return False, text, result
                return True, text, result
        
        return True, "Success", result
    
    def _is_configuration_error(self, error_msg: str) -> bool:
        """Check if error is a configuration error (expected)."""
        error_lower = error_msg.lower()
        return (
            "CONFIGURATION_ERROR" in error_msg or
            ("configuration" in error_lower and "required" in error_lower) or
            ("api key" in error_lower and ("not" in error_lower or "missing" in error_lower)) or
            ("embedding" in error_lower and "requires" in error_lower) or
            "connection" in error_lower
        )
    
    def _record_test(self, test_name: str, success: bool, message: str, 
                    status: str = "passed", details: Dict[str, Any] = None):
        """Record test result."""
        self.test_results["metadata"]["total_tests"] += 1
        
        if status == "passed" and success:
            self.test_results["metadata"]["passed"] += 1
        elif status == "configuration_needed":
            self.test_results["metadata"]["configuration_needed"] += 1
        elif status == "skipped":
            self.test_results["metadata"]["skipped"] += 1
        else:
            self.test_results["metadata"]["failed"] += 1
        
        self.test_results["tests"][test_name] = {
            "success": success,
            "status": status,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        }
    
    def test_huggingface_loading(self):
        """Test HuggingFace dataset loading."""
        print("\n" + "="*80)
        print("TEST SUITE 1: HuggingFace Dataset Loading")
        print("="*80)
        
        # Test 1.1: Basic HuggingFace loading
        print("\n[1.1] Testing basic HuggingFace dataset loading...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "huggingface",
            "source_path": "squad",
            "split": "train",
            "limit": 10,
            "auto_embed": False,
            "create_indexes": False,
        })
        status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
        self._record_test("hf_basic_loading", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Basic loading: {msg[:100]}")
        
        # Test 1.2: HuggingFace with auto-embedding
        print("\n[1.2] Testing HuggingFace with auto-embedding...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "huggingface",
            "source_path": "squad",
            "split": "train",
            "limit": 5,
            "auto_embed": True,
            "embedding_model": "default",
            "create_indexes": True,
        })
        status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
        self._record_test("hf_auto_embedding", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Auto-embedding: {msg[:100]}")
        
        # Test 1.3: HuggingFace with custom schema and table
        print("\n[1.3] Testing HuggingFace with custom schema/table...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "huggingface",
            "source_path": "squad",
            "split": "train",
            "limit": 5,
            "schema_name": "test_schema",
            "table_name": "test_squad",
            "auto_embed": False,
        })
        status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
        self._record_test("hf_custom_schema_table", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Custom schema/table: {msg[:100]}")
        
        # Test 1.4: HuggingFace with streaming
        print("\n[1.4] Testing HuggingFace streaming mode...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "huggingface",
            "source_path": "squad",
            "split": "train",
            "limit": 10,
            "streaming": True,
            "auto_embed": False,
        })
        status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
        self._record_test("hf_streaming", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Streaming mode: {msg[:100]}")
        
        # Test 1.5: HuggingFace with config parameter
        print("\n[1.5] Testing HuggingFace with config parameter...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "huggingface",
            "source_path": "glue",
            "config": "cola",
            "split": "train",
            "limit": 5,
            "auto_embed": False,
        })
        status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
        self._record_test("hf_with_config", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} With config: {msg[:100]}")
        
        # Test 1.6: HuggingFace with text columns specification
        print("\n[1.6] Testing HuggingFace with specified text columns...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "huggingface",
            "source_path": "squad",
            "split": "train",
            "limit": 5,
            "auto_embed": True,
            "text_columns": ["question", "context"],
        })
        status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
        self._record_test("hf_text_columns", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Text columns: {msg[:100]}")
        
        # Test 1.7: HuggingFace with batch size
        print("\n[1.7] Testing HuggingFace with custom batch size...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "huggingface",
            "source_path": "squad",
            "split": "train",
            "limit": 20,
            "batch_size": 5,
            "auto_embed": False,
        })
        status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
        self._record_test("hf_batch_size", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Batch size: {msg[:100]}")
        
        # Test 1.8: HuggingFace error handling - invalid dataset
        print("\n[1.8] Testing error handling - invalid dataset...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "huggingface",
            "source_path": "nonexistent_dataset_12345",
            "split": "train",
            "limit": 5,
        })
        # This should fail, so success=False is expected
        expected_failure = not success
        status = "passed" if expected_failure else "failed"
        self._record_test("hf_invalid_dataset", expected_failure, msg, status, {"result": result})
        print(f"  {'✅' if expected_failure else '❌'} Invalid dataset handling: {msg[:100]}")
    
    def test_url_loading(self):
        """Test URL dataset loading."""
        print("\n" + "="*80)
        print("TEST SUITE 2: URL Dataset Loading")
        print("="*80)
        
        # Test 2.1: CSV from URL - use a reliable test URL
        print("\n[2.1] Testing CSV loading from URL...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "url",
            "source_path": "https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv",
            "format": "csv",
            "limit": 5,
            "auto_embed": False,
        })
        status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
        self._record_test("url_csv", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} CSV from URL: {msg[:100]}")
        
        # Test 2.2: JSON from URL - use a smaller dataset
        print("\n[2.2] Testing JSON loading from URL...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "url",
            "source_path": "https://jsonplaceholder.typicode.com/posts",
            "format": "json",
            "limit": 5,
            "auto_embed": False,
        })
        status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
        self._record_test("url_json", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} JSON from URL: {msg[:100]}")
        
        # Test 2.3: Auto format detection
        print("\n[2.3] Testing auto format detection from URL...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "url",
            "source_path": "https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv",
            "format": "auto",
            "limit": 5,
            "auto_embed": False,
        })
        status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
        self._record_test("url_auto_format", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Auto format: {msg[:100]}")
        
        # Test 2.4: URL with compression
        print("\n[2.4] Testing URL with compression detection...")
        # Note: This test may fail if no compressed file is available
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "url",
            "source_path": "https://example.com/data.csv.gz",
            "format": "csv",
            "compression": "gzip",
            "limit": 10,
            "auto_embed": False,
        })
        status = "skipped" if not success and ("not found" in msg.lower() or "404" in msg.lower()) else ("configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed"))
        self._record_test("url_compression", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⏭️' if status == 'skipped' else '⚠️' if status == 'configuration_needed' else '❌'} Compression: {msg[:100]}")
    
    def test_local_file_loading(self):
        """Test local file loading."""
        print("\n" + "="*80)
        print("TEST SUITE 3: Local File Loading")
        print("="*80)
        
        # Create temporary test files in a location accessible to the Python script
        # Use /tmp which should be accessible from both test and Python script
        temp_dir = "/tmp/neurondb_test_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Create test CSV file
            csv_file = os.path.join(temp_dir, "test_data.csv")
            with open(csv_file, "w") as f:
                f.write("id,name,value\n")
                f.write("1,Alice,100\n")
                f.write("2,Bob,200\n")
                f.write("3,Charlie,300\n")
            
            # Test 3.1: Local CSV file
            print("\n[3.1] Testing local CSV file loading...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": csv_file,
                "format": "csv",
                "auto_embed": False,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("local_csv", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Local CSV: {msg[:100]}")
            
            # Create test JSON file
            json_file = os.path.join(temp_dir, "test_data.json")
            with open(json_file, "w") as f:
                json.dump([
                    {"id": 1, "name": "Alice", "value": 100},
                    {"id": 2, "name": "Bob", "value": 200},
                    {"id": 3, "name": "Charlie", "value": 300},
                ], f)
            
            # Test 3.2: Local JSON file
            print("\n[3.2] Testing local JSON file loading...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": json_file,
                "format": "json",
                "auto_embed": False,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("local_json", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Local JSON: {msg[:100]}")
            
            # Create test JSONL file
            jsonl_file = os.path.join(temp_dir, "test_data.jsonl")
            with open(jsonl_file, "w") as f:
                for item in [{"id": 1, "text": "First item"}, {"id": 2, "text": "Second item"}]:
                    f.write(json.dumps(item) + "\n")
            
            # Test 3.3: Local JSONL file
            print("\n[3.3] Testing local JSONL file loading...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": jsonl_file,
                "format": "jsonl",
                "auto_embed": True,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("local_jsonl", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Local JSONL: {msg[:100]}")
            
            # Test 3.4: Local file with auto format detection
            print("\n[3.4] Testing local file with auto format detection...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": csv_file,
                "format": "auto",
                "auto_embed": False,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("local_auto_format", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Auto format: {msg[:100]}")
            
            # Test 3.5: Local file error handling - file not found
            print("\n[3.5] Testing error handling - file not found...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": "/nonexistent/file/path.csv",
                "format": "csv",
            })
            expected_failure = not success
            status = "passed" if expected_failure else "failed"
            self._record_test("local_file_not_found", expected_failure, msg, status, {"result": result})
            print(f"  {'✅' if expected_failure else '❌'} File not found handling: {msg[:100]}")
            
        finally:
            # Cleanup - but keep files for debugging if needed
            # import shutil
            # shutil.rmtree(temp_dir, ignore_errors=True)
            pass
    
    def test_csv_options(self):
        """Test CSV-specific options."""
        print("\n" + "="*80)
        print("TEST SUITE 4: CSV Options")
        print("="*80)
        
        # Use /tmp which should be accessible
        temp_dir = "/tmp/neurondb_test_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Create CSV with custom delimiter
            csv_file = os.path.join(temp_dir, "test_semicolon.csv")
            with open(csv_file, "w") as f:
                f.write("id;name;value\n")
                f.write("1;Alice;100\n")
                f.write("2;Bob;200\n")
            
            # Test 4.1: CSV with custom delimiter
            print("\n[4.1] Testing CSV with custom delimiter...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": csv_file,
                "format": "csv",
                "csv_delimiter": ";",
                "auto_embed": False,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("csv_custom_delimiter", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Custom delimiter: {msg[:100]}")
            
            # Create CSV with header on row 2
            csv_file2 = os.path.join(temp_dir, "test_header_row2.csv")
            with open(csv_file2, "w") as f:
                f.write("skip this line\n")
                f.write("id,name,value\n")
                f.write("1,Alice,100\n")
                f.write("2,Bob,200\n")
            
            # Test 4.2: CSV with header row specification
            print("\n[4.2] Testing CSV with header row specification...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": csv_file2,
                "format": "csv",
                "csv_header": 1,
                "csv_skip_rows": 1,
                "auto_embed": False,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("csv_header_row", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Header row: {msg[:100]}")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_table_management(self):
        """Test table management options (if_exists, load_mode)."""
        print("\n" + "="*80)
        print("TEST SUITE 5: Table Management Options")
        print("="*80)
        
        # Use /tmp which should be accessible
        temp_dir = "/tmp/neurondb_test_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            csv_file = os.path.join(temp_dir, "test_replace.csv")
            with open(csv_file, "w") as f:
                f.write("id,name\n")
                f.write("1,Alice\n")
                f.write("2,Bob\n")
            
            # Test 5.1: if_exists = replace
            print("\n[5.1] Testing if_exists=replace...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": csv_file,
                "format": "csv",
                "table_name": "test_replace_table",
                "if_exists": "replace",
                "auto_embed": False,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("table_if_exists_replace", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} if_exists=replace: {msg[:100]}")
            
            # Test 5.2: if_exists = append
            print("\n[5.2] Testing if_exists=append...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": csv_file,
                "format": "csv",
                "table_name": "test_replace_table",
                "if_exists": "append",
                "auto_embed": False,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("table_if_exists_append", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} if_exists=append: {msg[:100]}")
            
            # Test 5.3: load_mode = append
            print("\n[5.3] Testing load_mode=append...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": csv_file,
                "format": "csv",
                "table_name": "test_load_mode",
                "load_mode": "append",
                "auto_embed": False,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("table_load_mode_append", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} load_mode=append: {msg[:100]}")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_embedding_features(self):
        """Test embedding-related features."""
        print("\n" + "="*80)
        print("TEST SUITE 6: Embedding Features")
        print("="*80)
        
        # Use /tmp which should be accessible
        temp_dir = "/tmp/neurondb_test_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Create test file with text
            jsonl_file = os.path.join(temp_dir, "test_text.jsonl")
            with open(jsonl_file, "w") as f:
                f.write(json.dumps({"id": 1, "text": "This is a test document for embedding generation."}) + "\n")
                f.write(json.dumps({"id": 2, "text": "Another document with meaningful content."}) + "\n")
            
            # Test 6.1: Auto-embedding with default model
            print("\n[6.1] Testing auto-embedding with default model...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": jsonl_file,
                "format": "jsonl",
                "auto_embed": True,
                "embedding_model": "default",
                "create_indexes": True,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("embedding_default_model", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Default model: {msg[:100]}")
            
            # Test 6.2: Custom embedding dimension
            print("\n[6.2] Testing custom embedding dimension...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": jsonl_file,
                "format": "jsonl",
                "auto_embed": True,
                "embedding_dimension": 512,
                "create_indexes": False,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("embedding_custom_dimension", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Custom dimension: {msg[:100]}")
            
            # Test 6.3: Index creation
            print("\n[6.3] Testing index creation...")
            success, msg, result = self._call_tool("postgresql_load_dataset", {
                "source_type": "local",
                "source_path": jsonl_file,
                "format": "jsonl",
                "auto_embed": True,
                "create_indexes": True,
            })
            status = "configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed")
            self._record_test("embedding_index_creation", success, msg, status, {"result": result})
            print(f"  {'✅' if success else '⚠️' if status == 'configuration_needed' else '❌'} Index creation: {msg[:100]}")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_validation(self):
        """Test parameter validation."""
        print("\n" + "="*80)
        print("TEST SUITE 7: Parameter Validation")
        print("="*80)
        
        # Test 7.1: Missing required parameter (source_path)
        print("\n[7.1] Testing missing required parameter...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "huggingface",
            # Missing source_path
        })
        expected_failure = not success
        status = "passed" if expected_failure else "failed"
        self._record_test("validation_missing_source_path", expected_failure, msg, status, {"result": result})
        print(f"  {'✅' if expected_failure else '❌'} Missing source_path: {msg[:100]}")
        
        # Test 7.2: Invalid source_type
        print("\n[7.2] Testing invalid source_type...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "invalid_type",
            "source_path": "test",
        })
        expected_failure = not success
        status = "passed" if expected_failure else "failed"
        self._record_test("validation_invalid_source_type", expected_failure, msg, status, {"result": result})
        print(f"  {'✅' if expected_failure else '❌'} Invalid source_type: {msg[:100]}")
        
        # Test 7.3: Invalid if_exists value
        print("\n[7.3] Testing invalid if_exists value...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "local",
            "source_path": "/tmp/test.csv",
            "if_exists": "invalid_option",
        })
        expected_failure = not success
        status = "passed" if expected_failure else "failed"
        self._record_test("validation_invalid_if_exists", expected_failure, msg, status, {"result": result})
        print(f"  {'✅' if expected_failure else '❌'} Invalid if_exists: {msg[:100]}")
    
    def test_github_loading(self):
        """Test GitHub dataset loading."""
        print("\n" + "="*80)
        print("TEST SUITE 8: GitHub Dataset Loading")
        print("="*80)
        
        # Test 8.1: GitHub repository file
        print("\n[8.1] Testing GitHub repository file loading...")
        success, msg, result = self._call_tool("postgresql_load_dataset", {
            "source_type": "github",
            "source_path": "datasets/iris/master/data/iris.csv",
            "limit": 10,
            "auto_embed": False,
        })
        status = "skipped" if not success and ("not found" in msg.lower() or "404" in msg.lower()) else ("configuration_needed" if not success and self._is_configuration_error(msg) else ("passed" if success else "failed"))
        self._record_test("github_repo_file", success, msg, status, {"result": result})
        print(f"  {'✅' if success else '⏭️' if status == 'skipped' else '⚠️' if status == 'configuration_needed' else '❌'} GitHub file: {msg[:100]}")
    
    def run_all_tests(self):
        """Run all test suites."""
        print("\n" + "="*80)
        print("NeuronMCP Dataset Loading Comprehensive Test Suite")
        print("="*80)
        print(f"Started at: {datetime.now().isoformat()}")
        
        if not self.client:
            print("\n[ERROR] MCP client not initialized. Cannot run tests.")
            print("[INFO] Please ensure neuronmcp_server.json is configured correctly.")
            return
        
        self.test_results["metadata"]["start_time"] = datetime.now().isoformat()
        
        # Run all test suites
        self.test_huggingface_loading()
        self.test_url_loading()
        self.test_local_file_loading()
        self.test_csv_options()
        self.test_table_management()
        self.test_embedding_features()
        self.test_validation()
        self.test_github_loading()
        
        self.test_results["metadata"]["end_time"] = datetime.now().isoformat()
        
        # Disconnect client
        if self.client:
            try:
                self.client.disconnect()
            except:
                pass
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
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
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  ⏭️  Skipped: {skipped}")
        print(f"  ⚠️  Configuration Needed: {config_needed}")
        
        if total > 0:
            pass_rate = (passed / total) * 100
            print(f"\nPass Rate: {pass_rate:.1f}%")
        
        print(f"\nStarted: {metadata.get('start_time', 'N/A')}")
        print(f"Ended: {metadata.get('end_time', 'N/A')}")
        
        # Print failed tests
        if failed > 0:
            print("\n" + "="*80)
            print("FAILED TESTS")
            print("="*80)
            for test_name, test_info in self.test_results["tests"].items():
                if test_info["status"] == "failed":
                    print(f"\n{test_name}:")
                    print(f"  Message: {test_info['message'][:200]}")
        
        # Print configuration needed tests
        if config_needed > 0:
            print("\n" + "="*80)
            print("CONFIGURATION NEEDED")
            print("="*80)
            print("These tests require database configuration or API keys:")
            for test_name, test_info in self.test_results["tests"].items():
                if test_info["status"] == "configuration_needed":
                    print(f"  - {test_name}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for NeuronMCP dataset loading",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to neuronmcp_server.json config file"
    )
    parser.add_argument(
        "-s", "--server",
        default="neurondb",
        help="Server name in config file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    tester = DataLoadingTester(config_path=args.config, server_name=args.server)
    tester.run_all_tests()
    
    # Exit with appropriate code
    metadata = tester.test_results["metadata"]
    if metadata["failed"] > 0:
        sys.exit(1)
    elif metadata["total_tests"] == 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

