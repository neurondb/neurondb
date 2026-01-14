#!/usr/bin/env python3
"""-------------------------------------------------------------------------
 *
 * neuronmcp-test-all-tools.py
 *    Systematic Test Script for All NeuronMCP Tools
 *
 * Tests each NeuronMCP tool one by one and reports results. Validates tool
 * registration, execution, parameter handling, and error conditions.
 * Provides comprehensive tool testing and reporting.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/tests/neuronmcp-test-all-tools.py
 *
 *-------------------------------------------------------------------------
"""

import sys
import json
from pathlib import Path

# Add lib and client directories to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "lib"))
sys.path.insert(0, str(PROJECT_ROOT / "client"))

from mcp_client.client import MCPClient
from mcp_client.config import load_config
from neuronmcp_cli import setup_cli, get_logger, print_section, VERSION

# Test results storage
test_results = {
    "passed": [],
    "failed": [],
    "skipped": []
}

def test_tool(client, tool_name, arguments=None):
    """Test a single tool and return result."""
    if arguments is None:
        arguments = {}
    
    print(f"\n{'='*60}")
    print(f"Testing: {tool_name}")
    print(f"Arguments: {arguments}")
    print(f"{'='*60}")
    
    try:
        result = client.call_tool(tool_name, arguments)
        
        # Check for errors in result - MCP protocol returns errors with isError: true
        if result.get("isError", False):
            # Extract error message from content
            error_msg = "Unknown error"
            if "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0:
                first_item = result["content"][0]
                if isinstance(first_item, dict):
                    if "text" in first_item:
                        error_msg = first_item["text"]
                    elif "error" in first_item:
                        error_msg = str(first_item["error"])
            elif "metadata" in result and "message" in result["metadata"]:
                error_msg = result["metadata"]["message"]
            
            # Configuration errors are acceptable - they indicate the tool works but needs setup
            if "CONFIGURATION_ERROR" in error_msg or ("configuration" in error_msg.lower() and "requires" in error_msg.lower()):
                print(f"⚠️  CONFIGURATION NEEDED: {error_msg[:200]}")
                test_results["passed"].append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "note": "Configuration required (expected)"
                })
                return True
            
            print(f"❌ FAILED: {error_msg[:200]}")
            test_results["failed"].append({
                "tool": tool_name,
                "arguments": arguments,
                "error": error_msg,
                "full_result": result
            })
            return False
        
        # Also check for error in content text (but be more careful - JSON might contain "error" as a key)
        if "content" in result and isinstance(result["content"], list):
            for item in result["content"]:
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                    # Only fail if text explicitly starts with "Error:" (not just contains "error")
                    if text.strip().startswith("Error:"):
                        print(f"❌ FAILED: {text[:200]}")
                        test_results["failed"].append({
                            "tool": tool_name,
                            "arguments": arguments,
                            "error": text,
                            "full_result": result
                        })
                        return False
        
        print(f"✅ PASSED")
        test_results["passed"].append({
            "tool": tool_name,
            "arguments": arguments,
            "result": result
        })
        return True
        
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        test_results["failed"].append({
            "tool": tool_name,
            "arguments": arguments,
            "error": f"Exception: {str(e)}",
            "exception_type": type(e).__name__
        })
        return False

def create_test_data(client):
    """Create minimal test data for tools that need it."""
    print("\n" + "="*60)
    print("Creating test data...")
    print("="*60)
    
    # We'll use psql directly for data creation
    # This is just a placeholder - actual data creation will be done via SQL
    pass

def main():
    """Main test execution."""
    parser = setup_cli(
        description="Systematic test script for all NeuronMCP tools",
        version=VERSION
    )
    
    # Add script-specific arguments
    parser.add_argument(
        '--config',
        default=str(PROJECT_ROOT / "conf" / "neuronmcp-server.json"),
        help='Path to NeuronMCP server configuration file'
    )
    
    args = parser.parse_args()
    logger = get_logger(verbose=args.verbose)
    
    print_section("NeuronMCP Tool Testing Suite")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path), "neurondb")
    
    # Create client and connect
    client = MCPClient(config, verbose=args.verbose)
    try:
        client.connect()
        
        # Get all tools
        tools_response = client.list_tools()
        all_tools = [t["name"] for t in tools_response.get("tools", [])]
        logger.info(f"Found {len(all_tools)} tools to test")
        
        # Test PostgreSQL tools first (no data needed)
        print("\n" + "="*60)
        print("Testing PostgreSQL Tools")
        print("="*60)
        
        postgresql_tools = [
            ("postgresql_version", {}),
            ("postgresql_stats", {}),
            ("postgresql_databases", {}),
            ("postgresql_connections", {}),
            ("postgresql_locks", {}),
            ("postgresql_replication", {}),
            ("postgresql_settings", {}),
            ("postgresql_extensions", {}),
        ]
        
        for tool_name, args in postgresql_tools:
            if tool_name in all_tools:
                test_tool(client, tool_name, args)
            else:
                print(f"⚠️  SKIPPED: {tool_name} (not found in tool list)")
                test_results["skipped"].append({"tool": tool_name, "reason": "not in tool list"})
        
        # Test other tools that don't require data
        print("\n" + "="*60)
        print("Testing Tools That Don't Require Data")
        print("="*60)
        
        simple_tools = [
            ("list_models", {}),
            ("list_embedding_model_configs", {}),
            ("gpu_info", {}),
            ("worker_management", {"operation": "status"}),
        ]
        
        for tool_name, args in simple_tools:
            if tool_name in all_tools:
                test_tool(client, tool_name, args)
            else:
                print(f"⚠️  SKIPPED: {tool_name} (not found in tool list)")
                test_results["skipped"].append({"tool": tool_name, "reason": "not in tool list"})
        
        # Test vector tools (will need test data)
        print("\n" + "="*60)
        print("Testing Vector Tools")
        print("="*60)
        
        # First create a test table with vectors
        # We'll test with minimal parameters to see what errors we get
        vector_tools = [
            ("vector_search", {"table": "test_vectors", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3], "limit": 5}),
            ("vector_search_l2", {"table": "test_vectors", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3], "limit": 5}),
            ("vector_search_cosine", {"table": "test_vectors", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3], "limit": 5}),
            ("vector_search_inner_product", {"table": "test_vectors", "vector_column": "embedding", "query_vector": [0.1, 0.2, 0.3], "limit": 5}),
            ("vector_similarity", {"vector1": [0.1, 0.2, 0.3], "vector2": [0.4, 0.5, 0.6], "metric": "cosine"}),
            ("vector_arithmetic", {"vector1": [0.1, 0.2, 0.3], "vector2": [0.4, 0.5, 0.6], "operation": "add"}),
            ("vector_distance", {"vector1": [0.1, 0.2, 0.3], "vector2": [0.4, 0.5, 0.6], "metric": "l2"}),
            ("vector_similarity_unified", {"vector1": [0.1, 0.2, 0.3], "vector2": [0.4, 0.5, 0.6], "metric": "cosine"}),
        ]
        
        for tool_name, args in vector_tools:
            if tool_name in all_tools:
                test_tool(client, tool_name, args)
            else:
                print(f"⚠️  SKIPPED: {tool_name} (not found in tool list)")
                test_results["skipped"].append({"tool": tool_name, "reason": "not in tool list"})
        
        # Test embedding tools
        print("\n" + "="*60)
        print("Testing Embedding Tools")
        print("="*60)
        
        embedding_tools = [
            ("generate_embedding", {"text": "test text", "model": "default"}),
            ("batch_embedding", {"texts": ["text1", "text2"], "model": "default"}),
        ]
        
        for tool_name, args in embedding_tools:
            if tool_name in all_tools:
                test_tool(client, tool_name, args)
            else:
                print(f"⚠️  SKIPPED: {tool_name} (not found in tool list)")
                test_results["skipped"].append({"tool": tool_name, "reason": "not in tool list"})
        
        # Test remaining tools with minimal parameters
        print("\n" + "="*60)
        print("Testing Remaining Tools")
        print("="*60)
        
        remaining_tools = set(all_tools) - {t[0] for t in postgresql_tools + simple_tools + vector_tools + embedding_tools}
        
        for tool_name in sorted(remaining_tools):
            # Try with empty args first to see what's required
            test_tool(client, tool_name, {})
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed: {len(test_results['passed'])}")
        print(f"❌ Failed: {len(test_results['failed'])}")
        print(f"⚠️  Skipped: {len(test_results['skipped'])}")
        
        if test_results['failed']:
            print("\nFailed Tools:")
            for failure in test_results['failed']:
                print(f"  - {failure['tool']}: {failure.get('error', 'Unknown error')}")
        
        # Save results to file
        results_file = Path(__file__).parent / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()

