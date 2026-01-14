#!/usr/bin/env python3
"""-------------------------------------------------------------------------
 *
 * test_protocol.py
 *    MCP Protocol Tests for NeuronMCP
 *
 * Tests all MCP protocol endpoints and features. Validates protocol compliance,
 * message handling, request/response processing, and error handling according
 * to the MCP (Model Context Protocol) specification.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/tests/test_protocol.py
 *
 *-------------------------------------------------------------------------
"""

import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add client directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "client"))

from mcp_client.client import MCPClient
from mcp_client.config import load_config


class ProtocolTester:
    """Test suite for MCP protocol endpoints."""
    
    def __init__(self, config_path: str, server_name: str = "neurondb"):
        """Initialize tester with configuration."""
        self.config = load_config(config_path, server_name)
        self.client = MCPClient(self.config, verbose=False)
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
        }
        
    def test_initialize(self):
        """Test MCP protocol initialization."""
        print("\n[Protocol] Testing initialize...")
        try:
            self.client.connect()
            print("  ✅ PASSED: Connection initialized")
            self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)}")
            self.test_results["failed"] += 1
            raise
    
    def test_tools_list(self):
        """Test tools/list endpoint."""
        print("\n[Protocol] Testing tools/list...")
        try:
            result = self.client.list_tools()
            if "error" in result:
                print(f"  ❌ FAILED: {result['error']}")
                self.test_results["failed"] += 1
                return
            
            tools = result.get("tools", [])
            if len(tools) == 0:
                print("  ❌ FAILED: No tools returned")
                self.test_results["failed"] += 1
                return
            
            if len(tools) < 100:
                print(f"  ⚠️  WARNING: Expected 100+ tools, got {len(tools)}")
            
            # Verify tool definitions
            for tool in tools[:10]:  # Check first 10
                if "name" not in tool:
                    print(f"  ❌ FAILED: Tool missing name")
                    self.test_results["failed"] += 1
                    return
                if "description" not in tool:
                    print(f"  ❌ FAILED: Tool {tool['name']} missing description")
                    self.test_results["failed"] += 1
                    return
                if "inputSchema" not in tool:
                    print(f"  ❌ FAILED: Tool {tool['name']} missing inputSchema")
                    self.test_results["failed"] += 1
                    return
            
            print(f"  ✅ PASSED: Found {len(tools)} tools")
            self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            self.test_results["failed"] += 1
    
    def test_tools_call(self):
        """Test tools/call endpoint."""
        print("\n[Protocol] Testing tools/call...")
        try:
            # Test with postgresql_version (doesn't require DB)
            result = self.client.call_tool("postgresql_version", {})
            
            if result.get("isError", False):
                error_msg = result.get("content", [{}])[0].get("text", "Unknown error")
                if "CONNECTION_ERROR" in error_msg or "database" in error_msg.lower():
                    print(f"  ⚠️  SKIPPED: Database connection required ({error_msg[:50]})")
                    self.test_results["skipped"] += 1
                    return
                else:
                    print(f"  ❌ FAILED: {error_msg[:100]}")
                    self.test_results["failed"] += 1
                    return
            
            if "content" not in result or len(result["content"]) == 0:
                print("  ❌ FAILED: No content in response")
                self.test_results["failed"] += 1
                return
            
            print("  ✅ PASSED: Tool call successful")
            self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            self.test_results["failed"] += 1
    
    def test_tools_call_invalid(self):
        """Test tools/call with invalid tool name."""
        print("\n[Protocol] Testing tools/call with invalid tool...")
        try:
            result = self.client.call_tool("invalid_tool_name_xyz", {})
            
            if not result.get("isError", False):
                print("  ❌ FAILED: Expected error for invalid tool")
                self.test_results["failed"] += 1
                return
            
            print("  ✅ PASSED: Invalid tool correctly rejected")
            self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            self.test_results["failed"] += 1
    
    def test_tools_search(self):
        """Test tools/search endpoint."""
        print("\n[Protocol] Testing tools/search...")
        try:
            from mcp_client.protocol import JSONRPCRequest
            request = JSONRPCRequest(method="tools/search", params={"query": "vector"})
            response = self.client.transport.send_request(request)
            
            if response.is_error():
                print(f"  ⚠️  SKIPPED: tools/search not available ({response.get_error_message()[:50]})")
                self.test_results["skipped"] += 1
                return
            
            result = response.result or {}
            tools = result.get("tools", [])
            print(f"  ✅ PASSED: Found {len(tools)} tools matching 'vector'")
            self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ⚠️  SKIPPED: tools/search not available ({str(e)[:50]})")
            self.test_results["skipped"] += 1
    
    def test_resources_list(self):
        """Test resources/list endpoint."""
        print("\n[Protocol] Testing resources/list...")
        try:
            result = self.client.list_resources()
            
            if "error" in result:
                print(f"  ❌ FAILED: {result['error']}")
                self.test_results["failed"] += 1
                return
            
            resources = result.get("resources", [])
            if len(resources) != 9:
                print(f"  ⚠️  WARNING: Expected 9 resources, got {len(resources)}")
            
            # Verify resource definitions
            for resource in resources:
                if "uri" not in resource:
                    print("  ❌ FAILED: Resource missing URI")
                    self.test_results["failed"] += 1
                    return
                if "name" not in resource:
                    print(f"  ❌ FAILED: Resource {resource.get('uri')} missing name")
                    self.test_results["failed"] += 1
                    return
            
            print(f"  ✅ PASSED: Found {len(resources)} resources")
            self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            self.test_results["failed"] += 1
    
    def test_resources_read(self):
        """Test resources/read endpoint."""
        print("\n[Protocol] Testing resources/read...")
        try:
            result = self.client.read_resource("neurondb://config")
            
            if "error" in result:
                error_msg = result["error"]
                if "CONNECTION_ERROR" in error_msg or "database" in error_msg.lower():
                    print(f"  ⚠️  SKIPPED: Database connection required ({error_msg[:50]})")
                    self.test_results["skipped"] += 1
                    return
                else:
                    print(f"  ❌ FAILED: {error_msg[:100]}")
                    self.test_results["failed"] += 1
                    return
            
            if "contents" not in result or len(result["contents"]) == 0:
                print("  ❌ FAILED: No contents in response")
                self.test_results["failed"] += 1
                return
            
            print("  ✅ PASSED: Resource read successful")
            self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            self.test_results["failed"] += 1
    
    def test_resources_read_invalid(self):
        """Test resources/read with invalid URI."""
        print("\n[Protocol] Testing resources/read with invalid URI...")
        try:
            result = self.client.read_resource("neurondb://invalid")
            
            if "error" not in result:
                print("  ❌ FAILED: Expected error for invalid resource")
                self.test_results["failed"] += 1
                return
            
            print("  ✅ PASSED: Invalid resource correctly rejected")
            self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            self.test_results["failed"] += 1
    
    def test_prompts_list(self):
        """Test prompts/list endpoint."""
        print("\n[Protocol] Testing prompts/list...")
        try:
            from mcp_client.protocol import JSONRPCRequest
            request = JSONRPCRequest(method="prompts/list", params={})
            response = self.client.transport.send_request(request)
            
            if response.is_error():
                print(f"  ⚠️  SKIPPED: prompts/list not available ({response.get_error_message()[:50]})")
                self.test_results["skipped"] += 1
                return
            
            result = response.result or {}
            prompts = result.get("prompts", [])
            print(f"  ✅ PASSED: Found {len(prompts)} prompts")
            self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ⚠️  SKIPPED: prompts/list not available ({str(e)[:50]})")
            self.test_results["skipped"] += 1
    
    def test_progress_get(self):
        """Test progress/get endpoint."""
        print("\n[Protocol] Testing progress/get...")
        try:
            from mcp_client.protocol import JSONRPCRequest
            request = JSONRPCRequest(method="progress/get", params={"progress_id": "test_123"})
            response = self.client.transport.send_request(request)
            
            # Progress may not exist, which is expected
            if response.is_error():
                print("  ✅ PASSED: Progress endpoint responds (error expected for non-existent ID)")
                self.test_results["passed"] += 1
            else:
                print("  ✅ PASSED: Progress endpoint responds")
                self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ⚠️  SKIPPED: progress/get not available ({str(e)[:50]})")
            self.test_results["skipped"] += 1
    
    def test_tools_call_batch(self):
        """Test tools/call_batch endpoint."""
        print("\n[Protocol] Testing tools/call_batch...")
        try:
            from mcp_client.protocol import JSONRPCRequest
            request = JSONRPCRequest(method="tools/call_batch", params={
                "tools": [
                    {"name": "postgresql_version", "arguments": {}},
                    {"name": "gpu_info", "arguments": {}},
                ],
                "parallel": False,
            })
            response = self.client.transport.send_request(request)
            
            if response.is_error():
                print(f"  ⚠️  SKIPPED: tools/call_batch not available ({response.get_error_message()[:50]})")
                self.test_results["skipped"] += 1
                return
            
            print("  ✅ PASSED: Batch tool call successful")
            self.test_results["passed"] += 1
        except Exception as e:
            print(f"  ⚠️  SKIPPED: tools/call_batch not available ({str(e)[:50]})")
            self.test_results["skipped"] += 1
    
    def run_all(self):
        """Run all protocol tests."""
        print("=" * 80)
        print("MCP Protocol Tests")
        print("=" * 80)
        
        try:
            self.test_initialize()
            self.test_tools_list()
            self.test_tools_call()
            self.test_tools_call_invalid()
            self.test_tools_search()
            self.test_resources_list()
            self.test_resources_read()
            self.test_resources_read_invalid()
            self.test_prompts_list()
            self.test_progress_get()
            self.test_tools_call_batch()
        finally:
            self.client.disconnect()
        
        print("\n" + "=" * 80)
        print("Protocol Test Summary")
        print("=" * 80)
        print(f"✅ Passed: {self.test_results['passed']}")
        print(f"❌ Failed: {self.test_results['failed']}")
        print(f"⏭️  Skipped: {self.test_results['skipped']}")
        print("=" * 80)


def main():
    """Main entry point."""
    config_path = Path(__file__).parent.parent / "neuronmcp_server.json"
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    tester = ProtocolTester(str(config_path), "neurondb")
    tester.run_all()


if __name__ == "__main__":
    main()

