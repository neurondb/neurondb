#!/usr/bin/env python3
"""-------------------------------------------------------------------------
 *
 * test_resources.py
 *    Resources Tests for NeuronMCP
 *
 * Tests all NeuronMCP resources including collections, datasets, indexes,
 * models, subscriptions, workers, and system resources. Validates resource
 * discovery, access, and management functionality.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/tests/test_resources.py
 *
 *-------------------------------------------------------------------------
"""

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "client"))

from mcp_client.client import MCPClient
from mcp_client.config import load_config


class ResourcesTester:
    """Test suite for resources."""
    
    def __init__(self, config_path: str, server_name: str = "neurondb"):
        """Initialize tester."""
        self.config = load_config(config_path, server_name)
        self.client = MCPClient(self.config, verbose=False)
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
        }
    
    def test_resource(self, uri: str) -> bool:
        """Test a single resource."""
        try:
            result = self.client.read_resource(uri)
            
            if "error" in result:
                error_msg = result["error"]
                if "CONNECTION_ERROR" in error_msg or "database" in error_msg.lower():
                    self.test_results["skipped"] += 1
                    return False
                else:
                    self.test_results["failed"] += 1
                    return False
            
            if "contents" not in result or len(result["contents"]) == 0:
                self.test_results["failed"] += 1
                return False
            
            self.test_results["passed"] += 1
            return True
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower() or "database" in error_msg.lower():
                self.test_results["skipped"] += 1
            else:
                self.test_results["failed"] += 1
            return False
    
    def test_all_resources(self):
        """Test all 9 resources."""
        print("\n[Resources] Testing All Resources...")
        
        resources = [
            "neurondb://schema",
            "neurondb://models",
            "neurondb://indexes",
            "neurondb://config",
            "neurondb://workers",
            "neurondb://vector_stats",
            "neurondb://index_health",
            "neurondb://datasets",
            "neurondb://collections",
        ]
        
        for uri in resources:
            success = self.test_resource(uri)
            status = "✅" if success else "⚠️" if self.test_results["skipped"] > 0 else "❌"
            print(f"  {status} {uri}")
    
    def test_resources_list(self):
        """Test resources/list endpoint."""
        print("\n[Resources] Testing resources/list...")
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
    
    def test_invalid_resource(self):
        """Test reading invalid resource."""
        print("\n[Resources] Testing invalid resource...")
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
    
    def run_all(self):
        """Run all resource tests."""
        print("=" * 80)
        print("Resources Tests (9 resources)")
        print("=" * 80)
        
        try:
            self.client.connect()
            
            self.test_resources_list()
            self.test_all_resources()
            self.test_invalid_resource()
        finally:
            self.client.disconnect()
        
        print("\n" + "=" * 80)
        print("Resources Test Summary")
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
    
    tester = ResourcesTester(str(config_path), "neurondb")
    tester.run_all()


if __name__ == "__main__":
    main()



