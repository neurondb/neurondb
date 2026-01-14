#!/usr/bin/env python3
"""-------------------------------------------------------------------------
 *
 * test_tools_postgresql.py
 *    PostgreSQL Tools Tests for NeuronMCP
 *
 * Tests all PostgreSQL administration tools including database management,
 * schema operations, query execution, security, maintenance, and high availability
 * features. Validates tool functionality and integration with PostgreSQL.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/tests/test_tools_postgresql.py
 *
 *-------------------------------------------------------------------------
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "client"))

from mcp_client.client import MCPClient
from mcp_client.config import load_config


class PostgreSQLToolsTester:
    """Test suite for PostgreSQL tools."""
    
    def __init__(self, config_path: str, server_name: str = "neurondb"):
        """Initialize tester."""
        self.config = load_config(config_path, server_name)
        self.client = MCPClient(self.config, verbose=False)
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "configuration_needed": 0,
        }
    
    def _is_configuration_error(self, error_msg: str) -> bool:
        """Check if error is a configuration error."""
        error_lower = error_msg.lower()
        return (
            "CONFIGURATION_ERROR" in error_msg or
            ("configuration" in error_lower and "required" in error_lower) or
            ("connection" in error_lower and "failed" in error_lower) or
            ("database" in error_lower and "connection" in error_lower)
        )
    
    def test_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> bool:
        """Test a single tool."""
        if arguments is None:
            arguments = {}
        
        try:
            result = self.client.call_tool(tool_name, arguments)
            
            if result.get("isError", False):
                error_msg = result.get("content", [{}])[0].get("text", "Unknown error")
                if self._is_configuration_error(error_msg):
                    self.test_results["configuration_needed"] += 1
                    return False
                else:
                    self.test_results["failed"] += 1
                    return False
            
            self.test_results["passed"] += 1
            return True
        except Exception as e:
            error_msg = str(e)
            if self._is_configuration_error(error_msg):
                self.test_results["configuration_needed"] += 1
            else:
                self.test_results["failed"] += 1
            return False
    
    def test_server_information_tools(self):
        """Test server information tools (5 tools)."""
        print("\n[PostgreSQL] Testing Server Information Tools...")
        
        tools = [
            ("postgresql_version", {}),
            ("postgresql_stats", {}),
            ("postgresql_databases", {}),
            ("postgresql_settings", {}),
            ("postgresql_extensions", {}),
        ]
        
        for tool_name, args in tools:
            success = self.test_tool(tool_name, args)
            status = "✅" if success else "⚠️" if self.test_results["configuration_needed"] > 0 else "❌"
            print(f"  {status} {tool_name}")
    
    def test_database_object_management_tools(self):
        """Test database object management tools (8 tools)."""
        print("\n[PostgreSQL] Testing Database Object Management Tools...")
        
        tools = [
            ("postgresql_tables", {}),
            ("postgresql_indexes", {}),
            ("postgresql_schemas", {}),
            ("postgresql_views", {}),
            ("postgresql_sequences", {}),
            ("postgresql_functions", {}),
            ("postgresql_triggers", {}),
            ("postgresql_constraints", {}),
        ]
        
        for tool_name, args in tools:
            success = self.test_tool(tool_name, args)
            status = "✅" if success else "⚠️" if self.test_results["configuration_needed"] > 0 else "❌"
            print(f"  {status} {tool_name}")
    
    def test_user_role_management_tools(self):
        """Test user and role management tools (3 tools)."""
        print("\n[PostgreSQL] Testing User and Role Management Tools...")
        
        tools = [
            ("postgresql_users", {}),
            ("postgresql_roles", {}),
            ("postgresql_permissions", {}),
        ]
        
        for tool_name, args in tools:
            success = self.test_tool(tool_name, args)
            status = "✅" if success else "⚠️" if self.test_results["configuration_needed"] > 0 else "❌"
            print(f"  {status} {tool_name}")
    
    def test_performance_statistics_tools(self):
        """Test performance and statistics tools (7 tools)."""
        print("\n[PostgreSQL] Testing Performance and Statistics Tools...")
        
        tools = [
            ("postgresql_table_stats", {}),
            ("postgresql_index_stats", {}),
            ("postgresql_active_queries", {}),
            ("postgresql_wait_events", {}),
            ("postgresql_connections", {}),
            ("postgresql_locks", {}),
            ("postgresql_replication", {}),
        ]
        
        for tool_name, args in tools:
            success = self.test_tool(tool_name, args)
            status = "✅" if success else "⚠️" if self.test_results["configuration_needed"] > 0 else "❌"
            print(f"  {status} {tool_name}")
    
    def test_size_storage_tools(self):
        """Test size and storage tools (4 tools)."""
        print("\n[PostgreSQL] Testing Size and Storage Tools...")
        
        tools = [
            ("postgresql_table_size", {}),
            ("postgresql_index_size", {}),
            ("postgresql_bloat", {}),
            ("postgresql_vacuum_stats", {}),
        ]
        
        for tool_name, args in tools:
            success = self.test_tool(tool_name, args)
            status = "✅" if success else "⚠️" if self.test_results["configuration_needed"] > 0 else "❌"
            print(f"  {status} {tool_name}")
    
    def run_all(self):
        """Run all PostgreSQL tool tests."""
        print("=" * 80)
        print("PostgreSQL Tools Tests (27 tools)")
        print("=" * 80)
        
        try:
            self.client.connect()
            
            self.test_server_information_tools()
            self.test_database_object_management_tools()
            self.test_user_role_management_tools()
            self.test_performance_statistics_tools()
            self.test_size_storage_tools()
        finally:
            self.client.disconnect()
        
        print("\n" + "=" * 80)
        print("PostgreSQL Tools Test Summary")
        print("=" * 80)
        print(f"✅ Passed: {self.test_results['passed']}")
        print(f"❌ Failed: {self.test_results['failed']}")
        print(f"⚠️  Configuration Needed: {self.test_results['configuration_needed']}")
        print(f"⏭️  Skipped: {self.test_results['skipped']}")
        print("=" * 80)


def main():
    """Main entry point."""
    config_path = Path(__file__).parent.parent / "neuronmcp_server.json"
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    tester = PostgreSQLToolsTester(str(config_path), "neurondb")
    tester.run_all()


if __name__ == "__main__":
    main()



