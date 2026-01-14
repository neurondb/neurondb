#!/usr/bin/env python3
"""-------------------------------------------------------------------------
 *
 * run_all_tests.py
 *    Comprehensive Test Runner for NeuronMCP
 *
 * Runs all test suites in the correct order. Orchestrates execution of all
 * NeuronMCP test scripts including unit tests, integration tests, and
 * comprehensive verification suites.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/tests/run_all_tests.py
 *
 *-------------------------------------------------------------------------
"""

import sys
import subprocess
from pathlib import Path


def run_test_suite(test_file: str, description: str) -> bool:
    """Run a test suite and return success status."""
    print("\n" + "=" * 80)
    print(f"Running: {description}")
    print("=" * 80)
    
    test_path = Path(__file__).parent / test_file
    if not test_path.exists():
        print(f"⚠️  Test file not found: {test_file}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=False,
            timeout=300,  # 5 minute timeout per suite
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Test suite timed out: {test_file}")
        return False
    except Exception as e:
        print(f"❌ Error running test suite: {e}")
        return False


def main():
    """Run all test suites."""
    print("=" * 80)
    print("NeuronMCP Comprehensive Test Suite")
    print("=" * 80)
    
    test_suites = [
        ("test_protocol.py", "MCP Protocol Tests"),
        ("test_tools_postgresql.py", "PostgreSQL Tools Tests (27 tools)"),
        ("test_resources.py", "Resources Tests (9 resources)"),
        ("test_comprehensive.py", "Comprehensive Tool Tests (100+ tools)"),
    ]
    
    results = {}
    for test_file, description in test_suites:
        success = run_test_suite(test_file, description)
        results[description] = success
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Suite Summary")
    print("=" * 80)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for description, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {description}")
    
    print("\n" + "=" * 80)
    print(f"Total: {passed}/{total} test suites passed")
    print("=" * 80)
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()



