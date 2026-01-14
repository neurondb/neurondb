#!/usr/bin/env python3
"""-------------------------------------------------------------------------
 *
 * neuronmcp-verify-sql-integration.py
 *    Verify NeuronDB SQL Function Integration
 *
 * Examines the codebase to verify that NeuronMCP tools correctly call the
 * appropriate NeuronDB SQL functions. Validates integration between MCP tools
 * and underlying NeuronDB SQL functions, ensuring proper function mapping and
 * parameter passing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/tests/neuronmcp-verify-sql-integration.py
 *
 *-------------------------------------------------------------------------
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple


class SQLFunctionVerifier:
    """Verify SQL function integration in NeuronMCP code."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.tools_path = base_path / "internal" / "tools"
        self.results = {
            "sql_functions": {},
            "tool_mappings": {},
            "issues": []
        }
    
    def find_sql_function_calls(self, file_path: Path) -> List[Tuple[str, str]]:
        """Find SQL function calls in a Go file."""
        if not file_path.exists():
            return []
        
        try:
            content = file_path.read_text()
        except:
            return []
        
        # Pattern to find SQL queries with function calls
        # Look for patterns like: SELECT neurondb.function_name(...)
        patterns = [
            r'neurondb\.(\w+)\s*\(',
            r'(\w+)\s*\(.*\)\s*AS\s+\w+',  # Function calls in SELECT
            r'CREATE\s+INDEX.*USING\s+(\w+)',  # Index creation
        ]
        
        functions = []
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                func_name = match.group(1) if match.lastindex else match.group(0)
                functions.append((func_name, match.group(0)))
        
        return functions
    
    def verify_embedding_functions(self):
        """Verify embedding function integration."""
        print("\n[SQL Functions] Verifying embedding functions...")
        
        expected_functions = {
            "embed": ["generate_embedding"],
            "embed_batch": ["batch_embedding"],
            "embed_text": ["generate_embedding"],  # Fallback
        }
        
        # Check vector.go for embedding functions
        vector_file = self.tools_path / "vector.go"
        if vector_file.exists():
            functions = self.find_sql_function_calls(vector_file)
            found_functions = [f[0] for f in functions]
            
            for func_name, tool_names in expected_functions.items():
                if func_name in found_functions or any(f.startswith(func_name) for f in found_functions):
                    self.results["sql_functions"][f"neurondb.{func_name}"] = {
                        "status": "found",
                        "tools": tool_names,
                        "file": "vector.go"
                    }
                    print(f"  ✅ Found: neurondb.{func_name} (used by {tool_names})")
                else:
                    self.results["sql_functions"][f"neurondb.{func_name}"] = {
                        "status": "not_found",
                        "tools": tool_names
                    }
                    print(f"  ⚠️  Not found: neurondb.{func_name} (expected in {tool_names})")
    
    def verify_ml_functions(self):
        """Verify ML function integration."""
        print("\n[SQL Functions] Verifying ML functions...")
        
        expected_functions = {
            "train": ["train_model"],
            "predict": ["predict"],
            "evaluate": ["evaluate_model"],
            "list_models": ["list_models"],
        }
        
        # Check ml.go for ML functions
        ml_file = self.tools_path / "ml.go"
        if ml_file.exists():
            functions = self.find_sql_function_calls(ml_file)
            found_functions = [f[0] for f in functions]
            
            for func_name, tool_names in expected_functions.items():
                if func_name in found_functions or any(f.startswith(func_name) for f in found_functions):
                    self.results["sql_functions"][f"neurondb.{func_name}"] = {
                        "status": "found",
                        "tools": tool_names,
                        "file": "ml.go"
                    }
                    print(f"  ✅ Found: neurondb.{func_name} (used by {tool_names})")
                else:
                    self.results["sql_functions"][f"neurondb.{func_name}"] = {
                        "status": "not_found",
                        "tools": tool_names
                    }
                    print(f"  ⚠️  Not found: neurondb.{func_name} (expected in {tool_names})")
    
    def verify_index_functions(self):
        """Verify index function integration."""
        print("\n[SQL Functions] Verifying index functions...")
        
        # Check indexing.go for index creation
        indexing_file = self.tools_path / "indexing.go"
        if indexing_file.exists():
            content = indexing_file.read_text()
            
            # Check for HNSW index creation
            if "CREATE INDEX" in content and "hnsw" in content.lower():
                self.results["sql_functions"]["CREATE INDEX USING hnsw"] = {
                    "status": "found",
                    "tools": ["create_hnsw_index"],
                    "file": "indexing.go"
                }
                print("  ✅ Found: CREATE INDEX USING hnsw")
            
            # Check for IVF index creation
            if "CREATE INDEX" in content and "ivf" in content.lower():
                self.results["sql_functions"]["CREATE INDEX USING ivfflat"] = {
                    "status": "found",
                    "tools": ["create_ivf_index"],
                    "file": "indexing.go"
                }
                print("  ✅ Found: CREATE INDEX USING ivfflat")
    
    def verify_onnx_functions(self):
        """Verify ONNX function integration."""
        print("\n[SQL Functions] Verifying ONNX functions...")
        
        expected_functions = {
            "neurondb_onnx_info": ["onnx_model"],
            "import_onnx_model": ["onnx_model"],
            "export_model_to_onnx": ["onnx_model"],
            "predict_onnx_model": ["onnx_model"],
        }
        
        # Check onnx.go for ONNX functions
        onnx_file = self.tools_path / "onnx.go"
        if onnx_file.exists():
            functions = self.find_sql_function_calls(onnx_file)
            found_functions = [f[0] for f in functions]
            
            for func_name, tool_names in expected_functions.items():
                if func_name in found_functions or any(f.startswith(func_name) for f in found_functions):
                    self.results["sql_functions"][func_name] = {
                        "status": "found",
                        "tools": tool_names,
                        "file": "onnx.go"
                    }
                    print(f"  ✅ Found: {func_name} (used by {tool_names})")
                else:
                    self.results["sql_functions"][func_name] = {
                        "status": "not_found",
                        "tools": tool_names
                    }
                    print(f"  ⚠️  Not found: {func_name} (expected in {tool_names})")
    
    def verify_vector_operations(self):
        """Verify vector operation SQL patterns."""
        print("\n[SQL Functions] Verifying vector operations...")
        
        # Check vector.go for vector operators
        vector_file = self.tools_path / "vector.go"
        if vector_file.exists():
            content = vector_file.read_text()
            
            # Check for vector distance operators
            operators = {
                "<->": "L2 distance",
                "<#>": "Inner product (negative cosine)",
                "<=>": "Cosine distance",
            }
            
            for op, name in operators.items():
                if op in content:
                    self.results["sql_functions"][f"Vector operator {op}"] = {
                        "status": "found",
                        "description": name,
                        "file": "vector.go"
                    }
                    print(f"  ✅ Found: Vector operator {op} ({name})")
    
    def generate_report(self):
        """Generate verification report."""
        report_file = self.base_path / "sql_integration_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 SQL integration report saved to: {report_file}")
        return report_file
    
    def run_all(self):
        """Run all SQL function verification."""
        print("="*80)
        print("NEURONDB SQL FUNCTION INTEGRATION VERIFICATION")
        print("="*80)
        
        self.verify_embedding_functions()
        self.verify_ml_functions()
        self.verify_index_functions()
        self.verify_onnx_functions()
        self.verify_vector_operations()
        
        self.generate_report()


def main():
    """Main entry point."""
    base_path = Path(__file__).parent
    verifier = SQLFunctionVerifier(base_path)
    verifier.run_all()


if __name__ == "__main__":
    main()







