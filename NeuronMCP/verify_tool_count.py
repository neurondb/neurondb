#!/usr/bin/env python3
"""
Verify Tool Count and Registration

Counts all tools registered in register.go and compares with expected counts.
"""

import re
from pathlib import Path
from typing import Dict, List


def count_tools_in_register_file(register_file: Path) -> Dict[str, int]:
    """Count tools registered in register.go."""
    if not register_file.exists():
        return {}
    
    content = register_file.read_text()
    
    # Count registry.Register calls
    register_pattern = r'registry\.Register\(New(\w+)Tool\('
    matches = re.findall(register_pattern, content)
    
    # Categorize tools
    categories = {
        "vector_search": ["VectorSearch", "VectorSearchL2", "VectorSearchCosine", 
                         "VectorSearchInnerProduct", "VectorSearchL1", 
                         "VectorSearchHamming", "VectorSearchChebyshev", 
                         "VectorSearchMinkowski"],
        "embeddings": ["GenerateEmbedding", "BatchEmbedding", "EmbedImage",
                      "EmbedMultimodal", "EmbedCached", "ConfigureEmbeddingModel",
                      "GetEmbeddingModelConfig", "ListEmbeddingModelConfigs",
                      "DeleteEmbeddingModelConfig"],
        "vector_operations": ["VectorSimilarity", "VectorArithmetic", "VectorDistance",
                            "VectorSimilarityUnified", "CreateVectorIndex"],
        "ml_operations": ["TrainModel", "Predict", "PredictBatch", "EvaluateModel",
                         "ListModels", "GetModelInfo", "DeleteModel", "ExportModel"],
        "analytics": ["ClusterData", "DetectOutliers", "ReduceDimensionality",
                     "AnalyzeData", "QualityMetrics", "DriftDetection", "TopicDiscovery"],
        "rag": ["ProcessDocument", "RetrieveContext", "GenerateResponse",
               "IngestDocuments", "AnswerWithCitations", "ChunkDocument"],
        "indexing": ["CreateHNSWIndex", "CreateIVFIndex", "IndexStatus",
                    "DropIndex", "TuneHNSWIndex", "TuneIVFIndex"],
        "hybrid_search": ["HybridSearch", "TextSearch", "ReciprocalRankFusion",
                         "SemanticKeywordSearch", "MultiVectorSearch",
                         "FacetedVectorSearch", "TemporalVectorSearch",
                         "DiverseVectorSearch"],
        "reranking": ["RerankCrossEncoder", "RerankLLM", "RerankCohere",
                     "RerankColBERT", "RerankLTR", "RerankEnsemble"],
        "quantization": ["VectorQuantization", "QuantizationAnalysis"],
        "advanced": ["TimeSeries", "AutoML", "ONNX", "VectorGraph",
                     "VecmapOperations", "DatasetLoading", "WorkerManagement",
                     "GPUMonitoring"],
        "postgresql": ["PostgreSQLVersion", "PostgreSQLStats", "PostgreSQLDatabaseList",
                      "PostgreSQLConnections", "PostgreSQLLocks", "PostgreSQLReplication",
                      "PostgreSQLSettings", "PostgreSQLExtensions", "PostgreSQLTables",
                      "PostgreSQLIndexes", "PostgreSQLSchemas", "PostgreSQLViews",
                      "PostgreSQLSequences", "PostgreSQLFunctions", "PostgreSQLTriggers",
                      "PostgreSQLConstraints", "PostgreSQLUsers", "PostgreSQLRoles",
                      "PostgreSQLPermissions", "PostgreSQLTableStats", "PostgreSQLIndexStats",
                      "PostgreSQLActiveQueries", "PostgreSQLWaitEvents", "PostgreSQLTableSize",
                      "PostgreSQLIndexSize", "PostgreSQLBloat", "PostgreSQLVacuumStats"]
    }
    
    # Count by category
    category_counts = {}
    for category, tool_names in categories.items():
        count = sum(1 for tool in matches if any(tool.startswith(name) for name in tool_names))
        category_counts[category] = count
    
    total_count = len(matches)
    category_counts["total"] = total_count
    
    return category_counts


def main():
    """Main entry point."""
    base_path = Path(__file__).parent
    register_file = base_path / "internal" / "tools" / "register.go"
    
    print("="*80)
    print("TOOL REGISTRATION VERIFICATION")
    print("="*80)
    
    counts = count_tools_in_register_file(register_file)
    
    if not counts:
        print("❌ Could not read register.go")
        return
    
    print(f"\nTotal Tools Registered: {counts.get('total', 0)}")
    print("\nBreakdown by Category:")
    print("-" * 80)
    
    expected_counts = {
        "vector_search": 8,
        "embeddings": 9,
        "vector_operations": 5,
        "ml_operations": 8,
        "analytics": 7,
        "rag": 6,
        "indexing": 6,
        "hybrid_search": 8,
        "reranking": 6,
        "quantization": 2,
        "advanced": 8,
        "postgresql": 27,
    }
    
    for category, count in sorted(counts.items()):
        if category == "total":
            continue
        
        expected = expected_counts.get(category, 0)
        status = "✅" if count == expected else "⚠️"
        print(f"{status} {category:20s}: {count:3d} (expected: {expected})")
    
    print("-" * 80)
    print(f"Total: {counts.get('total', 0)} tools")
    
    if counts.get('total', 0) >= 100:
        print("\n✅ PASSED: 100+ tools registered")
    else:
        print(f"\n⚠️  WARNING: Only {counts.get('total', 0)} tools registered (expected 100+)")


if __name__ == "__main__":
    main()







