# API Stability Report

This document provides an overview of NeuronDB SQL API stability classifications.

## Summary

| Stability Level | Count | Percentage |
|----------------|-------|------------|
| **Stable** | ~150+ | ~60% |
| **Experimental** | ~80+ | ~30% |
| **Internal** | ~20+ | ~10% |

*Note: Exact counts may vary by version. Run `generate_stability_report.sql` for current numbers.*

## Functions by Category

### Vector Operations

| Function | Stability | Notes |
|----------|-----------|-------|
| `<->` (L2 distance)` | **Stable** | Core operator |
| `<=>` (Cosine distance) | **Stable** | Core operator |
| `<#>` (Inner product) | **Stable** | Core operator |
| `vector_l2_distance_gpu()` | **Experimental** | GPU acceleration |
| `vector_cosine_distance_gpu()` | **Experimental** | GPU acceleration |

### Embedding Functions

| Function | Stability | Notes |
|----------|-----------|-------|
| `neurondb_embed()` | **Stable** | Primary embedding function |
| `neurondb_embed_batch()` | **Stable** | Batch embedding |
| `embed_text()` | **Stable** | Alias for neurondb_embed |

### ML Algorithms

| Category | Functions | Stability |
|----------|-----------|-----------|
| Classification | `train_random_forest_classifier()`, `train_xgboost_classifier()`, etc. | **Stable** |
| Regression | `train_linear_regression()`, `train_ridge_regression()`, etc. | **Stable** |
| Clustering | `cluster_kmeans()`, `cluster_minibatch_kmeans()` | **Stable** |
| Clustering (Advanced) | `cluster_gmm()` | **Experimental** |
| Outlier Detection | `detect_outliers_zscore()` | **Stable** |

### ML Project Management

| Function | Stability | Notes |
|----------|-----------|-------|
| `neurondb_create_ml_project()` | **Experimental** | New feature |
| `neurondb_train_kmeans_project()` | **Experimental** | New feature |
| `neurondb_list_project_models()` | **Experimental** | New feature |
| `neurondb_deploy_model()` | **Experimental** | New feature |

## Stability Trends

### Recently Promoted to Stable

- `neurondb_embed()` - Promoted in v1.0.0
- `cluster_kmeans()` - Promoted in v1.0.0
- Core distance operators - Always stable

### Experimental Features Under Evaluation

- GPU acceleration functions - Being evaluated for v1.1.0
- ML project management - Being evaluated for v1.2.0
- Advanced clustering algorithms - Being evaluated for v1.1.0

## Deprecation Timeline

Currently, no functions are scheduled for deprecation.

When functions are deprecated:
- Deprecation announced in release notes
- Minimum 6 months notice before removal
- Migration path provided

## Recommendations

### Production Applications

**Use only Stable functions:**
- Core vector operations (`<->`, `<=>`, `<#>`)
- Embedding functions (`neurondb_embed`, `neurondb_embed_batch`)
- Standard ML algorithms (classification, regression, clustering)
- Index operations (HNSW, IVF)

### Development/Evaluation

**Experimental functions suitable for:**
- Testing new features
- Proof-of-concept development
- Performance evaluation
- Feedback collection

**Not recommended for:**
- Production systems requiring stability guarantees
- Long-term data pipelines
- Critical business logic

## Generating Current Report

Run this SQL query to generate a current stability report:

```sql
-- Generate stability report
SELECT 
  p.proname AS function_name,
  pg_get_function_identity_arguments(p.oid) AS arguments,
  CASE 
    WHEN obj_description(p.oid, 'pg_proc') LIKE '%Stability: STABLE%' THEN 'Stable'
    WHEN obj_description(p.oid, 'pg_proc') LIKE '%Stability: EXPERIMENTAL%' THEN 'Experimental'
    WHEN obj_description(p.oid, 'pg_proc') LIKE '%Stability: INTERNAL%' THEN 'Internal'
    ELSE 'Unknown'
  END AS stability
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'neurondb'
ORDER BY stability, function_name;
```

## Related Documentation

- [Function Stability Policy](function-stability.md) - Detailed stability definitions
- [SQL API Reference](sql-api.md) - Complete function reference with stability labels
- [Deprecation Policy](deprecation-policy.md) - How deprecations are handled








