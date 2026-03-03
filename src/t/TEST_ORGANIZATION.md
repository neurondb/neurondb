# Test Organization (Regenerated)

This file explains how to organize tests as the suite grows.

## Conventions

- Keep TAP tests small and composable.
- Place heavy stress tests under `NeuronDB/tests/`.
- Prefer naming that reflects the feature area:
  - `vector_*`
  - `index_*`
  - `ml_*`
  - `gpu_*`

## When to add a new file vs extend an existing one

- New file when:
  - new feature area
  - different runtime constraints (GPU-only, long-running)
- Extend existing file when:
  - small regression near existing coverage

# NeuronDB TAP Test Suite Organization

## Test File Numbering Scheme

### 000-009: Foundation & Basic Tests
- `001_basic_minimal.t` - Minimal setup and basic operations
- `002_basic_maximal.t` - Maximal usage examples
- `003_comprehensive.t` - Comprehensive coverage (keep as integration test)

### 010-019: Vector Core Operations
- `010_vector_types.t` - Vector type creation, dimensions, NULL handling
- `011_vector_arithmetic.t` - Addition, subtraction, scalar ops
- `012_vector_functions.t` - vector_norm, vector_normalize, conversions
- `013_vector_operators.t` - Comparison operators, equality
- `014_vector_edge_cases.t` - Edge cases, boundary conditions

### 020-029: Distance Metrics
- `020_distance_l2.t` - L2 distance exhaustive tests
- `021_distance_cosine.t` - Cosine distance exhaustive tests
- `022_distance_inner_product.t` - Inner product tests
- `023_distance_manhattan.t` - Manhattan distance tests
- `024_distance_hamming.t` - Hamming distance tests
- `025_distance_jaccard.t` - Jaccard distance tests
- `026_distance_custom.t` - Custom distance metrics
- `027_distance_edge_cases.t` - Distance metric edge cases

### 030-039: Aggregates & Window Functions
- `030_aggregates_basic.t` - vector_avg, vector_sum
- `031_aggregates_advanced.t` - vector_min, vector_max, stats
- `032_aggregates_group_by.t` - Aggregates with GROUP BY
- `033_window_functions.t` - Window functions with vectors
- `034_aggregates_edge_cases.t` - Aggregate edge cases

### 040-049: Indexes
- `040_index_hnsw.t` - HNSW index creation and queries
- `041_index_ivf.t` - IVF index creation and queries
- `042_index_maintenance.t` - VACUUM, REINDEX, health
- `043_index_performance.t` - Performance benchmarks
- `044_index_hybrid.t` - Hybrid vector+FTS indexes
- `045_index_temporal.t` - Temporal indexes
- `046_index_edge_cases.t` - Index edge cases

### 050-059: ML Algorithms - Regression
- `050_ml_linear_regression.t` - Linear regression
- `051_ml_ridge.t` - Ridge regression
- `052_ml_lasso.t` - Lasso regression
- `053_ml_elastic_net.t` - Elastic net
- `054_ml_polynomial.t` - Polynomial regression

### 060-069: ML Algorithms - Classification
- `060_ml_logistic_regression.t` - Logistic regression
- `061_ml_svm.t` - Support Vector Machine
- `062_ml_decision_tree.t` - Decision trees
- `063_ml_naive_bayes.t` - Naive Bayes
- `064_ml_knn_classifier.t` - KNN classifier

### 070-079: ML Algorithms - Ensemble & Boosting
- `070_ml_random_forest.t` - Random Forest
- `071_ml_xgboost.t` - XGBoost
- `072_ml_lightgbm.t` - LightGBM
- `073_ml_catboost.t` - CatBoost

### 080-089: ML Algorithms - Clustering
- `080_ml_kmeans.t` - K-Means clustering
- `081_ml_minibatch_kmeans.t` - Mini-batch K-Means
- `082_ml_dbscan.t` - DBSCAN clustering
- `083_ml_gmm.t` - Gaussian Mixture Model
- `084_ml_hierarchical.t` - Hierarchical clustering

### 090-099: ML Algorithms - Other
- `090_ml_dimensionality.t` - PCA, dimensionality reduction
- `091_ml_outlier_detection.t` - Outlier detection
- `092_ml_quality_metrics.t` - Quality metrics (Recall@K, etc.)
- `093_ml_drift_detection.t` - Drift detection
- `094_ml_unified_api.t` - Unified ML API (train, predict, evaluate)
- `095_ml_topic_discovery.t` - Topic modeling

### 100-109: GPU Features
- `100_gpu_detection.t` - GPU availability detection
- `101_gpu_operations.t` - GPU distance computations
- `102_gpu_ml.t` - GPU-accelerated ML
- `103_gpu_errors.t` - GPU error handling

### 110-119: Sparse Vectors
- `110_sparse_vectors.t` - Sparse vector type and operations
- `111_sparse_index.t` - Sparse vector indexes
- `112_sparse_hybrid_search.t` - Hybrid dense+sparse search
- `113_sparse_bm25.t` - BM25 scoring

### 120-129: Quantization
- `120_quantization_pq.t` - Product Quantization
- `121_quantization_opq.t` - Optimized PQ
- `122_quantization_fp8.t` - FP8 quantization
- `123_quantization_int8.t` - INT8 quantization
- `124_quantization_accuracy.t` - Quantization accuracy tests

### 130-139: Multimodal & Reranking
- `130_multimodal_embeddings.t` - Image/text embeddings
- `131_multimodal_search.t` - Cross-modal search
- `132_reranking_cross_encoder.t` - Cross-encoder reranking
- `133_reranking_flash.t` - Flash attention reranking

### 140-149: Workers & Job Queue
- `140_workers.t` - Background workers
- `141_job_queue.t` - Job queue operations
- `142_async_operations.t` - Async operations

### 150-159: Advanced Features
- `150_distributed_search.t` - Distributed KNN search
- `151_data_management.t` - Time-travel, compression
- `152_tenant_management.t` - Multi-tenancy
- `153_catalog_operations.t` - Catalog queries
- `154_observability.t` - Metrics, monitoring

### 160-169: Edge Cases & Error Handling
- `160_edge_cases.t` - Edge cases (empty, NULL, extremes)
- `161_error_handling.t` - Error handling
- `162_negative_tests.t` - Negative test cases

### 170-179: Integration & Regression
- `170_integration.t` - End-to-end integration tests
- `171_regression_suite.t` - Regression tests

## Module Usage

All test files MUST use:
- `PostgresNode` - Database node management
- `TapTest` - Test assertions
- `NeuronDB` - General NeuronDB helpers
- Feature-specific modules (VectorOps, MLHelpers, etc.) as needed

## Test Structure Template

```perl
#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use VectorOps;  # or other feature modules as needed

=head1 NAME

XXX_feature_name.t - Description

=head1 DESCRIPTION

Comprehensive tests for [feature]

=cut

plan tests => N;

my $node = PostgresNode->new('test_name');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# Test sections with subtests

$node->stop();
$node->cleanup();

done_testing();
```

