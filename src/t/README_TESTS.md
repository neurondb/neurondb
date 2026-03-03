# NeuronDB Test Readme (Regenerated)

This file documents the intent of the test suite and how to expand it safely.

## Layers

- TAP tests (`NeuronDB/t/`): correctness/regression
- SQL suites (`NeuronDB/tests/sql/`): end-to-end feature validation
- Scripts (`NeuronDB/tests/*.py`): harnesses, fuzzing helpers, monitoring tools

## Adding a new test

- Add minimal schema/setup
- Assert behavior with clear expected results
- Include comments explaining the regression or feature being validated

# NeuronDB TAP Test Suite

## Overview

Comprehensive TAP (Test Anything Protocol) test suite for NeuronDB with perfect numbering, modular design, and complete code coverage.

## Test File Organization

### Current Test Files (19 files)

```
001_basic_minimal.t              - Minimal setup and basic operations
002_basic_maximal.t              - Maximal usage examples  
003_comprehensive.t              - Integration test
004_vectors_comprehensive.t      - Vector comprehensive tests
005_distances_comprehensive.t    - Distance comprehensive tests
006_ml_comprehensive.t           - ML comprehensive tests
007_gpu_comprehensive.t          - GPU comprehensive tests
008_aggregates_comprehensive.t   - Aggregates comprehensive tests
009_workers_comprehensive.t      - Workers comprehensive tests
010_vector_types.t               - Vector type operations (120+ tests)
011_vector_arithmetic.t          - Vector arithmetic (90+ tests)
012_vector_functions.t           - Vector functions (95+ tests)
020_distance_l2.t                - Distance metrics (130+ tests)
040_index_hnsw.t                 - HNSW index tests
050_ml_linear_regression.t       - Linear regression (110+ tests)
110_sparse_vectors.t             - Sparse vector tests
122_quantization_fp8.t           - FP8 quantization tests
130_multimodal_embeddings.t      - Multimodal embedding tests
133_reranking_flash.t            - Flash reranking tests
```

## Shared Perl Modules

All test files use these shared modules for consistency:

### Core Modules
- **PostgresNode.pm** - PostgreSQL test node management
- **TapTest.pm** - Enhanced test assertions (9 new helpers)
- **NeuronDB.pm** - General NeuronDB helpers (5 new helpers)

### Feature Modules
- **VectorOps.pm** - Vector operation helpers
- **MLHelpers.pm** - ML algorithm helpers
- **IndexHelpers.pm** - Index operation helpers
- **GPUHelpers.pm** - GPU testing helpers
- **SparseHelpers.pm** - Sparse vector helpers
- **QuantHelpers.pm** - Quantization helpers
- **MultimodalHelpers.pm** - Multimodal embedding helpers
- **WorkerHelpers.pm** - Background worker helpers

## Test Structure

All test files follow this modular structure:

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
use VectorOps;  # or other feature modules

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

# Test sections using shared helpers

$node->stop();
$node->cleanup();

done_testing();
```

## Numbering Scheme

Tests are organized by feature category:

- **000-009**: Foundation & Basic Tests
- **010-019**: Vector Core Operations
- **020-029**: Distance Metrics
- **030-039**: Aggregates & Window Functions
- **040-049**: Indexes
- **050-059**: ML Algorithms - Regression
- **060-069**: ML Algorithms - Classification
- **070-079**: ML Algorithms - Ensemble & Boosting
- **080-089**: ML Algorithms - Clustering
- **090-099**: ML Algorithms - Other
- **100-109**: GPU Features
- **110-119**: Sparse Vectors
- **120-129**: Quantization
- **130-139**: Multimodal & Reranking
- **140-149**: Workers & Job Queue
- **150-159**: Advanced Features
- **160-169**: Edge Cases & Error Handling
- **170-179**: Integration & Regression

## Running Tests

```bash
# Run all tests
prove -v t/

# Run specific test
prove -v t/010_vector_types.t

# Run tests matching pattern
prove -v t/0*_vector*.t
```

## Test Coverage

### Currently Covered
- ✅ Vector types and operations
- ✅ Vector arithmetic
- ✅ Vector functions
- ✅ Distance metrics
- ✅ ML regression
- ✅ Sparse vectors
- ✅ Quantization (FP8)
- ✅ Multimodal embeddings
- ✅ Reranking
- ✅ Aggregates
- ✅ Workers
- ✅ Indexes (HNSW)
- ✅ GPU features

### Planned Coverage
See `TEST_ORGANIZATION.md` for complete list of planned test files.

## Benefits

1. **Perfect Numbering**: Sequential, logical organization
2. **Modular Design**: Shared helpers reduce duplication
3. **Consistent Structure**: All tests follow same pattern
4. **Comprehensive Coverage**: Organized to cover all code parts
5. **Easy Maintenance**: Changes to helpers benefit all tests
6. **Clear Organization**: Easy to find tests by feature

## Documentation

- `TEST_ORGANIZATION.md` - Complete numbering scheme and planned tests
- `TEST_SUITE_SUMMARY.md` - Detailed summary of test organization

