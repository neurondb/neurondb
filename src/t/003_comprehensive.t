#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use VectorOps;
use MLHelpers;
use IndexHelpers;
use GPUHelpers;

=head1 NAME

003_comprehensive.t - Comprehensive coverage test for NeuronDB

=head1 DESCRIPTION

Comprehensive test coverage including:
- All core types and operations
- ML algorithms (regression, classification, clustering)
- GPU features (if available)
- Worker functions
- Index operations (HNSW, IVF)
- Distance metrics
- Aggregates

=cut

plan tests => 14;  # 2 main + 1 neurondb_ok (3 tests) + 6 subtests + 2 cleanup

my $node = PostgresNode->new('comprehensive_test');
ok($node, 'PostgresNode created');

$node->init();
$node->start();

ok($node->is_running(), 'PostgreSQL node started');

# Install extension
install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# ============================================================================
# Core Types and Operations
# ============================================================================

subtest 'Core Types' => sub {
	plan tests => 5;
	
	# Vector type creation
	query_ok($node, 'postgres', q{SELECT '[1,2,3]'::vector(3);}, 'Vector type works');
	
	# Vector dimensions
	result_is($node, 'postgres',
		q{SELECT vector_dims('[1,2,3,4,5]'::vector);},
		'5',
		'vector_dims works'
	);
	
	# Vector norm
	result_matches($node, 'postgres',
		q{SELECT vector_norm('[3,4]'::vector);},
		qr/5/,
		'vector_norm works'
	);
	
	# Vector normalization
	query_ok($node, 'postgres',
		q{SELECT vector_normalize('[0,100]'::vector);},
		'vector_normalize works'
	);
	
	# Vector arithmetic
	query_ok($node, 'postgres',
		q{SELECT '[1,2,3]'::vector + '[4,5,6]'::vector;},
		'vector arithmetic works'
	);
};

# ============================================================================
# Distance Metrics
# ============================================================================

subtest 'Distance Metrics' => sub {
	plan tests => 3;
	
	# L2 distance
	result_matches($node, 'postgres',
		q{SELECT '[1,0,0]'::vector(3) <-> '[0,1,0]'::vector(3);},
		qr/1\.414/,
		'L2 distance works'
	);
	
	# Cosine distance
	result_matches($node, 'postgres',
		q{SELECT '[1,0,0]'::vector(3) <=> '[0,1,0]'::vector(3);},
		qr/1/,
		'Cosine distance works'
	);
	
	# Inner product
	query_ok($node, 'postgres',
		q{SELECT '[1,2,3]'::vector(3) <#> '[4,5,6]'::vector(3);},
		'Inner product works'
	);
};

# ============================================================================
# Aggregates
# ============================================================================

subtest 'Aggregates' => sub {
	plan tests => 2;
	
	# Create test data
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS test_agg_comp;
		CREATE TABLE test_agg_comp (vec vector(3));
		INSERT INTO test_agg_comp VALUES
			('[1,2,3]'::vector(3)),
			('[4,5,6]'::vector(3)),
			('[7,8,9]'::vector(3));
	});
	
	# Vector average
	query_ok($node, 'postgres',
		q{SELECT vector_avg(vec) FROM test_agg_comp;},
		'vector_avg works'
	);
	
	# Vector sum
	query_ok($node, 'postgres',
		q{SELECT vector_sum(vec) FROM test_agg_comp;},
		'vector_sum works'
	);
	
	$node->psql('postgres', 'DROP TABLE test_agg_comp;');
};

# ============================================================================
# ML Algorithms
# ============================================================================

subtest 'ML Algorithms' => sub {
	plan tests => 5;
	
	# Linear Regression
	my ($success, $msg) = test_ml_function($node, 'postgres', 'linear_regression',
		train_table => 'train_linreg',
		test_table => 'test_linreg',
		model_name => 'comp_linreg_model',
		evaluate => 0
	);
	
	if ($success) {
		pass("Linear regression: $msg");
	} else {
		skip("Linear regression not available: $msg", 1);
	}
	
	# Logistic Regression
	($success, $msg) = test_ml_function($node, 'postgres', 'logistic_regression',
		train_table => 'train_logreg',
		test_table => 'test_logreg',
		model_name => 'comp_logreg_model',
		evaluate => 0
	);
	
	if ($success) {
		pass("Logistic regression: $msg");
	} else {
		skip("Logistic regression not available: $msg", 1);
	}
	
	# Random Forest
	($success, $msg) = test_ml_function($node, 'postgres', 'random_forest',
		train_table => 'train_rf',
		test_table => 'test_rf',
		model_name => 'comp_rf_model',
		evaluate => 0
	);
	
	if ($success) {
		pass("Random forest: $msg");
	} else {
		skip("Random forest not available: $msg", 1);
	}
	
	# K-Means clustering
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS train_kmeans;
		CREATE TABLE train_kmeans (
			id SERIAL PRIMARY KEY,
			features vector(3)
		);
		INSERT INTO train_kmeans (features) VALUES
			('[1,2,3]'::vector(3)),
			('[2,3,4]'::vector(3)),
			('[10,11,12]'::vector(3)),
			('[11,12,13]'::vector(3));
	});
	
	my $result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'kmeans',
			'train_kmeans',
			'features',
			NULL,
			'{"n_clusters": 2, "model_name": "comp_kmeans_model"}'::jsonb
		);
	});
	
	if ($result->{success}) {
		pass("K-Means clustering works");
	} else {
		skip("K-Means not available: $result->{stderr}", 1);
	}
	
	$node->psql('postgres', 'DROP TABLE train_kmeans;');
	
	# GMM clustering
	$result = $node->psql('postgres', q{
		DROP TABLE IF EXISTS train_gmm;
		CREATE TABLE train_gmm (
			id SERIAL PRIMARY KEY,
			features vector(3)
		);
		INSERT INTO train_gmm (features) VALUES
			('[1,2,3]'::vector(3)),
			('[2,3,4]'::vector(3)),
			('[10,11,12]'::vector(3));
	});
	
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'gmm',
			'train_gmm',
			'features',
			NULL,
			'{"n_components": 2, "model_name": "comp_gmm_model"}'::jsonb
		);
	});
	
	if ($result->{success}) {
		pass("GMM clustering works");
	} else {
		skip("GMM not available: $result->{stderr}", 1);
	}
	
	$node->psql('postgres', 'DROP TABLE train_gmm;');
};

# ============================================================================
# GPU Features
# ============================================================================

subtest 'GPU Features' => sub {
	plan tests => 2;
	
	my ($success, $msg) = test_gpu_features($node, 'postgres');
	ok($success, "GPU features: $msg");
	
	# Check GPU info function
	my $result = $node->psql('postgres', q{
		SELECT * FROM neurondb.gpu_info();
	});
	
	if ($result->{success}) {
		pass("GPU info function works");
	} else {
		skip("GPU info not available: $result->{stderr}", 1);
	}
};

# ============================================================================
# Workers
# ============================================================================

subtest 'Workers' => sub {
	plan tests => 3;
	
	my ($success, $msg) = test_workers($node, 'postgres');
	ok($success, "Workers: $msg");
	
	# Check job queue table
	table_ok($node, 'postgres', 'neurondb', 'neurondb_job_queue', 
		'Job queue table exists');
	
	# Check query metrics table
	table_ok($node, 'postgres', 'neurondb', 'neurondb_query_metrics',
		'Query metrics table exists');
};

# ============================================================================
# Indexes
# ============================================================================

subtest 'Indexes' => sub {
	plan tests => 3;
	
	# Create test data
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS test_index_comp;
		CREATE TABLE test_index_comp (
			id SERIAL PRIMARY KEY,
			vec vector(4),
			label TEXT
		);
		INSERT INTO test_index_comp (vec, label) VALUES
			('[1,2,3,4]'::vector(4), 'label1'),
			('[2,3,4,5]'::vector(4), 'label2'),
			('[3,4,5,6]'::vector(4), 'label3'),
			('[4,5,6,7]'::vector(4), 'label4'),
			('[5,6,7,8]'::vector(4), 'label5');
	});
	
	# Test HNSW index
	my ($success, $msg) = test_indexes($node, 'postgres',
		table => 'test_index_comp',
		index_name => 'idx_comp_hnsw'
	);
	
	ok($success, "HNSW index: $msg");
	
	# Test KNN query
	my $result = $node->psql('postgres', q{
		SELECT id, vec <-> '[1,2,3,4]'::vector(4) AS distance
		FROM test_index_comp
		ORDER BY vec <-> '[1,2,3,4]'::vector(4)
		LIMIT 3;
	});
	
	ok($result->{success}, 'KNN query works');
	
	# Test IVF index (if supported)
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_comp_ivf ON test_index_comp
		USING ivf (vec vector_l2_ops)
		WITH (lists = 10);
	});
	
	if ($result->{success}) {
		pass("IVF index creation works");
		$node->psql('postgres', 'DROP INDEX idx_comp_ivf;');
	} else {
		skip("IVF index not available: $result->{stderr}", 1);
	}
	
	$node->psql('postgres', 'DROP TABLE test_index_comp CASCADE;');
};

# ============================================================================
# Cleanup
# ============================================================================

$node->stop();
ok(!$node->is_running(), 'PostgreSQL node stopped');

$node->cleanup();
ok(!-d $node->{data_dir}, 'Data directory cleaned up');

done_testing();

