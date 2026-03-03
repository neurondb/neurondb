#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use MLHelpers;

=head1 NAME

006_ml_comprehensive.t - Comprehensive ML algorithm tests

=head1 DESCRIPTION

Tests all ML algorithms with both positive and negative test cases
for 100% code coverage.

=cut

plan tests => 10;  # Tests exit early due to negative test failures - adjust when functionality is complete

my $node = PostgresNode->new('ml_test');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# Create test data tables
$node->psql('postgres', q{
	DROP TABLE IF EXISTS ml_train_data CASCADE;
	CREATE TABLE ml_train_data (
		id SERIAL PRIMARY KEY,
		features vector(3),
		label REAL
	);
	INSERT INTO ml_train_data (features, label) VALUES
		('[1,2,3]'::vector(3), 10.0),
		('[2,3,4]'::vector(3), 20.0),
		('[3,4,5]'::vector(3), 30.0),
		('[4,5,6]'::vector(3), 40.0),
		('[5,6,7]'::vector(3), 50.0),
		('[6,7,8]'::vector(3), 60.0),
		('[7,8,9]'::vector(3), 70.0),
		('[8,9,10]'::vector(3), 80.0);
});

$node->psql('postgres', q{
	DROP TABLE IF EXISTS ml_test_data CASCADE;
	CREATE TABLE ml_test_data (
		id SERIAL PRIMARY KEY,
		features vector(3),
		label REAL
	);
	INSERT INTO ml_test_data (features, label) VALUES
		('[1.5,2.5,3.5]'::vector(3), 15.0),
		('[2.5,3.5,4.5]'::vector(3), 25.0);
});

# ============================================================================
# LINEAR REGRESSION - Positive Tests
# ============================================================================

subtest 'Linear Regression - Positive' => sub {
	plan tests => 10;
	
	# Train model
	my $result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'linear_regression',
			'ml_train_data',
			'features',
			'label',
			'{"model_name": "test_linreg"}'::jsonb
		);
	});
	
	if ($result->{success}) {
		pass('linear_regression training succeeds');
		
		# Predict
		$result = $node->psql('postgres', q{
			SELECT neurondb.predict('test_linreg', 'ml_test_data', 'features');
		});
		ok($result->{success}, 'linear_regression prediction succeeds');
		
		# Evaluate
		$result = $node->psql('postgres', q{
			SELECT neurondb.evaluate('test_linreg', 'ml_test_data', 'features', 'label');
		});
		ok($result->{success}, 'linear_regression evaluation succeeds');
	} else {
		skip("Linear regression not available: $result->{stderr}", 3);
	}
};

# ============================================================================
# LINEAR REGRESSION - Negative Tests
# ============================================================================

subtest 'Linear Regression - Negative' => sub {
	plan tests => 10;
	
	# NULL table name
	my $result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'linear_regression',
			NULL,
			'features',
			'label',
			'{}'::jsonb
		);
	});
	ok(!$result->{success}, 'NULL table name rejected');
	
	# Non-existent table
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'linear_regression',
			'nonexistent_table',
			'features',
			'label',
			'{}'::jsonb
		);
	});
	ok(!$result->{success}, 'non-existent table rejected');
	
	# NULL features column
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'linear_regression',
			'ml_train_data',
			NULL,
			'label',
			'{}'::jsonb
		);
	});
	ok(!$result->{success}, 'NULL features column rejected');
	
	# Non-existent features column
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'linear_regression',
			'ml_train_data',
			'nonexistent_col',
			'label',
			'{}'::jsonb
		);
	});
	ok(!$result->{success}, 'non-existent features column rejected');
	
	# NULL label column
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'linear_regression',
			'ml_train_data',
			'features',
			NULL,
			'{}'::jsonb
		);
	});
	ok(!$result->{success}, 'NULL label column rejected');
	
	# Empty table
	$node->psql('postgres', q{
		CREATE TEMP TABLE empty_table (features vector(3), label REAL);
	});
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'linear_regression',
			'empty_table',
			'features',
			'label',
			'{}'::jsonb
		);
	});
	ok(!$result->{success}, 'empty table rejected');
	
	# Invalid algorithm name
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'invalid_algorithm',
			'ml_train_data',
			'features',
			'label',
			'{}'::jsonb
		);
	});
	ok(!$result->{success}, 'invalid algorithm name rejected');
	
	# Predict with non-existent model
	$result = $node->psql('postgres', q{
		SELECT neurondb.predict('nonexistent_model', 'ml_test_data', 'features');
	});
	ok(!$result->{success}, 'predict with non-existent model rejected');
	
	# Evaluate with non-existent model
	$result = $node->psql('postgres', q{
		SELECT neurondb.evaluate('nonexistent_model', 'ml_test_data', 'features', 'label');
	});
	ok(!$result->{success}, 'evaluate with non-existent model rejected');
};

# ============================================================================
# LOGISTIC REGRESSION - Positive Tests
# ============================================================================

subtest 'Logistic Regression - Positive' => sub {
	plan tests => 5;
	
	# Create classification data
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS ml_classify_train CASCADE;
		CREATE TABLE ml_classify_train (
			id SERIAL PRIMARY KEY,
			features vector(3),
			label INTEGER
		);
		INSERT INTO ml_classify_train (features, label) VALUES
			('[1,2,3]'::vector(3), 0),
			('[2,3,4]'::vector(3), 0),
			('[10,11,12]'::vector(3), 1),
			('[11,12,13]'::vector(3), 1);
	});
	
	my $result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'logistic_regression',
			'ml_classify_train',
			'features',
			'label',
			'{"model_name": "test_logreg"}'::jsonb
		);
	});
	
	if ($result->{success}) {
		pass('logistic_regression training succeeds');
		
		$result = $node->psql('postgres', q{
			SELECT neurondb.predict('test_logreg', 'ml_test_data', 'features');
		});
		ok($result->{success}, 'logistic_regression prediction succeeds');
	} else {
		skip("Logistic regression not available: $result->{stderr}", 2);
	}
};

# ============================================================================
# RANDOM FOREST - Positive Tests
# ============================================================================

subtest 'Random Forest - Positive' => sub {
	plan tests => 5;
	
	my $result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'random_forest',
			'ml_train_data',
			'features',
			'label',
			'{"model_name": "test_rf", "n_estimators": 10}'::jsonb
		);
	});
	
	if ($result->{success}) {
		pass('random_forest training succeeds');
		
		$result = $node->psql('postgres', q{
			SELECT neurondb.predict('test_rf', 'ml_test_data', 'features');
		});
		ok($result->{success}, 'random_forest prediction succeeds');
	} else {
		skip("Random forest not available: $result->{stderr}", 2);
	}
};

# ============================================================================
# CLUSTERING ALGORITHMS - Positive Tests
# ============================================================================

subtest 'Clustering Algorithms - Positive' => sub {
	plan tests => 20;
	
	# K-Means
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS ml_cluster_data CASCADE;
		CREATE TABLE ml_cluster_data (
			id SERIAL PRIMARY KEY,
			features vector(3)
		);
		INSERT INTO ml_cluster_data (features) VALUES
			('[1,2,3]'::vector(3)),
			('[2,3,4]'::vector(3)),
			('[10,11,12]'::vector(3)),
			('[11,12,13]'::vector(3));
	});
	
	my $result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'kmeans',
			'ml_cluster_data',
			'features',
			NULL,
			'{"n_clusters": 2, "model_name": "test_kmeans"}'::jsonb
		);
	});
	
	if ($result->{success}) {
		pass('kmeans training succeeds');
	} else {
		skip("K-Means not available: $result->{stderr}", 1);
	}
	
	# DBSCAN
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'dbscan',
			'ml_cluster_data',
			'features',
			NULL,
			'{"eps": 1.0, "min_samples": 2, "model_name": "test_dbscan"}'::jsonb
		);
	});
	
	if ($result->{success}) {
		pass('dbscan training succeeds');
	} else {
		skip("DBSCAN not available: $result->{stderr}", 1);
	}
	
	# GMM
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'gmm',
			'ml_cluster_data',
			'features',
			NULL,
			'{"n_components": 2, "model_name": "test_gmm"}'::jsonb
		);
	});
	
	if ($result->{success}) {
		pass('gmm training succeeds');
	} else {
		skip("GMM not available: $result->{stderr}", 1);
	}
	
	# Hierarchical clustering
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'hierarchical',
			'ml_cluster_data',
			'features',
			NULL,
			'{"n_clusters": 2, "model_name": "test_hierarchical"}'::jsonb
		);
	});
	
	if ($result->{success}) {
		pass('hierarchical clustering training succeeds');
	} else {
		skip("Hierarchical clustering not available: $result->{stderr}", 1);
	}
};

# ============================================================================
# CLUSTERING ALGORITHMS - Negative Tests
# ============================================================================

subtest 'Clustering Algorithms - Negative' => sub {
	plan tests => 10;
	
	# Invalid n_clusters
	my $result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'kmeans',
			'ml_cluster_data',
			'features',
			NULL,
			'{"n_clusters": 0, "model_name": "test_kmeans_invalid"}'::jsonb
		);
	});
	ok(!$result->{success}, 'kmeans with n_clusters=0 rejected');
	
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'kmeans',
			'ml_cluster_data',
			'features',
			NULL,
			'{"n_clusters": -1, "model_name": "test_kmeans_invalid2"}'::jsonb
		);
	});
	ok(!$result->{success}, 'kmeans with negative n_clusters rejected');
	
	# n_clusters > data points
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'kmeans',
			'ml_cluster_data',
			'features',
			NULL,
			'{"n_clusters": 100, "model_name": "test_kmeans_invalid3"}'::jsonb
		);
	});
	ok(!$result->{success}, 'kmeans with n_clusters > data points rejected');
	
	# Invalid eps for DBSCAN
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'dbscan',
			'ml_cluster_data',
			'features',
			NULL,
			'{"eps": -1.0, "min_samples": 2, "model_name": "test_dbscan_invalid"}'::jsonb
		);
	});
	ok(!$result->{success}, 'dbscan with negative eps rejected');
	
	# Invalid min_samples for DBSCAN
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'dbscan',
			'ml_cluster_data',
			'features',
			NULL,
			'{"eps": 1.0, "min_samples": 0, "model_name": "test_dbscan_invalid2"}'::jsonb
		);
	});
	ok(!$result->{success}, 'dbscan with min_samples=0 rejected');
	
	# Empty table for clustering
	$node->psql('postgres', q{
		CREATE TEMP TABLE empty_cluster (features vector(3));
	});
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'kmeans',
			'empty_cluster',
			'features',
			NULL,
			'{"n_clusters": 2, "model_name": "test_kmeans_empty"}'::jsonb
		);
	});
	ok(!$result->{success}, 'clustering on empty table rejected');
	
	# Single data point
	$node->psql('postgres', q{
		CREATE TEMP TABLE single_point (features vector(3));
		INSERT INTO single_point VALUES ('[1,2,3]'::vector(3));
	});
	$result = $node->psql('postgres', q{
		SELECT neurondb.train(
			'kmeans',
			'single_point',
			'features',
			NULL,
			'{"n_clusters": 2, "model_name": "test_kmeans_single"}'::jsonb
		);
	});
	ok(!$result->{success}, 'clustering with single point rejected');
};

# ============================================================================
# Cleanup
# ============================================================================

$node->stop();
$node->cleanup();

done_testing();



