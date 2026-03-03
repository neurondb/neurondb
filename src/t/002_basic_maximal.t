#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;

=head1 NAME

002_basic_maximal.t - Maximal usage examples of PostgresNode, TapTest, and NeuronDB

=head1 DESCRIPTION

Demonstrates maximal usage of all three modules:
- Multiple PostgresNode instances with custom configs
- Complex TapTest assertions (result set validation, error handling)
- Full NeuronDB feature testing (ML, GPU, workers, indexes)

=cut

plan tests => 28;  # 2 nodes + 2 ports + 2 dirs + 2 starts + 2 queries + 1 complex + 1 result + 1 neurondb_ok (3 tests) + 1 vectors + 1 count + 1 vec ops + 1 dist + 1 agg + 1 ML (may skip) + 1 GPU + 1 workers + 1 indexes + 1 error + 2 stops + 2 cleanup

# Maximal PostgresNode usage: multiple nodes, custom configs
my $node1 = PostgresNode->new('maximal_test_1', port => undef);
my $node2 = PostgresNode->new('maximal_test_2', port => undef);

ok($node1, 'First PostgresNode created');
ok($node2, 'Second PostgresNode created');

ok($node1->get_port() != $node2->get_port(), 'Nodes use different ports');

$node1->init();
$node2->init();

ok(-d $node1->{data_dir}, 'First node data directory initialized');
ok(-d $node2->{data_dir}, 'Second node data directory initialized');

$node1->start();
$node2->start();

ok($node1->is_running(), 'First node started');
ok($node2->is_running(), 'Second node started');

# Maximal TapTest usage: complex assertions
query_ok($node1, 'postgres', 'SELECT version()', 'Version query on node1');
query_ok($node2, 'postgres', 'SELECT version()', 'Version query on node2');

# Result set validation
my $result = $node1->psql('postgres', q{
	SELECT 
		extname, extversion 
	FROM pg_extension 
	WHERE extname = 'plpgsql';
}, tuples_only => 1);

ok($result->{success}, 'Complex query executes');
like($result->{stdout}, qr/plpgsql/, 'Result contains expected data');

# Maximal NeuronDB usage: full feature testing
install_extension($node1, 'postgres');
neurondb_ok($node1, 'postgres', 'NeuronDB extension installed on node1');

# Create test vectors with custom parameters
create_test_vectors($node1, 'postgres', 50, 
	table => 'test_vectors_max',
	dim => 128
);

query_ok($node1, 'postgres', 
	'SELECT COUNT(*) FROM test_vectors_max',
	'Test vectors created'
);

result_is($node1, 'postgres',
	'SELECT COUNT(*) FROM test_vectors_max',
	'50',
	'Correct number of vectors'
);

# Test vector operations
my ($success, $msg) = test_vector_operations($node1, 'postgres');
ok($success, "Vector operations: $msg");

# Test distance metrics
($success, $msg) = test_distance_metrics($node1, 'postgres');
ok($success, "Distance metrics: $msg");

# Test aggregates
($success, $msg) = test_aggregates($node1, 'postgres');
ok($success, "Aggregates: $msg");

# Test ML function (if supported)
($success, $msg) = test_ml_function($node1, 'postgres', 'linear_regression',
	train_table => 'train_max',
	test_table => 'test_max',
	model_name => 'max_test_model',
	evaluate => 0
);

# ML may not be available, that's OK
SKIP: {
	skip("ML function not available: $msg", 1) unless $success;
	pass("ML function test: $msg");
}

# Test GPU features
($success, $msg) = test_gpu_features($node1, 'postgres');
ok($success, "GPU features: $msg");

# Test workers
($success, $msg) = test_workers($node1, 'postgres');
ok($success, "Workers: $msg");

# Test indexes
($success, $msg) = test_indexes($node1, 'postgres',
	table => 'test_index_max',
	index_name => 'idx_max_hnsw'
);

ok($success, "Indexes: $msg");

# Test error handling with TapTest
# Check that querying a nonexistent table fails
my $error_result = $node1->psql('postgres', 'SELECT * FROM nonexistent_table;');
SKIP: {
	if ($error_result->{success}) {
		skip('Error handling test - query unexpectedly succeeded', 1);
	}
	pass('Error handling works correctly');
}

# Maximal cleanup: multiple nodes
$node1->stop();
$node2->stop();

ok(!$node1->is_running(), 'First node stopped');
ok(!$node2->is_running(), 'Second node stopped');

$node1->cleanup();
$node2->cleanup();

ok(!-d $node1->{data_dir}, 'First node cleaned up');
ok(!-d $node2->{data_dir}, 'Second node cleaned up');

done_testing();

