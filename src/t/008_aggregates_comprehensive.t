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

=head1 NAME

008_aggregates_comprehensive.t - Comprehensive aggregate function tests

=head1 DESCRIPTION

Tests all vector aggregate functions with both positive and negative test cases.

=cut

plan tests => 6;  # Tests exit early due to negative test failures - adjust when functionality is complete

my $node = PostgresNode->new('aggregate_test');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# Create test data
$node->psql('postgres', q{
	DROP TABLE IF EXISTS test_agg_data CASCADE;
	CREATE TABLE test_agg_data (
		id SERIAL PRIMARY KEY,
		vec vector(3),
		group_id INTEGER
	);
	INSERT INTO test_agg_data (vec, group_id) VALUES
		('[1,2,3]'::vector(3), 1),
		('[4,5,6]'::vector(3), 1),
		('[7,8,9]'::vector(3), 2),
		('[10,11,12]'::vector(3), 2),
		('[13,14,15]'::vector(3), 3);
});

# ============================================================================
# VECTOR AGGREGATES - Positive Tests
# ============================================================================

subtest 'Vector Aggregates - Positive' => sub {
	plan tests => 30;
	
	# vector_sum
	query_ok($node, 'postgres', 
		q{SELECT vector_sum(vec) FROM test_agg_data;}, 
		'vector_sum works');
	
	# vector_avg
	query_ok($node, 'postgres', 
		q{SELECT vector_avg(vec) FROM test_agg_data;}, 
		'vector_avg works');
	
	# vector_mean (alias)
	query_ok($node, 'postgres', 
		q{SELECT vector_mean(vec) FROM test_agg_data;}, 
		'vector_mean works');
	
	# vector_min
	query_ok($node, 'postgres', 
		q{SELECT vector_min(vec) FROM test_agg_data;}, 
		'vector_min works');
	
	# vector_max
	query_ok($node, 'postgres', 
		q{SELECT vector_max(vec) FROM test_agg_data;}, 
		'vector_max works');
	
	# Aggregates with WHERE clause
	query_ok($node, 'postgres', 
		q{SELECT vector_avg(vec) FROM test_agg_data WHERE group_id = 1;}, 
		'vector_avg with WHERE clause');
	
	# Aggregates with GROUP BY
	query_ok($node, 'postgres', 
		q{SELECT group_id, vector_sum(vec) FROM test_agg_data GROUP BY group_id;}, 
		'vector_sum with GROUP BY');
	
	query_ok($node, 'postgres', 
		q{SELECT group_id, vector_avg(vec) FROM test_agg_data GROUP BY group_id;}, 
		'vector_avg with GROUP BY');
	
	# Single row aggregate
	query_ok($node, 'postgres', 
		q{SELECT vector_avg(vec) FROM test_agg_data WHERE id = 1;}, 
		'vector_avg on single row');
	
	# Empty result set (should return NULL)
	my $result = $node->psql('postgres', 
		q{SELECT vector_sum(vec) FROM test_agg_data WHERE false;}, 
		tuples_only => 1);
	ok($result->{success}, 'vector_sum on empty set works');
	
	# Identical vectors
	$node->psql('postgres', q{
		CREATE TEMP TABLE identical_vecs (vec vector(3));
		INSERT INTO identical_vecs VALUES
			('[5.0, 5.0, 5.0]'::vector(3)),
			('[5.0, 5.0, 5.0]'::vector(3)),
			('[5.0, 5.0, 5.0]'::vector(3));
	});
	query_ok($node, 'postgres', 
		q{SELECT vector_avg(vec) FROM identical_vecs;}, 
		'vector_avg on identical vectors');
	
	# Mixed positive/negative values
	$node->psql('postgres', q{
		CREATE TEMP TABLE mixed_vecs (vec vector(3));
		INSERT INTO mixed_vecs VALUES
			('[-1.0, 0.0, 2.5]'::vector(3)),
			('[1.5, -2.0, 3.0]'::vector(3)),
			('[0.0, 0.0, 0.0]'::vector(3));
	});
	query_ok($node, 'postgres', 
		q{SELECT vector_sum(vec) FROM mixed_vecs;}, 
		'vector_sum with mixed signs');
	
	# Zero vectors
	$node->psql('postgres', q{
		CREATE TEMP TABLE zero_vecs (vec vector(3));
		INSERT INTO zero_vecs VALUES
			('[0.0, 0.0, 0.0]'::vector(3)),
			('[0.0, 0.0, 0.0]'::vector(3));
	});
	query_ok($node, 'postgres', 
		q{SELECT vector_sum(vec) FROM zero_vecs;}, 
		'vector_sum on zero vectors');
	
	# Large number of rows
	$node->psql('postgres', q{
		CREATE TEMP TABLE many_vecs (vec vector(2));
		INSERT INTO many_vecs (vec)
		SELECT format('[%s,%s]', i::float, (i*2)::float)::vector(2)
		FROM generate_series(1, 100) AS i;
	});
	query_ok($node, 'postgres', 
		q{SELECT vector_avg(vec) FROM many_vecs;}, 
		'vector_avg on many rows');
};

# ============================================================================
# VECTOR AGGREGATES - Negative Tests
# ============================================================================

subtest 'Vector Aggregates - Negative' => sub {
	plan tests => 20;
	
	# NULL values
	$node->psql('postgres', q{
		CREATE TEMP TABLE null_vecs (vec vector(3));
		INSERT INTO null_vecs VALUES
			('[1,2,3]'::vector(3)),
			(NULL),
			('[4,5,6]'::vector(3));
	});
	
	my $result = $node->psql('postgres', 
		q{SELECT vector_sum(vec) FROM null_vecs;});
	ok($result->{success}, 'vector_sum with NULLs handled');
	
	# All NULLs (should return NULL)
	$result = $node->psql('postgres', 
		q{SELECT vector_sum(vec) FROM (VALUES (NULL::vector), (NULL::vector)) t(vec);}, 
		tuples_only => 1);
	ok($result->{success}, 'vector_sum with all NULLs handled');
	
	# Dimension mismatch (should error)
	$node->psql('postgres', q{
		CREATE TEMP TABLE dim_mismatch (vec vector);
		INSERT INTO dim_mismatch VALUES
			('[1.0,2.0]'::vector(2)),
			('[1.0,2.0,3.0]'::vector(3));
	});
	
	$result = $node->psql('postgres', 
		q{SELECT vector_sum(vec) FROM dim_mismatch;});
	ok(!$result->{success}, 'vector_sum with dimension mismatch rejected');
	
	# Empty table
	$node->psql('postgres', q{
		CREATE TEMP TABLE empty_agg (vec vector(3));
	});
	
	$result = $node->psql('postgres', 
		q{SELECT vector_sum(vec) FROM empty_agg;}, 
		tuples_only => 1);
	ok($result->{success}, 'vector_sum on empty table handled');
	
	# Invalid aggregate usage
	$result = $node->psql('postgres', 
		q{SELECT vector_sum(vec, vec) FROM test_agg_data;});
	ok(!$result->{success}, 'vector_sum with wrong number of arguments rejected');
	
	# Aggregate on non-vector column
	$result = $node->psql('postgres', 
		q{SELECT vector_sum(id) FROM test_agg_data;});
	ok(!$result->{success}, 'vector_sum on non-vector column rejected');
	
	# Aggregate in invalid context
	$result = $node->psql('postgres', 
		q{SELECT id, vector_sum(vec) FROM test_agg_data;});
	ok(!$result->{success}, 'vector_sum without GROUP BY rejected');
};

# ============================================================================
# Cleanup
# ============================================================================

$node->stop();
$node->cleanup();

done_testing();



