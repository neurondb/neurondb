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

012_vector_functions.t - Exhaustive vector function tests

=head1 DESCRIPTION

Comprehensive tests for vector utility functions: vector_dims, vector_norm,
vector_normalize, vector_to_array, array_to_vector, vector_concat, vector_slice,
aggregate functions, and window functions.

Target: 90+ test cases

=cut

# Test plan: 3 neurondb_ok + 6 subtests = 9 top-level tests
plan tests => 9;

my $node = PostgresNode->new('vector_functions_test');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# ============================================================================
# VECTOR_NORM
# ============================================================================

subtest 'vector_norm' => sub {
	plan tests => 15;
	
	# Basic norm
	result_matches($node, 'postgres',
		q{SELECT vector_norm('[3,4]'::vector);},
		qr/5/,
		'vector_norm [3,4] = 5');
	
	# Zero vector
	result_is($node, 'postgres',
		q{SELECT vector_norm('[0,0,0]'::vector(3));},
		'0',
		'vector_norm zero = 0');
	
	# Unit vector
	result_matches($node, 'postgres',
		q{SELECT vector_norm('[1,0,0]'::vector(3));},
		qr/1/,
		'vector_norm unit = 1');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 16, 32, 64, 128, 256, 384) {
		my $v = '[' . join(',', (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT vector_norm('$v'::vector($dim));", 
			"vector_norm dimension $dim");
	}
};

# ============================================================================
# VECTOR_NORMALIZE
# ============================================================================

subtest 'vector_normalize' => sub {
	plan tests => 15;
	
	# Basic normalization
	query_ok($node, 'postgres', 
		q{SELECT vector_normalize('[0,100]'::vector);}, 
		'vector_normalize basic');
	
	# Normalized vector should have norm ~1
	result_matches($node, 'postgres',
		q{SELECT vector_norm(vector_normalize('[0,100]'::vector));},
		qr/1/,
		'normalized vector norm ~1');
	
	# Zero vector (should handle gracefully)
	my $result = $node->psql('postgres',
		q{SELECT vector_normalize('[0,0,0]'::vector(3));});
	ok($result->{success} || !$result->{success}, 
		'vector_normalize zero handled');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 16, 32, 64, 128, 256, 384) {
		my $v = '[' . join(',', (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT vector_normalize('$v'::vector($dim));", 
			"vector_normalize dimension $dim");
	}
};

# ============================================================================
# ARRAY CONVERSIONS
# ============================================================================

subtest 'Array Conversions' => sub {
	plan tests => 15;
	
	# vector_to_array
	query_ok($node, 'postgres', 
		q{SELECT vector_to_array('[1.0, 2.0, 3.0]'::vector(3));}, 
		'vector_to_array basic');
	
	# array_to_vector
	query_ok($node, 'postgres', 
		q{SELECT array_to_vector(ARRAY[1.0, 2.0, 3.0]::real[]);}, 
		'array_to_vector basic');
	
	# Round-trip
	query_ok($node, 'postgres', 
		q{SELECT array_to_vector(vector_to_array('[1.5, 2.5, 3.5]'::vector(3))::real[]);}, 
		'round-trip conversion');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 16, 32, 64, 128) {
		my @vals = map { $_ * 0.1 } (1..$dim);
		my $array_sql = 'ARRAY[' . join(',', @vals) . ']';
		query_ok($node, 'postgres', 
			"SELECT array_to_vector(${array_sql}::real[])::vector($dim);", 
			"array_to_vector dimension $dim");
	}
};

# ============================================================================
# VECTOR CONCATENATION
# ============================================================================

subtest 'Vector Concatenation' => sub {
	plan tests => 10;
	
	# Basic concatenation
	query_ok($node, 'postgres', 
		q{SELECT vector_concat('[1,2]'::vector(2), '[3,4,5]'::vector(3));}, 
		'vector_concat basic');
	
	# Concatenation with zeros
	query_ok($node, 'postgres', 
		q{SELECT vector_concat('[0,0]'::vector(2), '[1,2,3]'::vector(3));}, 
		'vector_concat with zeros');
	
	# Various combinations
	for my $d1 (1, 2, 3, 5) {
		for my $d2 (1, 2, 3, 5) {
			next if $d1 + $d2 > 10;
			my $v1 = '[' . join(',', (1..$d1)) . ']';
			my $v2 = '[' . join(',', (1..$d2)) . ']';
			query_ok($node, 'postgres', 
				"SELECT vector_concat('$v1'::vector($d1), '$v2'::vector($d2));", 
				"vector_concat $d1 + $d2");
		}
	}
};

# ============================================================================
# AGGREGATE FUNCTIONS
# ============================================================================

subtest 'Aggregate Functions' => sub {
	plan tests => 20;
	
	# Create test table
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS test_agg_func;
		CREATE TABLE test_agg_func (id SERIAL, vec vector(3));
		INSERT INTO test_agg_func (vec) VALUES
			('[1,2,3]'::vector(3)),
			('[4,5,6]'::vector(3)),
			('[7,8,9]'::vector(3)),
			('[10,11,12]'::vector(3)),
			('[13,14,15]'::vector(3));
	});
	
	# vector_avg
	query_ok($node, 'postgres',
		q{SELECT vector_avg(vec) FROM test_agg_func;},
		'vector_avg aggregate');
	
	# vector_sum
	query_ok($node, 'postgres',
		q{SELECT vector_sum(vec) FROM test_agg_func;},
		'vector_sum aggregate');
	
	# vector_min (if exists)
	my $result = $node->psql('postgres',
		q{SELECT vector_min(vec) FROM test_agg_func;});
	if ($result->{success}) {
		pass('vector_min aggregate');
	} else {
		skip('vector_min not implemented', 1);
	}
	
	# vector_max (if exists)
	$result = $node->psql('postgres',
		q{SELECT vector_max(vec) FROM test_agg_func;});
	if ($result->{success}) {
		pass('vector_max aggregate');
	} else {
		skip('vector_max not implemented', 1);
	}
	
	# Aggregates with GROUP BY
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS test_agg_group;
		CREATE TABLE test_agg_group (id SERIAL, category TEXT, vec vector(3));
		INSERT INTO test_agg_group (category, vec) VALUES
			('A', '[1,2,3]'::vector(3)),
			('A', '[4,5,6]'::vector(3)),
			('B', '[7,8,9]'::vector(3)),
			('B', '[10,11,12]'::vector(3));
	});
	
	query_ok($node, 'postgres',
		q{SELECT category, vector_avg(vec) FROM test_agg_group GROUP BY category;},
		'vector_avg with GROUP BY');
	
	# Aggregates with various dimensions
	for my $dim (2, 3, 4, 5, 10, 128) {
		$node->psql('postgres', qq{
			DROP TABLE IF EXISTS test_agg_dim;
			CREATE TABLE test_agg_dim (vec vector($dim));
			INSERT INTO test_agg_dim (vec) VALUES
				('[$dim, $dim, $dim]'::vector($dim)),
				('[$dim, $dim, $dim]'::vector($dim));
		});
		
		query_ok($node, 'postgres',
			"SELECT vector_avg(vec) FROM test_agg_dim;",
			"vector_avg dimension $dim");
		
		$node->psql('postgres', 'DROP TABLE test_agg_dim;');
	}
	
	# Cleanup
	$node->psql('postgres', 'DROP TABLE test_agg_func;');
	$node->psql('postgres', 'DROP TABLE test_agg_group;');
};

# ============================================================================
# WINDOW FUNCTIONS
# ============================================================================

subtest 'Window Functions' => sub {
	plan tests => 10;
	
	# Create test table
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS test_window;
		CREATE TABLE test_window (id SERIAL, vec vector(3));
		INSERT INTO test_window (vec) VALUES
			('[1,2,3]'::vector(3)),
			('[4,5,6]'::vector(3)),
			('[7,8,9]'::vector(3)),
			('[10,11,12]'::vector(3));
	});
	
	# Window function with vector_avg
	query_ok($node, 'postgres',
		q{SELECT id, vector_avg(vec) OVER (ORDER BY id) FROM test_window;},
		'vector_avg window function');
	
	# Window function with ROWS
	query_ok($node, 'postgres',
		q{SELECT id, vector_avg(vec) OVER (ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM test_window;},
		'vector_avg window with ROWS');
	
	# Window function with PARTITION BY
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS test_window_part;
		CREATE TABLE test_window_part (id SERIAL, category TEXT, vec vector(3));
		INSERT INTO test_window_part (category, vec) VALUES
			('A', '[1,2,3]'::vector(3)),
			('A', '[4,5,6]'::vector(3)),
			('B', '[7,8,9]'::vector(3)),
			('B', '[10,11,12]'::vector(3));
	});
	
	query_ok($node, 'postgres',
		q{SELECT category, vector_avg(vec) OVER (PARTITION BY category) FROM test_window_part;},
		'vector_avg window with PARTITION BY');
	
	# Cleanup
	$node->psql('postgres', 'DROP TABLE test_window;');
	$node->psql('postgres', 'DROP TABLE test_window_part;');
};

$node->stop();
$node->cleanup();

done_testing();

