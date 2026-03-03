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

050_vector_types_exhaustive.t - Exhaustive vector type operations tests

=head1 DESCRIPTION

Comprehensive tests for vector type creation, validation, dimensions,
NULL handling, edge cases, and malformed input rejection.

Target: 100+ test cases

=cut

# Test plan: 2 setup + 3 neurondb_ok + 7 subtests + 2 cleanup = 14 top-level tests
plan tests => 14;

my $node = PostgresNode->new('vector_types_test');
ok($node, 'PostgresNode created');

$node->init();
$node->start();

ok($node->is_running(), 'PostgreSQL node started');

# Install extension
install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# ============================================================================
# VECTOR CREATION - String Format
# ============================================================================

subtest 'Vector Creation - String Format' => sub {
	plan tests => 25;
	
	# Basic integer vectors
	query_ok($node, 'postgres', q{SELECT '[1,2,3]'::vector(3);}, 
		'vector creation with integers');
	query_ok($node, 'postgres', q{SELECT '[1, 2, 3]'::vector(3);}, 
		'vector creation with spaces');
	
	# Float vectors
	query_ok($node, 'postgres', q{SELECT '[1.0, 2.0, 3.0]'::vector(3);}, 
		'vector creation with floats');
	query_ok($node, 'postgres', q{SELECT '[1.5, 2.5, 3.5]'::vector(3);}, 
		'vector creation with decimal floats');
	
	# Scientific notation
	query_ok($node, 'postgres', q{SELECT '[1e0, 2e0, 3e0]'::vector(3);}, 
		'vector creation with scientific notation');
	query_ok($node, 'postgres', q{SELECT '[1.5e1, 2.5e-1, 3.5e2]'::vector(3);}, 
		'vector creation with scientific notation variations');
	
	# Negative values
	query_ok($node, 'postgres', q{SELECT '[-1, -2, -3]'::vector(3);}, 
		'vector creation with negative values');
	query_ok($node, 'postgres', q{SELECT '[-1.5, 2.5, -3.5]'::vector(3);}, 
		'vector creation with mixed signs');
	
	# Zero values
	query_ok($node, 'postgres', q{SELECT '[0, 0, 0]'::vector(3);}, 
		'zero vector');
	query_ok($node, 'postgres', q{SELECT '[1, 0, 2]'::vector(3);}, 
		'vector with zero elements');
	
	# Various dimensions
	for my $dim (1, 2, 3, 4, 5, 10, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048) {
		my $vec = '[' . join(',', (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$vec'::vector($dim);", 
			"vector creation with dimension $dim");
	}
};

# ============================================================================
# VECTOR CREATION - Array Conversion
# ============================================================================

subtest 'Vector Creation - Array Conversion' => sub {
	plan tests => 10;
	
	# Array to vector
	query_ok($node, 'postgres', 
		q{SELECT array_to_vector(ARRAY[1.0, 2.0, 3.0]::real[]);}, 
		'array_to_vector with real array');
	query_ok($node, 'postgres', 
		q{SELECT array_to_vector(ARRAY[1, 2, 3]::integer[]);}, 
		'array_to_vector with integer array');
	
	# Vector to array
	query_ok($node, 'postgres', 
		q{SELECT vector_to_array('[1.0, 2.0, 3.0]'::vector(3));}, 
		'vector_to_array conversion');
	
	# Round-trip conversion
	query_ok($node, 'postgres', 
		q{SELECT array_to_vector(vector_to_array('[1.5, 2.5, 3.5]'::vector(3))::real[]);}, 
		'round-trip array conversion');
	
	# Various dimensions
	for my $dim (3, 5, 10, 128) {
		my @vals = map { $_ * 0.1 } (1..$dim);
		my $array_sql = 'ARRAY[' . join(',', @vals) . ']';
		query_ok($node, 'postgres', 
			"SELECT array_to_vector(${array_sql}::real[])::vector($dim);", 
			"array_to_vector with dimension $dim");
	}
};

# ============================================================================
# VECTOR DIMENSIONS
# ============================================================================

subtest 'Vector Dimensions' => sub {
	plan tests => 20;
	
	# Test vector_dims function
	for my $dim (1, 2, 3, 4, 5, 10, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192) {
		my $vec = '[' . join(',', (1..$dim)) . ']';
		result_is($node, 'postgres',
			"SELECT vector_dims('$vec'::vector);",
			"$dim",
			"vector_dims returns $dim for dimension $dim");
	}
};

# ============================================================================
# NULL HANDLING
# ============================================================================

subtest 'NULL Handling' => sub {
	plan tests => 8;
	
	# NULL vector creation
	query_ok($node, 'postgres', 
		q{SELECT NULL::vector(3);}, 
		'NULL vector creation');
	
	# NULL in operations
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) + NULL::vector(3);}, 
		'NULL in vector addition');
	query_ok($node, 'postgres', 
		q{SELECT NULL::vector(3) <-> '[1,2,3]'::vector(3);}, 
		'NULL in distance calculation');
	
	# NULL in functions
	query_ok($node, 'postgres', 
		q{SELECT vector_dims(NULL::vector);}, 
		'vector_dims with NULL');
	query_ok($node, 'postgres', 
		q{SELECT vector_norm(NULL::vector);}, 
		'vector_norm with NULL');
	
	# NULL in table
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS test_null_vectors;
		CREATE TABLE test_null_vectors (id SERIAL, vec vector(3));
		INSERT INTO test_null_vectors (vec) VALUES 
			('[1,2,3]'::vector(3)),
			(NULL),
			('[4,5,6]'::vector(3));
	});
	
	query_ok($node, 'postgres', 
		q{SELECT COUNT(*) FROM test_null_vectors WHERE vec IS NULL;}, 
		'NULL vector in table');
	
	result_is($node, 'postgres',
		q{SELECT COUNT(*) FROM test_null_vectors WHERE vec IS NULL;},
		'1',
		'correct NULL count');
	
	$node->psql('postgres', 'DROP TABLE test_null_vectors;');
};

# ============================================================================
# EDGE CASES
# ============================================================================

subtest 'Edge Cases' => sub {
	plan tests => 15;
	
	# Singleton vector
	query_ok($node, 'postgres', q{SELECT '[5]'::vector(1);}, 
		'singleton vector');
	
	# Very small values
	query_ok($node, 'postgres', q{SELECT '[1e-10, 2e-10, 3e-10]'::vector(3);}, 
		'vector with very small values');
	
	# Very large values
	query_ok($node, 'postgres', q{SELECT '[1e10, 2e10, 3e10]'::vector(3);}, 
		'vector with very large values');
	
	# Mixed precision
	query_ok($node, 'postgres', q{SELECT '[1, 2.5, 3e2]'::vector(3);}, 
		'vector with mixed precision');
	
	# Whitespace variations
	query_ok($node, 'postgres', q{SELECT '[ 1 , 2 , 3 ]'::vector(3);}, 
		'vector with extra whitespace');
	query_ok($node, 'postgres', q{SELECT '[-3.5,4.01e1 , 0 , 2e-2]'::vector(4);}, 
		'vector with whitespace and scientific notation');
	
	# Boundary dimensions
	for my $dim (1, 2, 3, 128, 384, 768, 1536, 2048) {
		my $vec = '[' . join(',', map { sprintf("%.6f", $_ * 0.001) } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$vec'::vector($dim);", 
			"edge case vector with dimension $dim");
	}
};

# ============================================================================
# MALFORMED INPUT REJECTION
# ============================================================================

subtest 'Malformed Input Rejection' => sub {
	plan tests => 20;
	
	# Empty vector (should fail)
	query_fails($node, 'postgres', q{SELECT '[]'::vector;}, 
		'empty vector rejected');
	
	# Dimension mismatch
	query_fails($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(2);}, 
		'dimension mismatch rejected');
	query_fails($node, 'postgres', 
		q{SELECT '[1,2]'::vector(3);}, 
		'dimension mismatch rejected (too few)');
	
	# Invalid syntax
	query_fails($node, 'postgres', q{SELECT 'invalid'::vector;}, 
		'invalid syntax rejected');
	query_fails($node, 'postgres', q{SELECT '[1,2,3'::vector(3);}, 
		'missing closing bracket rejected');
	query_fails($node, 'postgres', q{SELECT '1,2,3]'::vector(3);}, 
		'missing opening bracket rejected');
	
	# Invalid characters
	query_fails($node, 'postgres', q{SELECT '[a,b,c]'::vector(3);}, 
		'non-numeric values rejected');
	query_fails($node, 'postgres', q{SELECT '[1,2,3,4,5]'::vector(3);}, 
		'too many elements rejected');
	
	# Invalid array format
	query_fails($node, 'postgres', 
		q{SELECT array_to_vector(ARRAY[]::real[]);}, 
		'empty array rejected');
	
	# Type mismatches
	query_fails($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) + '[1,2]'::vector(2);}, 
		'dimension mismatch in addition rejected');
	query_fails($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <-> '[1,2]'::vector(2);}, 
		'dimension mismatch in distance rejected');
	
	# Invalid dimension specifications
	for my $invalid_dim (0, -1, 1000000) {
		my $vec = '[1,2,3]';
		my $result = $node->psql('postgres', 
			"SELECT '$vec'::vector($invalid_dim);");
		ok(!$result->{success}, "invalid dimension $invalid_dim rejected");
	}
};

# ============================================================================
# NUMERIC PRECISION
# ============================================================================

subtest 'Numeric Precision' => sub {
	plan tests => 12;
	
	# High precision floats
	query_ok($node, 'postgres', 
		q{SELECT '[1.123456789, 2.987654321, 3.141592653]'::vector(3);}, 
		'high precision floats');
	
	# Scientific notation precision
	query_ok($node, 'postgres', 
		q{SELECT '[1.5e-10, 2.5e10, 3.5e-5]'::vector(3);}, 
		'scientific notation with various exponents');
	
	# Precision preservation
	result_within_tolerance($node, 'postgres',
		q{SELECT vector_norm('[3,4]'::vector);},
		'5.0',
		0.0001,
		'vector norm precision');
	
	# Float32 precision limits
	for my $val (0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0) {
		query_ok($node, 'postgres', 
			"SELECT '[$val, $val, $val]'::vector(3);", 
			"precision test with value $val");
	}
};

# Cleanup
$node->stop();
ok(!$node->is_running(), 'PostgreSQL node stopped');

$node->cleanup();
ok(!-d $node->{data_dir}, 'Data directory cleaned up');

done_testing();



