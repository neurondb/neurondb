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

011_vector_arithmetic.t - Exhaustive vector arithmetic tests

=head1 DESCRIPTION

Comprehensive tests for vector arithmetic operations: addition, subtraction,
scalar multiplication/division, negation, concatenation, and boundary conditions.

Target: 80+ test cases

=cut

# Test plan: 3 neurondb_ok + 7 subtests = 10 top-level tests
plan tests => 10;

my $node = PostgresNode->new('vector_arithmetic_test');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# ============================================================================
# VECTOR ADDITION
# ============================================================================

subtest 'Vector Addition' => sub {
	plan tests => 15;
	
	# Basic addition
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) + '[4,5,6]'::vector(3);}, 
		'vector addition basic');
	
	# Addition with zeros
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) + '[0,0,0]'::vector(3);}, 
		'vector addition with zero vector');
	
	# Addition with negatives
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) + '[-1,-2,-3]'::vector(3);}, 
		'vector addition with negatives');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 128, 384) {
		my $v1 = '[' . join(',', (1..$dim)) . ']';
		my $v2 = '[' . join(',', map { $_ + 10 } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$v1'::vector($dim) + '$v2'::vector($dim);", 
			"vector addition dimension $dim");
	}
	
	# Dimension mismatch (should fail)
	query_fails($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) + '[1,2]'::vector(2);}, 
		'dimension mismatch in addition rejected');
};

# ============================================================================
# VECTOR SUBTRACTION
# ============================================================================

subtest 'Vector Subtraction' => sub {
	plan tests => 15;
	
	# Basic subtraction
	query_ok($node, 'postgres', 
		q{SELECT '[5,7,9]'::vector(3) - '[2,3,4]'::vector(3);}, 
		'vector subtraction basic');
	
	# Subtraction resulting in zero
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) - '[1,2,3]'::vector(3);}, 
		'vector subtraction to zero');
	
	# Subtraction with negatives
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) - '[4,5,6]'::vector(3);}, 
		'vector subtraction with negatives');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 128, 384) {
		my $v1 = '[' . join(',', map { $_ + 10 } (1..$dim)) . ']';
		my $v2 = '[' . join(',', (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$v1'::vector($dim) - '$v2'::vector($dim);", 
			"vector subtraction dimension $dim");
	}
	
	# Dimension mismatch (should fail)
	query_fails($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) - '[1,2]'::vector(2);}, 
		'dimension mismatch in subtraction rejected');
};

# ============================================================================
# SCALAR MULTIPLICATION
# ============================================================================

subtest 'Scalar Multiplication' => sub {
	plan tests => 15;
	
	# Basic multiplication
	query_ok($node, 'postgres', 
		q{SELECT '[2,3,4]'::vector(3) * 2.5;}, 
		'scalar multiplication basic');
	
	# Multiplication by zero
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) * 0;}, 
		'scalar multiplication by zero');
	
	# Multiplication by negative
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) * -1;}, 
		'scalar multiplication by negative');
	
	# Multiplication by small value
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) * 0.001;}, 
		'scalar multiplication by small value');
	
	# Multiplication by large value
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) * 1000;}, 
		'scalar multiplication by large value');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 128) {
		my $v = '[' . join(',', (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$v'::vector($dim) * 2.5;", 
			"scalar multiplication dimension $dim");
	}
	
	# Operator and function forms
	query_ok($node, 'postgres', 
		q{SELECT vector_mul('[2.0, 3.0]'::vector, 2.5);}, 
		'vector_mul function');
};

# ============================================================================
# SCALAR DIVISION
# ============================================================================

subtest 'Scalar Division' => sub {
	plan tests => 12;
	
	# Basic division
	query_ok($node, 'postgres', 
		q{SELECT '[6,9,12]'::vector(3) / 3.0;}, 
		'scalar division basic');
	
	# Division by small value
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) / 0.1;}, 
		'scalar division by small value');
	
	# Division by negative
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) / -2;}, 
		'scalar division by negative');
	
	# Division by zero (should fail or handle gracefully)
	my $result = $node->psql('postgres', 
		q{SELECT '[1,2,3]'::vector(3) / 0;});
	ok(!$result->{success} || $result->{stdout} =~ /inf|nan/i, 
		'division by zero handled');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 128) {
		my $v = '[' . join(',', map { $_ * 2 } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$v'::vector($dim) / 2.0;", 
			"scalar division dimension $dim");
	}
	
	# Function form
	query_ok($node, 'postgres', 
		q{SELECT vector_div('[6.0, 9.0]'::vector, 3.0);}, 
		'vector_div function');
};

# ============================================================================
# VECTOR NEGATION
# ============================================================================

subtest 'Vector Negation' => sub {
	plan tests => 10;
	
	# Basic negation
	query_ok($node, 'postgres', 
		q{SELECT -('[1,2,3]'::vector(3));}, 
		'vector negation operator');
	
	# Negation of negative
	query_ok($node, 'postgres', 
		q{SELECT -('[-1,-2,-3]'::vector(3));}, 
		'negation of negative vector');
	
	# Negation of zero
	query_ok($node, 'postgres', 
		q{SELECT -('[0,0,0]'::vector(3));}, 
		'negation of zero vector');
	
	# Function form
	query_ok($node, 'postgres', 
		q{SELECT vector_neg('[1.0, -2.0]'::vector);}, 
		'vector_neg function');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 128) {
		my $v = '[' . join(',', (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT -('$v'::vector($dim));", 
			"vector negation dimension $dim");
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
		'vector concatenation basic');
	
	# Concatenation with zeros
	query_ok($node, 'postgres', 
		q{SELECT vector_concat('[0,0]'::vector(2), '[1,2,3]'::vector(3));}, 
		'vector concatenation with zeros');
	
	# Various dimension combinations
	for my $d1 (1, 2, 3, 5) {
		for my $d2 (1, 2, 3, 5) {
			next if $d1 + $d2 > 10;  # Limit total size
			my $v1 = '[' . join(',', (1..$d1)) . ']';
			my $v2 = '[' . join(',', (1..$d2)) . ']';
			query_ok($node, 'postgres', 
				"SELECT vector_concat('$v1'::vector($d1), '$v2'::vector($d2));", 
				"vector concatenation $d1 + $d2");
		}
	}
};

# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================

subtest 'Boundary Conditions' => sub {
	plan tests => 13;
	
	# Very small values
	query_ok($node, 'postgres', 
		q{SELECT '[1e-10, 2e-10]'::vector(2) + '[1e-10, 2e-10]'::vector(2);}, 
		'addition with very small values');
	
	# Very large values
	query_ok($node, 'postgres', 
		q{SELECT '[1e10, 2e10]'::vector(2) * 0.5;}, 
		'multiplication with very large values');
	
	# Overflow scenarios
	my $large = '[1e20, 2e20]';
	query_ok($node, 'postgres', 
		"SELECT '$large'::vector(2) * 2;", 
		'potential overflow scenario');
	
	# Underflow scenarios
	my $small = '[1e-20, 2e-20]';
	query_ok($node, 'postgres', 
		"SELECT '$small'::vector(2) * 0.5;", 
		'potential underflow scenario');
	
	# Maximum dimensions
	for my $dim (128, 384, 768, 1536, 2048) {
		my $v1 = '[' . join(',', map { sprintf("%.6f", $_ * 0.001) } (1..$dim)) . ']';
		my $v2 = '[' . join(',', map { sprintf("%.6f", $_ * 0.002) } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$v1'::vector($dim) + '$v2'::vector($dim);", 
			"boundary test dimension $dim");
	}
};

$node->stop();
$node->cleanup();

done_testing();

