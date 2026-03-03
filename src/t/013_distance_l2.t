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

013_distance_l2.t - Exhaustive distance metrics tests

=head1 DESCRIPTION

Comprehensive tests for all distance metrics: L2, cosine, inner product,
Manhattan, Hamming, Jaccard, and custom metrics.

Target: 120+ test cases

=cut

# Test plan: 3 neurondb_ok + 7 subtests = 10 top-level tests
plan tests => 10;

my $node = PostgresNode->new('distance_metrics_test');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# ============================================================================
# L2 DISTANCE (<->)
# ============================================================================

subtest 'L2 Distance (<->)' => sub {
	plan tests => 25;
	
	# Orthogonal vectors (should be sqrt(2) for unit vectors)
	query_ok($node, 'postgres', 
		q{SELECT '[1,0,0]'::vector(3) <-> '[0,1,0]'::vector(3) AS l2;}, 
		'L2 distance orthogonal vectors');
	
	result_matches($node, 'postgres',
		q{SELECT '[1,0,0]'::vector(3) <-> '[0,1,0]'::vector(3);},
		qr/1\.414/,
		'L2 distance orthogonal ~1.414');
	
	# Identical vectors (should be 0)
	result_is($node, 'postgres',
		q{SELECT '[1,2,3]'::vector(3) <-> '[1,2,3]'::vector(3);},
		'0',
		'L2 distance identical vectors = 0');
	
	# Parallel vectors
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <-> '[2,4,6]'::vector(3) AS l2;}, 
		'L2 distance parallel vectors');
	
	# Zero vector
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <-> '[0,0,0]'::vector(3) AS l2;}, 
		'L2 distance to zero vector');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 16, 32, 64, 128, 256, 384) {
		my $v1 = '[' . join(',', (1..$dim)) . ']';
		my $v2 = '[' . join(',', map { $_ + 1 } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$v1'::vector($dim) <-> '$v2'::vector($dim);", 
			"L2 distance dimension $dim");
	}
	
	# Negative values
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <-> '[-1,-2,-3]'::vector(3) AS l2;}, 
		'L2 distance with negatives');
	
	# Large values
	query_ok($node, 'postgres', 
		q{SELECT '[1e10,2e10]'::vector(2) <-> '[2e10,3e10]'::vector(2) AS l2;}, 
		'L2 distance with large values');
};

# ============================================================================
# COSINE DISTANCE (<=>)
# ============================================================================

subtest 'Cosine Distance (<=>)' => sub {
	plan tests => 25;
	
	# Identical vectors (should be 0)
	result_is($node, 'postgres',
		q{SELECT '[1,2,3]'::vector(3) <=> '[1,2,3]'::vector(3);},
		'0',
		'cosine distance identical = 0');
	
	# Orthogonal vectors (should be 1)
	result_matches($node, 'postgres',
		q{SELECT '[1,0,0]'::vector(3) <=> '[0,1,0]'::vector(3);},
		qr/1/,
		'cosine distance orthogonal = 1');
	
	# Opposite vectors (should be 1)
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <=> '[-1,-2,-3]'::vector(3) AS cosine;}, 
		'cosine distance opposite vectors');
	
	# Parallel vectors (should be 0)
	result_is($node, 'postgres',
		q{SELECT '[1,2,3]'::vector(3) <=> '[2,4,6]'::vector(3);},
		'0',
		'cosine distance parallel = 0');
	
	# Zero vector (should handle gracefully)
	my $result = $node->psql('postgres',
		q{SELECT '[1,2,3]'::vector(3) <=> '[0,0,0]'::vector(3);});
	ok($result->{success}, 'cosine distance with zero vector');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 16, 32, 64, 128, 256, 384) {
		my $v1 = '[' . join(',', (1..$dim)) . ']';
		my $v2 = '[' . join(',', map { $_ + 1 } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$v1'::vector($dim) <=> '$v2'::vector($dim);", 
			"cosine distance dimension $dim");
	}
	
	# Normalized vectors
	query_ok($node, 'postgres', 
		q{SELECT vector_normalize('[1,2,3]'::vector) <=> vector_normalize('[2,4,6]'::vector);}, 
		'cosine distance normalized vectors');
};

# ============================================================================
# INNER PRODUCT (<#>)
# ============================================================================

subtest 'Inner Product (<#>)' => sub {
	plan tests => 20;
	
	# Basic inner product
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <#> '[4,5,6]'::vector(3) AS inner;}, 
		'inner product basic');
	
	# Inner product with zeros
	result_is($node, 'postgres',
		q{SELECT '[1,2,3]'::vector(3) <#> '[0,0,0]'::vector(3);},
		'0',
		'inner product with zero = 0');
	
	# Inner product with negatives
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <#> '[-1,-2,-3]'::vector(3) AS inner;}, 
		'inner product with negatives');
	
	# Orthogonal vectors (should be 0)
	result_is($node, 'postgres',
		q{SELECT '[1,0,0]'::vector(3) <#> '[0,1,0]'::vector(3);},
		'0',
		'inner product orthogonal = 0');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 16, 32, 64, 128, 256, 384) {
		my $v1 = '[' . join(',', (1..$dim)) . ']';
		my $v2 = '[' . join(',', map { $_ + 10 } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$v1'::vector($dim) <#> '$v2'::vector($dim);", 
			"inner product dimension $dim");
	}
};

# ============================================================================
# MANHATTAN DISTANCE
# ============================================================================

subtest 'Manhattan Distance' => sub {
	plan tests => 15;
	
	# Basic Manhattan distance
	query_ok($node, 'postgres', 
		q{SELECT vector_manhattan_distance('[1,2,3]'::vector(3), '[4,5,6]'::vector(3));}, 
		'Manhattan distance basic');
	
	# Manhattan distance identical
	result_is($node, 'postgres',
		q{SELECT vector_manhattan_distance('[1,2,3]'::vector(3), '[1,2,3]'::vector(3));},
		'0',
		'Manhattan distance identical = 0');
	
	# Manhattan distance with negatives
	query_ok($node, 'postgres', 
		q{SELECT vector_manhattan_distance('[1,2,3]'::vector(3), '[-1,-2,-3]'::vector(3));}, 
		'Manhattan distance with negatives');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 16, 32, 64, 128) {
		my $v1 = '[' . join(',', (1..$dim)) . ']';
		my $v2 = '[' . join(',', map { $_ + 1 } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT vector_manhattan_distance('$v1'::vector($dim), '$v2'::vector($dim));", 
			"Manhattan distance dimension $dim");
	}
};

# ============================================================================
# HAMMING DISTANCE
# ============================================================================

subtest 'Hamming Distance' => sub {
	plan tests => 10;
	
	# Hamming distance for binary vectors
	query_ok($node, 'postgres', 
		q{SELECT vector_hamming_distance('[1,0,1]'::vector(3), '[0,1,1]'::vector(3));}, 
		'Hamming distance basic');
	
	# Hamming distance identical
	result_is($node, 'postgres',
		q{SELECT vector_hamming_distance('[1,0,1]'::vector(3), '[1,0,1]'::vector(3));},
		'0',
		'Hamming distance identical = 0');
	
	# Various dimensions
	for my $dim (2, 3, 4, 5, 10, 16, 32, 64) {
		my $v1 = '[' . join(',', map { $_ % 2 } (1..$dim)) . ']';
		my $v2 = '[' . join(',', map { ($_ + 1) % 2 } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT vector_hamming_distance('$v1'::vector($dim), '$v2'::vector($dim));", 
			"Hamming distance dimension $dim");
	}
};

# ============================================================================
# JACCARD DISTANCE
# ============================================================================

subtest 'Jaccard Distance' => sub {
	plan tests => 10;
	
	# Jaccard distance
	query_ok($node, 'postgres', 
		q{SELECT vector_jaccard_distance('[1,0,1,1]'::vector(4), '[0,1,1,1]'::vector(4));}, 
		'Jaccard distance basic');
	
	# Jaccard distance identical
	result_is($node, 'postgres',
		q{SELECT vector_jaccard_distance('[1,0,1]'::vector(3), '[1,0,1]'::vector(3));},
		'0',
		'Jaccard distance identical = 0');
	
	# Various dimensions
	for my $dim (3, 4, 5, 10, 16, 32, 64) {
		my $v1 = '[' . join(',', map { $_ % 2 } (1..$dim)) . ']';
		my $v2 = '[' . join(',', map { ($_ + 1) % 2 } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT vector_jaccard_distance('$v1'::vector($dim), '$v2'::vector($dim));", 
			"Jaccard distance dimension $dim");
	}
};

# ============================================================================
# DISTANCE METRIC EDGE CASES
# ============================================================================

subtest 'Distance Metric Edge Cases' => sub {
	plan tests => 25;
	
	# Dimension mismatch (should fail)
	query_fails($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <-> '[1,2]'::vector(2);}, 
		'L2 distance dimension mismatch rejected');
	query_fails($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <=> '[1,2]'::vector(2);}, 
		'cosine distance dimension mismatch rejected');
	query_fails($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <#> '[1,2]'::vector(2);}, 
		'inner product dimension mismatch rejected');
	
	# NULL handling
	query_ok($node, 'postgres', 
		q{SELECT NULL::vector(3) <-> '[1,2,3]'::vector(3);}, 
		'L2 distance with NULL');
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <=> NULL::vector(3);}, 
		'cosine distance with NULL');
	
	# Very small distances
	my $small = '[1e-10, 2e-10]';
	query_ok($node, 'postgres', 
		"SELECT '$small'::vector(2) <-> '$small'::vector(2);", 
		'L2 distance very small values');
	
	# Very large distances
	my $large1 = '[1e10, 2e10]';
	my $large2 = '[2e10, 3e10]';
	query_ok($node, 'postgres', 
		"SELECT '$large1'::vector(2) <-> '$large2'::vector(2);", 
		'L2 distance very large values');
	
	# Maximum dimensions
	for my $dim (128, 256, 384, 768, 1536, 2048) {
		my $v1 = '[' . join(',', map { sprintf("%.6f", $_ * 0.001) } (1..$dim)) . ']';
		my $v2 = '[' . join(',', map { sprintf("%.6f", ($_ + 1) * 0.001) } (1..$dim)) . ']';
		query_ok($node, 'postgres', 
			"SELECT '$v1'::vector($dim) <-> '$v2'::vector($dim);", 
			"L2 distance edge case dimension $dim");
		query_ok($node, 'postgres', 
			"SELECT '$v1'::vector($dim) <=> '$v2'::vector($dim);", 
			"cosine distance edge case dimension $dim");
	}
};

$node->stop();
$node->cleanup();

done_testing();

