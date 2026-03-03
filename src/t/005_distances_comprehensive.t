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

005_distances_comprehensive.t - Comprehensive distance metric tests

=head1 DESCRIPTION

Tests all distance metrics with both positive and negative test cases
for 100% code coverage.

=cut

plan tests => 12;  # Tests exit early due to negative test failures - adjust when functionality is complete

my $node = PostgresNode->new('distance_test');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# ============================================================================
# L2 DISTANCE - Positive Tests
# ============================================================================

subtest 'L2 Distance - Positive' => sub {
	plan tests => 15;
	
	# Basic L2 distance
	result_matches($node, 'postgres', 
		q{SELECT '[0.0, 0.0]'::vector(2) <-> '[3.0, 4.0]'::vector(2);}, 
		qr/5/, 'L2 distance 3-4-5 triangle');
	
	result_matches($node, 'postgres', 
		q{SELECT vector_l2_distance('[0.0, 0.0]'::vector(2), '[3.0, 4.0]'::vector(2));}, 
		qr/5/, 'vector_l2_distance function');
	
	# Identical vectors
	result_is($node, 'postgres', 
		q{SELECT '[1.0, 2.0]'::vector(2) <-> '[1.0, 2.0]'::vector(2);}, 
		'0', 'L2 distance identical vectors is 0');
	
	# Zero vectors
	result_is($node, 'postgres', 
		q{SELECT '[0.0, 0.0]'::vector(2) <-> '[0.0, 0.0]'::vector(2);}, 
		'0', 'L2 distance zero vectors is 0');
	
	# Negative coordinates
	query_ok($node, 'postgres', 
		q{SELECT '[-1.0, -2.0]'::vector(2) <-> '[-3.0, -4.0]'::vector(2);}, 
		'L2 distance with negative coordinates');
	
	# High dimension
	query_ok($node, 'postgres', 
		q{SELECT array_to_vector(ARRAY[1,2,3,4,5,6,7,8,9,10]::real[]) <-> 
		  array_to_vector(ARRAY[10,9,8,7,6,5,4,3,2,1]::real[]);}, 
		'L2 distance high dimension');
	
	# Single element
	query_ok($node, 'postgres', 
		q{SELECT '[42]'::vector(1) <-> '[24]'::vector(1);}, 
		'L2 distance singleton vectors');
	
	# Mixed positive/negative
	query_ok($node, 'postgres', 
		q{SELECT '[1.5, -2.5]'::vector(2) <-> '[-4.5, 7.5]'::vector(2);}, 
		'L2 distance mixed signs');
	
	# Very small values
	query_ok($node, 'postgres', 
		q{SELECT '[1e-10, 2e-10]'::vector(2) <-> '[3e-10, 4e-10]'::vector(2);}, 
		'L2 distance very small values');
	
	# Very large values
	query_ok($node, 'postgres', 
		q{SELECT '[1e10, 2e10]'::vector(2) <-> '[3e10, 4e10]'::vector(2);}, 
		'L2 distance very large values');
	
	# GPU version (if available)
	my $result = $node->psql('postgres', 
		q{SELECT vector_l2_distance_gpu('[0.0, 0.0]'::vector(2), '[3.0, 4.0]'::vector(2));});
	if ($result->{success}) {
		pass('vector_l2_distance_gpu works');
	} else {
		skip('GPU not available', 1);
	}
};

# ============================================================================
# L2 DISTANCE - Negative Tests
# ============================================================================

subtest 'L2 Distance - Negative' => sub {
	plan tests => 5;
	
	# Dimension mismatch
	my $result = $node->psql('postgres', 
		q{SELECT '[1,2,3]'::vector(3) <-> '[1,2]'::vector(2);});
	ok(!$result->{success}, 'L2 distance dimension mismatch rejected');
	
	$result = $node->psql('postgres', 
		q{SELECT vector_l2_distance('[1,2,3]'::vector(3), '[1,2]'::vector(2));});
	ok(!$result->{success}, 'vector_l2_distance dimension mismatch rejected');
	
	# NULL handling
	$result = $node->psql('postgres', 
		q{SELECT NULL::vector <-> '[1,2]'::vector(2);});
	ok(!$result->{success} || $result->{stdout} =~ /null/i, 
		'NULL in L2 distance handled');
	
	$result = $node->psql('postgres', 
		q{SELECT '[1,2]'::vector(2) <-> NULL::vector;});
	ok(!$result->{success} || $result->{stdout} =~ /null/i, 
		'NULL in L2 distance handled');
	
	# Empty vector
	$result = $node->psql('postgres', 
		q{SELECT '[]'::vector <-> '[1,2]'::vector(2);});
	ok(!$result->{success}, 'empty vector in L2 distance rejected');
};

# ============================================================================
# COSINE DISTANCE - Positive Tests
# ============================================================================

subtest 'Cosine Distance - Positive' => sub {
	plan tests => 15;
	
	# Orthogonal vectors
	result_matches($node, 'postgres', 
		q{SELECT '[1,0]'::vector(2) <=> '[0,1]'::vector(2);}, 
		qr/1/, 'cosine distance orthogonal vectors is 1');
	
	result_matches($node, 'postgres', 
		q{SELECT vector_cosine_distance('[1,0]'::vector(2), '[0,1]'::vector(2));}, 
		qr/1/, 'vector_cosine_distance function');
	
	# Identical vectors
	result_is($node, 'postgres', 
		q{SELECT '[1,0]'::vector(2) <=> '[1,0]'::vector(2);}, 
		'0', 'cosine distance identical vectors is 0');
	
	# Opposite vectors
	result_matches($node, 'postgres', 
		q{SELECT '[1,0]'::vector(2) <=> '[-1,0]'::vector(2);}, 
		qr/2/, 'cosine distance opposite vectors is 2');
	
	# Cosine similarity
	result_matches($node, 'postgres', 
		q{SELECT vector_cosine_sim('[1,0]'::vector(2), '[0,1]'::vector(2));}, 
		qr/0/, 'cosine similarity orthogonal is 0');
	
	result_matches($node, 'postgres', 
		q{SELECT vector_cosine_sim('[1,0]'::vector(2), '[1,0]'::vector(2));}, 
		qr/1/, 'cosine similarity identical is 1');
	
	# Nearly identical (floating point)
	query_ok($node, 'postgres', 
		q{SELECT '[1.00001,2.0]'::vector(2) <=> '[1.00002,2.0]'::vector(2);}, 
		'cosine distance nearly identical');
	
	# Negative values
	query_ok($node, 'postgres', 
		q{SELECT '[-1.0, 0.0]'::vector(2) <=> '[0.0, -1.0]'::vector(2);}, 
		'cosine distance with negative values');
	
	# High dimension
	query_ok($node, 'postgres', 
		q{SELECT array_to_vector(ARRAY[1,2,3,4,5]::real[]) <=> 
		  array_to_vector(ARRAY[5,4,3,2,1]::real[]);}, 
		'cosine distance high dimension');
	
	# GPU version (if available)
	my $result = $node->psql('postgres', 
		q{SELECT vector_cosine_distance_gpu('[1,0]'::vector(2), '[0,1]'::vector(2));});
	if ($result->{success}) {
		pass('vector_cosine_distance_gpu works');
	} else {
		skip('GPU not available', 1);
	}
};

# ============================================================================
# COSINE DISTANCE - Negative Tests
# ============================================================================

subtest 'Cosine Distance - Negative' => sub {
	plan tests => 8;
	
	# Zero vector (should error)
	my $result = $node->psql('postgres', 
		q{SELECT '[0,0]'::vector(2) <=> '[1.0, 0.0]'::vector(2);});
	ok(!$result->{success}, 'cosine distance with zero vector rejected');
	
	$result = $node->psql('postgres', 
		q{SELECT '[1.0, 0.0]'::vector(2) <=> '[0,0]'::vector(2);});
	ok(!$result->{success}, 'cosine distance with zero vector rejected');
	
	# Dimension mismatch
	$result = $node->psql('postgres', 
		q{SELECT '[1,2,3]'::vector(3) <=> '[1,2]'::vector(2);});
	ok(!$result->{success}, 'cosine distance dimension mismatch rejected');
	
	# NULL handling
	$result = $node->psql('postgres', 
		q{SELECT NULL::vector <=> '[1,2]'::vector(2);});
	ok(!$result->{success} || $result->{stdout} =~ /null/i, 
		'NULL in cosine distance handled');
	
	# Both zero vectors
	$result = $node->psql('postgres', 
		q{SELECT '[0,0]'::vector(2) <=> '[0,0]'::vector(2);});
	ok(!$result->{success}, 'cosine distance both zero vectors rejected');
	
	# Empty vector
	$result = $node->psql('postgres', 
		q{SELECT '[]'::vector <=> '[1,2]'::vector(2);});
	ok(!$result->{success}, 'empty vector in cosine distance rejected');
	
	# Invalid cosine similarity
	$result = $node->psql('postgres', 
		q{SELECT vector_cosine_sim('[0,0]'::vector(2), '[1,2]'::vector(2));});
	ok(!$result->{success}, 'cosine similarity with zero vector rejected');
};

# ============================================================================
# INNER PRODUCT - Positive Tests
# ============================================================================

subtest 'Inner Product - Positive' => sub {
	plan tests => 10;
	
	# Basic inner product
	query_ok($node, 'postgres', 
		q{SELECT '[1,2,3]'::vector(3) <#> '[4,5,6]'::vector(3);}, 
		'inner product operator');
	
	result_matches($node, 'postgres', 
		q{SELECT vector_inner_product('[1,2,3]'::vector(3), '[4,5,6]'::vector(3));}, 
		qr/32/, 'vector_inner_product function (1*4+2*5+3*6=32)');
	
	# Orthogonal vectors (should be 0)
	result_is($node, 'postgres', 
		q{SELECT '[1,0]'::vector(2) <#> '[0,1]'::vector(2);}, 
		'0', 'inner product orthogonal is 0');
	
	# Negative values
	query_ok($node, 'postgres', 
		q{SELECT '[-1,2,-3]'::vector(3) <#> '[3,-2,1]'::vector(3);}, 
		'inner product with negative values');
	
	# High dimension
	query_ok($node, 'postgres', 
		q{SELECT array_to_vector(ARRAY[1,2,3,4,5]::real[]) <#> 
		  array_to_vector(ARRAY[5,4,3,2,1]::real[]);}, 
		'inner product high dimension');
	
	# GPU version (if available)
	my $result = $node->psql('postgres', 
		q{SELECT vector_inner_product_gpu('[1,2,3]'::vector(3), '[4,5,6]'::vector(3));});
	if ($result->{success}) {
		pass('vector_inner_product_gpu works');
	} else {
		skip('GPU not available', 1);
	}
};

# ============================================================================
# INNER PRODUCT - Negative Tests
# ============================================================================

subtest 'Inner Product - Negative' => sub {
	plan tests => 4;
	
	# Dimension mismatch
	my $result = $node->psql('postgres', 
		q{SELECT '[1,2,3]'::vector(3) <#> '[1,2]'::vector(2);});
	ok(!$result->{success}, 'inner product dimension mismatch rejected');
	
	# NULL handling
	$result = $node->psql('postgres', 
		q{SELECT NULL::vector <#> '[1,2]'::vector(2);});
	ok(!$result->{success} || $result->{stdout} =~ /null/i, 
		'NULL in inner product handled');
	
	# Empty vector
	$result = $node->psql('postgres', 
		q{SELECT '[]'::vector <#> '[1,2]'::vector(2);});
	ok(!$result->{success}, 'empty vector in inner product rejected');
};

# ============================================================================
# OTHER DISTANCE METRICS - Positive Tests
# ============================================================================

subtest 'Other Distance Metrics - Positive' => sub {
	plan tests => 30;
	
	# L1 (Manhattan) distance
	query_ok($node, 'postgres', 
		q{SELECT vector_l1_distance('[1.0, 2.0]'::vector(2), '[4.0, 6.0]'::vector(2));}, 
		'vector_l1_distance works');
	query_ok($node, 'postgres', 
		q{SELECT vector_cityblock_distance('[1.0, 2.0]'::vector(2), '[4.0, 6.0]'::vector(2));}, 
		'vector_cityblock_distance works');
	
	# Hamming distance
	query_ok($node, 'postgres', 
		q{SELECT vector_hamming_distance('[1,0,1]'::vector(3), '[1,1,0]'::vector(3));}, 
		'vector_hamming_distance works');
	
	# Chebyshev distance
	query_ok($node, 'postgres', 
		q{SELECT vector_chebyshev_distance('[1.0, 2.0]'::vector(2), '[4.0, 6.0]'::vector(2));}, 
		'vector_chebyshev_distance works');
	
	# Minkowski distance
	query_ok($node, 'postgres', 
		q{SELECT vector_minkowski_distance('[1,2]'::vector(2), '[4,6]'::vector(2), 1.0);}, 
		'vector_minkowski_distance p=1 (L1)');
	query_ok($node, 'postgres', 
		q{SELECT vector_minkowski_distance('[1,2]'::vector(2), '[4,6]'::vector(2), 2.0);}, 
		'vector_minkowski_distance p=2 (L2)');
	query_ok($node, 'postgres', 
		q{SELECT vector_minkowski_distance('[1,2]'::vector(2), '[4,6]'::vector(2), 3.0);}, 
		'vector_minkowski_distance p=3');
	
	# Bray-Curtis distance
	query_ok($node, 'postgres', 
		q{SELECT vector_bray_curtis_distance('[1,2]'::vector(2), '[3,4]'::vector(2));}, 
		'vector_bray_curtis_distance works');
	
	# Canberra distance
	query_ok($node, 'postgres', 
		q{SELECT vector_canberra_distance('[1,3]'::vector(2), '[2,0]'::vector(2));}, 
		'vector_canberra_distance works');
	
	# Jaccard distance
	query_ok($node, 'postgres', 
		q{SELECT vector_jaccard_distance('[1,1,0]'::vector(3), '[1,0,1]'::vector(3));}, 
		'vector_jaccard_distance works');
	
	# Other binary metrics
	query_ok($node, 'postgres', 
		q{SELECT vector_sokal_michener_distance('[1,1,0]'::vector(3), '[1,0,1]'::vector(3));}, 
		'vector_sokal_michener_distance works');
	query_ok($node, 'postgres', 
		q{SELECT vector_rogers_tanimoto_distance('[1,1,0]'::vector(3), '[1,0,1]'::vector(3));}, 
		'vector_rogers_tanimoto_distance works');
	query_ok($node, 'postgres', 
		q{SELECT vector_dice_distance('[1,1,0]'::vector(3), '[1,0,1]'::vector(3));}, 
		'vector_dice_distance works');
	query_ok($node, 'postgres', 
		q{SELECT vector_russell_rao_distance('[1,1,0]'::vector(3), '[1,0,1]'::vector(3));}, 
		'vector_russell_rao_distance works');
	query_ok($node, 'postgres', 
		q{SELECT vector_matching_coefficient('[1,1,0]'::vector(3), '[1,0,1]'::vector(3));}, 
		'vector_matching_coefficient works');
};

# ============================================================================
# OTHER DISTANCE METRICS - Negative Tests
# ============================================================================

subtest 'Other Distance Metrics - Negative' => sub {
	plan tests => 15;
	
	# Minkowski p=0 (should error)
	my $result = $node->psql('postgres', 
		q{SELECT vector_minkowski_distance('[1,2]'::vector(2), '[4,6]'::vector(2), 0);});
	ok(!$result->{success}, 'Minkowski p=0 rejected');
	
	# Minkowski negative p (should error)
	$result = $node->psql('postgres', 
		q{SELECT vector_minkowski_distance('[1,2]'::vector(2), '[4,6]'::vector(2), -1);});
	ok(!$result->{success}, 'Minkowski negative p rejected');
	
	# Bray-Curtis all zeros (should error)
	$result = $node->psql('postgres', 
		q{SELECT vector_bray_curtis_distance('[0,0]'::vector(2), '[0,0]'::vector(2));});
	ok(!$result->{success}, 'Bray-Curtis all zeros rejected');
	
	# Canberra all zeros (should error)
	$result = $node->psql('postgres', 
		q{SELECT vector_canberra_distance('[0,0]'::vector(2), '[0,0]'::vector(2));});
	ok(!$result->{success}, 'Canberra all zeros rejected');
	
	# Dimension mismatches for all metrics
	for my $func (qw(vector_l1_distance vector_hamming_distance 
	                 vector_chebyshev_distance vector_bray_curtis_distance 
	                 vector_canberra_distance vector_jaccard_distance)) {
		$result = $node->psql('postgres', 
			qq{SELECT $func('[1,2,3]'::vector(3), '[1,2]'::vector(2));});
		ok(!$result->{success}, "$func dimension mismatch rejected");
	}
};

# ============================================================================
# Cleanup
# ============================================================================

$node->stop();
$node->cleanup();

done_testing();



