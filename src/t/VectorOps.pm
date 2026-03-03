package VectorOps;

use strict;
use warnings;
use PostgresNode;
use TapTest;
use Exporter 'import';

our @EXPORT = qw(
	test_vector_creation
	test_vector_dimensions
	test_vector_arithmetic
	test_vector_comparison
	test_vector_functions
	test_distance_metrics_all
	generate_test_vector
	generate_test_vectors
	validate_vector_result
	test_vector_edge_cases
);

=head1 NAME

VectorOps - Vector operation test helpers

=head1 SYNOPSIS

  use VectorOps;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  test_vector_creation($node, 'postgres', 3);
  test_distance_metrics_all($node, 'postgres');

=head1 DESCRIPTION

Provides comprehensive test helpers for vector type operations,
distance metrics, arithmetic, and edge cases.

=cut

=head2 test_vector_creation

Test vector creation with various formats and dimensions.

=cut

sub test_vector_creation {
	my ($node, $dbname, $dim, %params) = @_;
	$dbname ||= 'postgres';
	$dim ||= 3;
	
	my @results;
	
	# Test string format
	my $vec_str = '[' . join(',', (1..$dim)) . ']';
	my $result = $node->psql($dbname, 
		"SELECT '$vec_str'::vector($dim) AS v;"
	);
	push @results, $result->{success};
	
	# Test array conversion
	my @array_vals = map { $_ * 0.1 } (1..$dim);
	my $array_sql = 'ARRAY[' . join(',', @array_vals) . ']';
	$result = $node->psql($dbname,
		"SELECT array_to_vector(${array_sql}::real[])::vector($dim);"
	);
	push @results, $result->{success};
	
	# Test with floats
	my $float_vec = '[' . join(',', map { sprintf("%.2f", $_ * 0.1) } (1..$dim)) . ']';
	$result = $node->psql($dbname,
		"SELECT '$float_vec'::vector($dim);"
	);
	push @results, $result->{success};
	
	# Test with scientific notation
	my $sci_vec = '[1.0e0,2.0e0,3.0e0]';
	$result = $node->psql($dbname,
		"SELECT '$sci_vec'::vector(3);"
	);
	push @results, $result->{success};
	
	return \@results;
}

=head2 test_vector_dimensions

Test vector dimension operations.

=cut

sub test_vector_dimensions {
	my ($node, $dbname, $dim) = @_;
	$dbname ||= 'postgres';
	$dim ||= 5;
	
	my $vec = '[' . join(',', (1..$dim)) . ']';
	my $result = $node->psql($dbname,
		"SELECT vector_dims('$vec'::vector) AS dims;",
		tuples_only => 1
	);
	
	if ($result->{success} && $result->{stdout} =~ /$dim/) {
		return 1;
	}
	
	return 0;
}

=head2 test_vector_arithmetic

Test all vector arithmetic operations.

=cut

sub test_vector_arithmetic {
	my ($node, $dbname, $dim) = @_;
	$dbname ||= 'postgres';
	$dim ||= 3;
	
	my @results;
	
	# Addition
	my $v1 = '[' . join(',', (1..$dim)) . ']';
	my $v2 = '[' . join(',', map { $_ + 10 } (1..$dim)) . ']';
	my $result = $node->psql($dbname,
		"SELECT '$v1'::vector($dim) + '$v2'::vector($dim) AS v_add;"
	);
	push @results, $result->{success};
	
	# Subtraction
	$result = $node->psql($dbname,
		"SELECT '$v2'::vector($dim) - '$v1'::vector($dim) AS v_sub;"
	);
	push @results, $result->{success};
	
	# Scalar multiplication
	$result = $node->psql($dbname,
		"SELECT '$v1'::vector($dim) * 2.5 AS v_mul;"
	);
	push @results, $result->{success};
	
	# Scalar division
	$result = $node->psql($dbname,
		"SELECT '$v1'::vector($dim) / 2.0 AS v_div;"
	);
	push @results, $result->{success};
	
	# Negation
	$result = $node->psql($dbname,
		"SELECT -('$v1'::vector($dim)) AS v_neg;"
	);
	push @results, $result->{success};
	
	# Concatenation
	if ($dim >= 2) {
		my $v3 = '[' . join(',', (1..2)) . ']';
		my $v4 = '[' . join(',', (3..$dim)) . ']';
		$result = $node->psql($dbname,
			"SELECT vector_concat('$v3'::vector(2), '$v4'::vector(" . ($dim-2) . "));"
		);
		push @results, $result->{success};
	}
	
	return \@results;
}

=head2 test_vector_comparison

Test vector comparison operators.

=cut

sub test_vector_comparison {
	my ($node, $dbname, $dim) = @_;
	$dbname ||= 'postgres';
	$dim ||= 3;
	
	my @results;
	
	my $v1 = '[1,2,3]';
	my $v2 = '[1,2,3]';
	my $v3 = '[1,2,4]';
	
	# Equality
	my $result = $node->psql($dbname,
		"SELECT '$v1'::vector(3) = '$v2'::vector(3) AS eq;",
		tuples_only => 1
	);
	push @results, $result->{success} && $result->{stdout} =~ /t/;
	
	# Inequality
	$result = $node->psql($dbname,
		"SELECT '$v1'::vector(3) != '$v3'::vector(3) AS ne;",
		tuples_only => 1
	);
	push @results, $result->{success} && $result->{stdout} =~ /t/;
	
	return \@results;
}

=head2 test_vector_functions

Test vector utility functions.

=cut

sub test_vector_functions {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	my @results;
	
	# vector_norm
	my $result = $node->psql($dbname,
		"SELECT vector_norm('[3,4]'::vector) AS norm;",
		tuples_only => 1
	);
	push @results, $result->{success} && $result->{stdout} =~ /5/;
	
	# vector_normalize
	$result = $node->psql($dbname,
		"SELECT vector_norm(vector_normalize('[0,100]'::vector)) AS norm_norm;",
		tuples_only => 1
	);
	push @results, $result->{success} && $result->{stdout} =~ /1/;
	
	# vector_to_array
	$result = $node->psql($dbname,
		"SELECT vector_to_array('[1,2,3]'::vector(3));"
	);
	push @results, $result->{success};
	
	# array_to_vector
	$result = $node->psql($dbname,
		"SELECT array_to_vector(ARRAY[1.0, 2.0, 3.0]::real[]);"
	);
	push @results, $result->{success};
	
	return \@results;
}

=head2 test_distance_metrics_all

Test all distance metrics comprehensively.

=cut

sub test_distance_metrics_all {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	my @results;
	
	# L2 distance (<->)
	my $result = $node->psql($dbname,
		"SELECT '[1,0,0]'::vector(3) <-> '[0,1,0]'::vector(3) AS l2;",
		tuples_only => 1
	);
	push @results, $result->{success};
	
	# Cosine distance (<=>)
	$result = $node->psql($dbname,
		"SELECT '[1,0,0]'::vector(3) <=> '[0,1,0]'::vector(3) AS cosine;",
		tuples_only => 1
	);
	push @results, $result->{success};
	
	# Inner product (<#>)
	$result = $node->psql($dbname,
		"SELECT '[1,2,3]'::vector(3) <#> '[4,5,6]'::vector(3) AS inner_prod;",
		tuples_only => 1
	);
	push @results, $result->{success};
	
	# Test with identical vectors
	$result = $node->psql($dbname,
		"SELECT '[1,2,3]'::vector(3) <-> '[1,2,3]'::vector(3) AS l2_identical;",
		tuples_only => 1
	);
	push @results, $result->{success} && $result->{stdout} =~ /^0/;
	
	# Test with orthogonal vectors (cosine should be 1)
	$result = $node->psql($dbname,
		"SELECT '[1,0]'::vector(2) <=> '[0,1]'::vector(2) AS cosine_ortho;",
		tuples_only => 1
	);
	push @results, $result->{success};
	
	return \@results;
}

=head2 generate_test_vector

Generate a test vector string for a given dimension.

=cut

sub generate_test_vector {
	my ($dim, %params) = @_;
	$dim ||= 3;
	
	my $start = $params{start} || 1;
	my $step = $params{step} || 1;
	my $format = $params{format} || '%.2f';
	
	my @coords;
	for my $i (0..$dim-1) {
		my $val = $start + $i * $step;
		push @coords, sprintf($format, $val);
	}
	
	return '[' . join(',', @coords) . ']';
}

=head2 generate_test_vectors

Generate multiple test vectors.

=cut

sub generate_test_vectors {
	my ($num_vectors, $dim, %params) = @_;
	$num_vectors ||= 10;
	$dim ||= 3;
	
	my @vectors;
	for my $i (1..$num_vectors) {
		push @vectors, generate_test_vector($dim, 
			start => $i * 0.1,
			step => 0.01,
			format => '%.6f'
		);
	}
	
	return \@vectors;
}

=head2 validate_vector_result

Validate that a vector result has expected properties.

=cut

sub validate_vector_result {
	my ($node, $dbname, $sql, $expected_dim, $test_name) = @_;
	$dbname ||= 'postgres';
	$test_name ||= "Vector result validation";
	
	my $result = $node->psql($dbname, $sql);
	
	unless ($result->{success}) {
		return (0, "Query failed: $result->{stderr}");
	}
	
	if ($expected_dim) {
		my $dim_result = $node->psql($dbname,
			"SELECT vector_dims(($sql)) AS dims;",
			tuples_only => 1
		);
		
		if ($dim_result->{success} && $dim_result->{stdout} =~ /$expected_dim/) {
			return (1, "Dimension matches");
		} else {
			return (0, "Dimension mismatch");
		}
	}
	
	return (1, "Validation passed");
}

=head2 test_vector_edge_cases

Test edge cases for vectors.

=cut

sub test_vector_edge_cases {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	my @results;
	
	# Zero vector
	my $zero_vec = '[0,0,0]';
	my $result = $node->psql($dbname,
		"SELECT '$zero_vec'::vector(3) AS zero;"
	);
	push @results, $result->{success};
	
	# Single dimension
	$result = $node->psql($dbname,
		"SELECT '[5]'::vector(1) AS single;"
	);
	push @results, $result->{success};
	
	# Large dimension
	my $large_dim = 128;
	my $large_vec = '[' . join(',', (1..$large_dim)) . ']';
	$result = $node->psql($dbname,
		"SELECT '$large_vec'::vector($large_dim) AS large;"
	);
	push @results, $result->{success};
	
	# Negative values
	my $neg_vec = '[-1,-2,-3]';
	$result = $node->psql($dbname,
		"SELECT '$neg_vec'::vector(3) AS neg;"
	);
	push @results, $result->{success};
	
	# Very small values
	my $small_vec = '[1e-10,2e-10,3e-10]';
	$result = $node->psql($dbname,
		"SELECT '$small_vec'::vector(3) AS small;"
	);
	push @results, $result->{success};
	
	return \@results;
}

1;

