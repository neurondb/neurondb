package QuantHelpers;

use strict;
use warnings;
use PostgresNode;
use TapTest;
use Exporter 'import';

our @EXPORT = qw(
	test_pq_quantization
	test_opq_quantization
	test_fp8_quantization
	test_int8_quantization
	test_quantization_accuracy
	get_compression_ratio
);

=head1 NAME

QuantHelpers - Quantization test helpers

=head1 SYNOPSIS

  use QuantHelpers;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  test_pq_quantization($node, 'postgres', ...);
  test_quantization_accuracy($node, 'postgres', ...);

=head1 DESCRIPTION

Provides test helpers for vector quantization (PQ, OPQ, FP8, INT8).

=cut

=head2 test_pq_quantization

Test Product Quantization.

=cut

sub test_pq_quantization {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $m = $params{m} || 8;
	my $nbits = $params{nbits} || 8;
	my $table = $params{table} || 'test_vectors';
	my $column = $params{column} || 'vec';
	
	# Create PQ quantizer
	my $result = $node->psql($dbname, qq{
		SELECT neurondb.train_quantizer(
			'pq',
			'$table',
			'$column',
			'{"m": $m, "nbits": $nbits}'::jsonb
		);
	});
	
	if ($result->{success}) {
		return (1, "PQ quantization trained");
	} else {
		return (0, "PQ quantization failed: $result->{stderr}");
	}
}

=head2 test_opq_quantization

Test Optimized Product Quantization.

=cut

sub test_opq_quantization {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $m = $params{m} || 8;
	my $nbits = $params{nbits} || 8;
	my $table = $params{table} || 'test_vectors';
	my $column = $params{column} || 'vec';
	
	# Create OPQ quantizer
	my $result = $node->psql($dbname, qq{
		SELECT neurondb.train_quantizer(
			'opq',
			'$table',
			'$column',
			'{"m": $m, "nbits": $nbits}'::jsonb
		);
	});
	
	if ($result->{success}) {
		return (1, "OPQ quantization trained");
	} else {
		return (0, "OPQ quantization failed: $result->{stderr}");
	}
}

=head2 test_fp8_quantization

Test FP8 quantization.

=cut

sub test_fp8_quantization {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $table = $params{table} || 'test_vectors';
	my $column = $params{column} || 'vec';
	
	# Test FP8 quantization
	my $result = $node->psql($dbname, qq{
		SELECT neurondb.quantize_fp8(
			'$table',
			'$column'
		);
	});
	
	if ($result->{success}) {
		return (1, "FP8 quantization successful");
	} else {
		return (0, "FP8 quantization failed: $result->{stderr}");
	}
}

=head2 test_int8_quantization

Test INT8 quantization.

=cut

sub test_int8_quantization {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $table = $params{table} || 'test_vectors';
	my $column = $params{column} || 'vec';
	
	# Test INT8 quantization
	my $result = $node->psql($dbname, qq{
		SELECT neurondb.quantize_int8(
			'$table',
			'$column'
		);
	});
	
	if ($result->{success}) {
		return (1, "INT8 quantization successful");
	} else {
		return (0, "INT8 quantization failed: $result->{stderr}");
	}
}

=head2 test_quantization_accuracy

Test quantization accuracy (reconstruction error).

=cut

sub test_quantization_accuracy {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $table = $params{table} || 'test_vectors';
	my $column = $params{column} || 'vec';
	my $quantized_col = $params{quantized_col} || 'vec_quantized';
	
	# Calculate reconstruction error
	my $result = $node->psql($dbname, qq{
		SELECT AVG(
			vector_l2_distance(
				$column,
				dequantize($quantized_col)
			)
		) AS reconstruction_error
		FROM $table;
	}, tuples_only => 1);
	
	if ($result->{success}) {
		my $error = $result->{stdout};
		chomp $error;
		$error =~ s/^\s+|\s+$//g;
		return (1, "Reconstruction error: $error");
	} else {
		return (0, "Accuracy test failed: $result->{stderr}");
	}
}

=head2 get_compression_ratio

Get compression ratio for quantized vectors.

=cut

sub get_compression_ratio {
	my ($node, $dbname, $table, $original_col, $quantized_col) = @_;
	$dbname ||= 'postgres';
	
	# Get sizes
	my $original = $node->psql($dbname, qq{
		SELECT pg_column_size($original_col) FROM $table LIMIT 1;
	}, tuples_only => 1);
	
	my $quantized = $node->psql($dbname, qq{
		SELECT pg_column_size($quantized_col) FROM $table LIMIT 1;
	}, tuples_only => 1);
	
	if ($original->{success} && $quantized->{success}) {
		my $orig_size = $original->{stdout};
		my $quant_size = $quantized->{stdout};
		chomp $orig_size;
		chomp $quant_size;
		$orig_size =~ s/^\s+|\s+$//g;
		$quant_size =~ s/^\s+|\s+$//g;
		
		if ($quant_size > 0) {
			my $ratio = $orig_size / $quant_size;
			return (1, "Compression ratio: $ratio");
		}
	}
	
	return (0, "Could not calculate compression ratio");
}

1;



