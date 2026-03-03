package SparseHelpers;

use strict;
use warnings;
use PostgresNode;
use TapTest;
use Exporter 'import';

our @EXPORT = qw(
	create_sparse_vector
	test_sparse_vector_ops
	create_sparse_index
	test_sparse_search
	test_hybrid_search
	test_bm25_score
);

=head1 NAME

SparseHelpers - Sparse vector test helpers

=head1 SYNOPSIS

  use SparseHelpers;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  create_sparse_vector($node, 'postgres', ...);
  test_sparse_search($node, 'postgres', ...);

=head1 DESCRIPTION

Provides test helpers for sparse vectors, SPLADE, ColBERT, and hybrid search.

=cut

=head2 create_sparse_vector

Create a sparse vector for testing.

=cut

sub create_sparse_vector {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $vocab_size = $params{vocab_size} || 30522;
	my $model = $params{model} || 'SPLADE';
	my $tokens = $params{tokens} || [100, 200, 300];
	my $weights = $params{weights} || [0.5, 0.8, 0.3];
	
	my $tokens_str = '[' . join(',', @$tokens) . ']';
	my $weights_str = '[' . join(',', @$weights) . ']';
	
	my $sparse_str = "{vocab_size:$vocab_size, model:$model, tokens:$tokens_str, weights:$weights_str}";
	
	my $result = $node->psql($dbname,
		"SELECT sparse_vector_in('$sparse_str') AS sv;"
	);
	
	if ($result->{success}) {
		return (1, $sparse_str);
	} else {
		return (0, "Sparse vector creation failed: $result->{stderr}");
	}
}

=head2 test_sparse_vector_ops

Test sparse vector operations.

=cut

sub test_sparse_vector_ops {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	my @results;
	
	# Create two sparse vectors
	my ($success1, $sv1) = create_sparse_vector($node, $dbname,
		tokens => [100, 200],
		weights => [0.5, 0.8]
	);
	my ($success2, $sv2) = create_sparse_vector($node, $dbname,
		tokens => [100, 200],
		weights => [0.3, 0.7]
	);
	
	unless ($success1 && $success2) {
		return (0, "Failed to create sparse vectors");
	}
	
	# Test dot product
	my $result = $node->psql($dbname, qq{
		SELECT sparse_vector_dot_product(
			'$sv1'::sparse_vector,
			'$sv2'::sparse_vector
		) AS dot_prod;
	}, tuples_only => 1);
	
	push @results, $result->{success};
	
	return \@results;
}

=head2 create_sparse_index

Create a sparse vector index.

=cut

sub create_sparse_index {
	my ($node, $dbname, $table, $column, %params) = @_;
	$dbname ||= 'postgres';
	
	my $index_name = $params{index_name} || "idx_${table}_${column}_sparse";
	my $min_freq = $params{min_freq} || 1;
	
	my $result = $node->psql($dbname, qq{
		SELECT sparse_index_create(
			'$table',
			'$column',
			'$index_name',
			$min_freq
		);
	});
	
	if ($result->{success}) {
		return (1, $index_name);
	} else {
		return (0, "Sparse index creation failed: $result->{stderr}");
	}
}

=head2 test_sparse_search

Test sparse vector search.

=cut

sub test_sparse_search {
	my ($node, $dbname, $index_name, $query_vec, %params) = @_;
	$dbname ||= 'postgres';
	
	my $k = $params{k} || 10;
	
	my $result = $node->psql($dbname, qq{
		SELECT * FROM sparse_index_search(
			'$index_name',
			'$query_vec'::sparse_vector,
			$k
		);
	});
	
	if ($result->{success}) {
		return (1, "Sparse search successful");
	} else {
		return (0, "Sparse search failed: $result->{stderr}");
	}
}

=head2 test_hybrid_search

Test hybrid dense+sparse search.

=cut

sub test_hybrid_search {
	my ($node, $dbname, $table, %params) = @_;
	$dbname ||= 'postgres';
	
	my $dense_col = $params{dense_col} || 'dense_embedding';
	my $sparse_col = $params{sparse_col} || 'sparse_embedding';
	my $query_dense = $params{query_dense} || '[0.1,0.2,0.3]';
	my $query_sparse = $params{query_sparse} || '{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.5,0.8]}';
	my $k = $params{k} || 10;
	my $alpha = $params{alpha} || 0.6;
	
	my $result = $node->psql($dbname, qq{
		SELECT * FROM hybrid_dense_sparse_search(
			'$table',
			'$dense_col',
			'$sparse_col',
			'$query_dense'::vector,
			'$query_sparse'::sparse_vector,
			$k,
			$alpha,
			@{[1 - $alpha]}
		);
	});
	
	if ($result->{success}) {
		return (1, "Hybrid search successful");
	} else {
		return (0, "Hybrid search failed: $result->{stderr}");
	}
}

=head2 test_bm25_score

Test BM25 scoring.

=cut

sub test_bm25_score {
	my ($node, $dbname, $query, $document, %params) = @_;
	$dbname ||= 'postgres';
	
	my $k1 = $params{k1} || 1.5;
	my $b = $params{b} || 0.75;
	
	my $result = $node->psql($dbname, qq{
		SELECT bm25_score('$query', '$document', $k1, $b) AS bm25;
	}, tuples_only => 1);
	
	if ($result->{success}) {
		return (1, "BM25 score computed");
	} else {
		return (0, "BM25 score failed: $result->{stderr}");
	}
}

1;



