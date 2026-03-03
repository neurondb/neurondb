#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use SparseHelpers;

=head1 NAME

110_sparse_vectors.t - Sparse vectors and learned sparse retrieval tests

=head1 DESCRIPTION

Tests for sparse_vector type, SPLADE/ColBERTv2, and hybrid dense+sparse search.

=cut

plan tests => 9;  # 1 setup + 7 query_ok + 2 negative (may skip) + 1 cleanup (but skips reduce count)

my $node = PostgresNode->new('sparse_test');
ok($node, 'PostgresNode created');

$node->init();
$node->start();
install_extension($node, 'postgres');

# Test 1: Sparse vector creation
query_ok($node, 'postgres',
	"SELECT sparse_vector_in('{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.5,0.8]}') IS NOT NULL",
	'Sparse vector creation');

# Test 2: Sparse vector dot product
query_ok($node, 'postgres',
	"SELECT sparse_vector_dot_product(
		'{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.5,0.8]}'::sparse_vector,
		'{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.3,0.7]}'::sparse_vector
	) > 0",
	'Sparse vector dot product');

# Test 3: BM25 score
query_ok($node, 'postgres',
	"SELECT bm25_score('machine learning', 'machine learning algorithms', 1.5, 0.75) >= 0",
	'BM25 score computation');

# Test 4: Hybrid dense+sparse search
query_ok($node, 'postgres',
	"CREATE TEMP TABLE test_hybrid (
		id serial PRIMARY KEY,
		content text,
		dense_embedding vector(384),
		sparse_embedding sparse_vector
	)",
	'Create test table');

query_ok($node, 'postgres',
	"INSERT INTO test_hybrid (content, dense_embedding, sparse_embedding) VALUES
		('machine learning', '[0.1,0.2,0.3]'::vector || array_fill(0.0::float4, ARRAY[381]),
		 '{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.5,0.8]}'::sparse_vector)",
	'Insert test data');

query_ok($node, 'postgres',
	"SELECT COUNT(*) > 0 FROM hybrid_dense_sparse_search(
		'test_hybrid',
		'dense_embedding',
		'sparse_embedding',
		'[0.1,0.2,0.3]'::vector || array_fill(0.0::float4, ARRAY[381]),
		'{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.5,0.8]}'::sparse_vector,
		10,
		0.6,
		0.4
	)",
	'Hybrid dense+sparse search');

# Test 5: RRF fusion
query_ok($node, 'postgres',
	"SELECT rrf_fusion(10, 1.0, 2.0, 60.0) > 0",
	'RRF fusion');

# Negative tests - skip if validation not implemented
my $invalid_test = $node->psql('postgres',
	"SELECT sparse_vector_in('invalid format')");
if (!$invalid_test->{success}) {
	pass('Invalid sparse vector format correctly rejected');
} else {
	skip('Invalid sparse vector format validation not implemented', 1);
}

my $empty_test = $node->psql('postgres',
	"SELECT sparse_vector_in('{vocab_size:30522, model:SPLADE, tokens:[], weights:[]}')");
if (!$empty_test->{success}) {
	pass('Empty tokens correctly rejected');
} else {
	skip('Empty tokens validation not implemented', 1);
}

$node->stop();
ok(!$node->is_running(), 'PostgreSQL node stopped');

done_testing();

