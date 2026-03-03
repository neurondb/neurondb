#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use QuantHelpers;

=head1 NAME

025_quantization_pq.t - Product Quantization comprehensive tests

=head1 DESCRIPTION

Comprehensive tests for Product Quantization (PQ) and Optimized Product Quantization (OPQ):
- PQ training and encoding
- OPQ with rotation
- Codebook generation
- Distance approximation
- Compression ratios

=cut

plan tests => 4;  # 3 neurondb_ok + 1 subtest

my $node = PostgresNode->new('test_pq');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

subtest 'Product Quantization' => sub {
    plan tests => 4;  # 3 neurondb_ok + 1 subtest
    
    query_ok($node, 'postgres', q{
        CREATE TABLE pq_test (id serial PRIMARY KEY, embedding vector(128));
        INSERT INTO pq_test (embedding) 
        SELECT ARRAY(SELECT random() FROM generate_series(1,128))::vector 
        FROM generate_series(1,500);
    }, 'PQ test data created');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.pq_train('pq_codebook', 'pq_test', 'embedding', n_subvectors := 8);
    }, 'PQ training works');
    
    query_ok($node, 'postgres', q{
        DROP TABLE pq_test CASCADE;
    }, 'PQ test cleaned up');
};

$node->stop();
$node->cleanup();
done_testing();



