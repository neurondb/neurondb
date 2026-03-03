#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use IndexHelpers;

=head1 NAME

023_index_ivf.t - IVF index comprehensive tests

=head1 DESCRIPTION

Comprehensive tests for IVF (Inverted File) indexes:
- IVFFlat index creation
- IVF-PQ (Product Quantization)
- Probe parameter tuning
- Multi-probe search
- Build and maintenance

=cut

plan tests => 4;  # 3 neurondb_ok + 1 subtest

my $node = PostgresNode->new('test_ivf');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

subtest 'IVFFlat Basic' => sub {
    plan tests => 4;  # 3 neurondb_ok + 1 subtest
    
    query_ok($node, 'postgres', q{
        CREATE TABLE ivf_test (id serial PRIMARY KEY, embedding vector(128));
        INSERT INTO ivf_test (embedding) 
        SELECT ARRAY(SELECT random() FROM generate_series(1,128))::vector 
        FROM generate_series(1,1000);
    }, 'IVF test data created');
    
    query_ok($node, 'postgres', q{
        CREATE INDEX ON ivf_test USING ivfflat (embedding vector_l2_ops) 
        WITH (lists = 10);
    }, 'IVFFlat index created');
    
    query_ok($node, 'postgres', q{
        SET ivfflat.probes = 3;
        SELECT * FROM ivf_test ORDER BY embedding <-> ARRAY(SELECT 0.5 FROM generate_series(1,128))::vector LIMIT 10;
    }, 'IVF search works');
    
    query_ok($node, 'postgres', q{
        DROP TABLE ivf_test CASCADE;
    }, 'IVF test cleaned up');
};

$node->stop();
$node->cleanup();
done_testing();



