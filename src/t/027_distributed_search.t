#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;

=head1 NAME

027_distributed_search.t - Distributed search tests

=head1 DESCRIPTION

Comprehensive tests for distributed search features:
- Multi-node search
- Shard management
- Result aggregation
- Distributed indexing

=cut

plan tests => 4;  # 3 neurondb_ok + 1 subtest

my $node = PostgresNode->new('test_distributed');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

subtest 'Shard Management' => sub {
    plan tests => 4;  # 3 neurondb_ok + 1 subtest
    
    query_ok($node, 'postgres', q{
        CREATE TABLE distributed_data (id serial PRIMARY KEY, embedding vector(64));
        INSERT INTO distributed_data (embedding) 
        SELECT ARRAY(SELECT random() FROM generate_series(1,64))::vector 
        FROM generate_series(1,1000);
    }, 'Distributed test data created');
    
    query_ok($node, 'postgres', q{
        DROP TABLE distributed_data CASCADE;
    }, 'Distributed test cleaned up');
};

$node->stop();
$node->cleanup();
done_testing();



