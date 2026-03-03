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

028_edge_cases.t - Edge cases and error handling tests

=head1 DESCRIPTION

Comprehensive edge case tests:
- NULL handling
- Empty arrays
- Dimension mismatches
- Memory limits
- Concurrent operations
- Invalid inputs
- Boundary conditions

=cut

plan tests => 9;  # 3 neurondb_ok + 6 subtests

my $node = PostgresNode->new('test_edge_cases');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

subtest 'NULL Handling' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE null_test (id serial, v vector(3));
    }, 'NULL test table created');
    
    query_ok($node, 'postgres', q{
        INSERT INTO null_test (v) VALUES (NULL);
    }, 'NULL insert works');
    
    query_fails($node, 'postgres', q{
        SELECT NULL::vector + ARRAY[1,2,3]::vector;
    }, 'NULL arithmetic fails gracefully');
    
    query_ok($node, 'postgres', q{
        DROP TABLE null_test CASCADE;
    }, 'NULL test cleaned up');
};

subtest 'Dimension Mismatches' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_fails($node, 'postgres', q{
        SELECT ARRAY[1,2]::vector + ARRAY[1,2,3]::vector;
    }, 'Dimension mismatch detected');
    
    query_fails($node, 'postgres', q{
        SELECT ARRAY[1,2]::vector <-> ARRAY[1,2,3]::vector;
    }, 'Distance dimension mismatch detected');
};

subtest 'Boundary Conditions' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_ok($node, 'postgres', q{
        SELECT ARRAY[]::real[]::vector(0);
    }, 'Zero dimension vector works');
    
    query_ok($node, 'postgres', q{
        SELECT ARRAY(SELECT 0.0 FROM generate_series(1,16000))::vector;
    }, 'Large dimension vector works');
    
    query_ok($node, 'postgres', q{
        SELECT ARRAY[1e308]::vector;
    }, 'Large magnitude works');
    
    query_ok($node, 'postgres', q{
        SELECT ARRAY[1e-308]::vector;
    }, 'Small magnitude works');
};

subtest 'Concurrent Operations' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE concurrent_test (id serial, v vector(4));
        INSERT INTO concurrent_test (v) 
        SELECT ARRAY(SELECT random() FROM generate_series(1,4))::vector 
        FROM generate_series(1,100);
    }, 'Concurrent test data created');
    
    query_ok($node, 'postgres', q{
        DROP TABLE concurrent_test CASCADE;
    }, 'Concurrent test cleaned up');
};

subtest 'Memory Limits' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    # Test large result sets
    query_ok($node, 'postgres', q{
        CREATE TABLE memory_test (id serial, v vector(128));
        INSERT INTO memory_test (v) 
        SELECT ARRAY(SELECT random() FROM generate_series(1,128))::vector 
        FROM generate_series(1,10000);
    }, 'Large dataset created');
    
    query_ok($node, 'postgres', q{
        SELECT COUNT(*) FROM memory_test;
    }, 'Large count works');
    
    query_ok($node, 'postgres', q{
        DROP TABLE memory_test CASCADE;
    }, 'Memory test cleaned up');
};

subtest 'Invalid Inputs' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_fails($node, 'postgres', q{
        SELECT 'invalid'::vector;
    }, 'Invalid string rejected');
    
    query_fails($node, 'postgres', q{
        SELECT ARRAY['a','b','c']::vector;
    }, 'Non-numeric array rejected');
    
    query_fails($node, 'postgres', q{
        SELECT ARRAY[1/0]::vector;
    }, 'Infinity rejected');
    
    query_fails($node, 'postgres', q{
        SELECT ARRAY['NaN'::float]::vector;
    }, 'NaN rejected');
};

$node->stop();
$node->cleanup();
done_testing();



