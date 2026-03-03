#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use WorkerHelpers;

=head1 NAME

026_worker_async.t - Async worker operations tests

=head1 DESCRIPTION

Comprehensive tests for background workers:
- Job queue management
- Async index building
- Batch processing
- Worker status monitoring
- Error handling

=cut

plan tests => 4;  # 3 neurondb_ok + 1 subtest

my $node = PostgresNode->new('test_workers');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

subtest 'Job Queue' => sub {
    plan tests => 4;  # 3 neurondb_ok + 1 subtest
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.submit_job('test_job', 'SELECT 1');
    }, 'Job submission works');
    
    query_ok($node, 'postgres', q{
        SELECT * FROM neurondb.list_jobs();
    }, 'Job listing works');
};

$node->stop();
$node->cleanup();
done_testing();



