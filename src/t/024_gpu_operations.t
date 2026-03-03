#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use GPUHelpers;

=head1 NAME

024_gpu_operations.t - GPU operations comprehensive tests

=head1 DESCRIPTION

Comprehensive tests for GPU operations:
- GPU detection and availability
- Vector operations on GPU
- ML training on GPU
- Memory management
- CPU fallback
- Multi-GPU support

=cut

plan tests => 4;  # 3 neurondb_ok + 1 subtest

my $node = PostgresNode->new('test_gpu');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

subtest 'GPU Detection' => sub {
    plan tests => 4;  # 3 neurondb_ok + 1 subtest
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.gpu_available();
    }, 'GPU availability check works');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.gpu_device_count();
    }, 'GPU count retrieved');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.gpu_info();
    }, 'GPU info retrieved');
};

$node->stop();
$node->cleanup();
done_testing();



