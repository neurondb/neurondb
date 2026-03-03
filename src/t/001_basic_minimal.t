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

001_basic_minimal.t - Minimal usage examples of PostgresNode, TapTest, and NeuronDB

=head1 DESCRIPTION

Demonstrates minimal usage of all three modules:
- Single PostgresNode instance
- Basic TapTest assertions
- Simple NeuronDB operations

=cut

plan tests => 14;  # Updated to match actual test count (neurondb_ok runs 3 tests)

# Minimal PostgresNode usage: single node, basic init/start/stop
my $node = PostgresNode->new('minimal_test');
ok($node, 'PostgresNode created');

$node->init();
ok(-d $node->{data_dir}, 'Data directory initialized');

$node->start();
ok($node->is_running(), 'PostgreSQL node started');

# Minimal TapTest usage: basic assertions
my $result = $node->psql('postgres', 'SELECT 1');
query_ok($node, 'postgres', 'SELECT 1', 'Simple query executes');

result_is($node, 'postgres', 'SELECT 1', '1', 'Result matches expected');

# Minimal NeuronDB usage: extension installation and basic vector operations
install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

extension_ok($node, 'postgres', 'neurondb', 'Extension verified');

schema_ok($node, 'postgres', 'neurondb', 'NeuronDB schema exists');

# Minimal vector operations using VectorOps module
my $vec_results = VectorOps::test_vector_creation($node, 'postgres', 3);
ok(scalar @$vec_results > 0 && $vec_results->[0], "Vector creation works");

my ($success, $msg) = test_vector_operations($node, 'postgres');
ok($success, "Vector operations: $msg");

# Cleanup
$node->stop();
ok(!$node->is_running(), 'PostgreSQL node stopped');

$node->cleanup();
ok(!-d $node->{data_dir}, 'Data directory cleaned up');

done_testing();

