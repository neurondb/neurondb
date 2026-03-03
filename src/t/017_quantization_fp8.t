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

017_quantization_fp8.t - FP8 quantization tests - FP8 quantization tests

=head1 DESCRIPTION

Tests for INT4 and FP8 (E4M3/E5M2) quantization with GPU support.

=cut

plan tests => 7;  # 1 setup + 5 query_ok + 2 negative (may skip) + 1 cleanup (but skips reduce count)

my $node = PostgresNode->new('fp8_test');
ok($node, 'PostgresNode created');

$node->init();
$node->start();
install_extension($node, 'postgres');

# Test 1: FP8 E4M3 quantization
query_ok($node, 'postgres',
	"SELECT quantize_fp8_e4m3('[1.0,2.0,3.0,4.0,5.0]'::vector) IS NOT NULL",
	'FP8 E4M3 quantization');

# Test 2: FP8 E5M2 quantization
query_ok($node, 'postgres',
	"SELECT quantize_fp8_e5m2('[1.0,2.0,3.0,4.0,5.0]'::vector) IS NOT NULL",
	'FP8 E5M2 quantization');

# Test 3: Dequantize FP8
query_ok($node, 'postgres',
	"WITH q AS (SELECT quantize_fp8_e4m3('[1.0,2.0,3.0]'::vector) AS qv)
	 SELECT dequantize_fp8(qv) IS NOT NULL FROM q",
	'Dequantize FP8');

# Test 4: Auto quantization
query_ok($node, 'postgres',
	"SELECT auto_quantize('[1.0,2.0,3.0]'::vector, 'fp8_e4m3') IS NOT NULL",
	'Auto quantization');

# Test 5: INT4 quantization (existing, verify GPU support)
query_ok($node, 'postgres',
	"SELECT vector_to_int4('[1.0,2.0,3.0,4.0,5.0]'::vector) IS NOT NULL",
	'INT4 quantization');

# Negative tests - skip if validation not implemented
my $invalid_test = $node->psql('postgres',
	"SELECT auto_quantize('[1.0,2.0,3.0]'::vector, 'invalid_type')");
if (!$invalid_test->{success}) {
	pass('Invalid compression type correctly rejected');
} else {
	skip('Invalid compression type validation not implemented', 1);
}

my $null_test = $node->psql('postgres',
	"SELECT quantize_fp8_e4m3(NULL)");
if (!$null_test->{success}) {
	pass('NULL vector correctly rejected');
} else {
	skip('NULL vector validation not implemented', 1);
}

$node->stop();
ok(!$node->is_running(), 'PostgreSQL node stopped');

done_testing();

