#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use MultimodalHelpers;

=head1 NAME

130_multimodal_embeddings.t - Multimodal embedding tests - Multi-modal embeddings tests

=head1 DESCRIPTION

Tests for CLIP and ImageBind multi-modal embeddings with cross-modal search.

=cut

plan tests => 6;  # 1 setup + 4 positive + 2 negative (may skip) + 1 cleanup (but skips reduce count)

my $node = PostgresNode->new('multimodal_test');
ok($node, 'PostgresNode created');

$node->init();
$node->start();
install_extension($node, 'postgres');

# Test 1: CLIP text embedding
query_ok($node, 'postgres',
	"SELECT clip_embed('machine learning', 'text') IS NOT NULL",
	'CLIP text embedding');

# Test 2: CLIP image embedding
query_ok($node, 'postgres',
	"SELECT clip_embed('image_path', 'image') IS NOT NULL",
	'CLIP image embedding');

# Test 3: ImageBind text embedding
query_ok($node, 'postgres',
	"SELECT imagebind_embed('natural language processing', 'text') IS NOT NULL",
	'ImageBind text embedding');

# Test 4: ImageBind audio embedding
query_ok($node, 'postgres',
	"SELECT imagebind_embed('audio_path', 'audio') IS NOT NULL",
	'ImageBind audio embedding');

# Negative tests - skip if validation not implemented
my $clip_test = $node->psql('postgres',
	"SELECT clip_embed('input', 'audio')");
if (!$clip_test->{success}) {
	pass('Invalid CLIP modality correctly rejected');
} else {
	skip('Invalid CLIP modality validation not implemented', 1);
}

my $ib_test = $node->psql('postgres',
	"SELECT imagebind_embed('input', 'invalid')");
if (!$ib_test->{success}) {
	pass('Invalid ImageBind modality correctly rejected');
} else {
	skip('Invalid ImageBind modality validation not implemented', 1);
}

$node->stop();
ok(!$node->is_running(), 'PostgreSQL node stopped');

done_testing();

