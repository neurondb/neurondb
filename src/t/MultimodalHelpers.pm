package MultimodalHelpers;

use strict;
use warnings;
use PostgresNode;
use TapTest;
use Exporter 'import';

our @EXPORT = qw(
	test_image_embedding
	test_text_embedding
	test_cross_modal_similarity
	test_multimodal_fusion
	test_image_to_text_search
	test_text_to_image_search
);

=head1 NAME

MultimodalHelpers - Multimodal embedding test helpers

=head1 SYNOPSIS

  use MultimodalHelpers;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  test_image_embedding($node, 'postgres', ...);
  test_cross_modal_similarity($node, 'postgres', ...);

=head1 DESCRIPTION

Provides test helpers for multimodal embeddings (CLIP, DINOv2, etc.).

=cut

=head2 test_image_embedding

Test image embedding generation.

=cut

sub test_image_embedding {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $model = $params{model} || 'CLIP';
	my $image_path = $params{image_path} || '/path/to/image.jpg';
	
	# Test image embedding function
	my $result = $node->psql($dbname, qq{
		SELECT neurondb.image_embed(
			'$model',
			'$image_path'
		) AS embedding;
	});
	
	if ($result->{success}) {
		return (1, "Image embedding generated");
	} else {
		return (0, "Image embedding failed: $result->{stderr}");
	}
}

=head2 test_text_embedding

Test text embedding generation.

=cut

sub test_text_embedding {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $model = $params{model} || 'CLIP';
	my $text = $params{text} || 'test text';
	
	# Test text embedding function
	my $result = $node->psql($dbname, qq{
		SELECT neurondb.text_embed(
			'$model',
			'$text'
		) AS embedding;
	});
	
	if ($result->{success}) {
		return (1, "Text embedding generated");
	} else {
		return (0, "Text embedding failed: $result->{stderr}");
	}
}

=head2 test_cross_modal_similarity

Test cross-modal similarity computation.

=cut

sub test_cross_modal_similarity {
	my ($node, $dbname, $image_embedding, $text_embedding) = @_;
	$dbname ||= 'postgres';
	
	# Compute similarity between image and text embeddings
	my $result = $node->psql($dbname, qq{
		SELECT '$image_embedding'::vector <=> '$text_embedding'::vector AS similarity;
	}, tuples_only => 1);
	
	if ($result->{success}) {
		return (1, "Cross-modal similarity computed");
	} else {
		return (0, "Cross-modal similarity failed: $result->{stderr}");
	}
}

=head2 test_multimodal_fusion

Test multimodal fusion strategies.

=cut

sub test_multimodal_fusion {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $image_emb = $params{image_emb} || '[0.1,0.2,0.3]';
	my $text_emb = $params{text_emb} || '[0.4,0.5,0.6]';
	my $strategy = $params{strategy} || 'concat';
	
	my $result;
	if ($strategy eq 'concat') {
		$result = $node->psql($dbname, qq{
			SELECT vector_concat(
				'$image_emb'::vector,
				'$text_emb'::vector
			) AS fused;
		});
	} elsif ($strategy eq 'average') {
		$result = $node->psql($dbname, qq{
			SELECT vector_avg(ARRAY[
				'$image_emb'::vector,
				'$text_emb'::vector
			]) AS fused;
		});
	}
	
	if ($result && $result->{success}) {
		return (1, "Multimodal fusion successful");
	} else {
		return (0, "Multimodal fusion failed");
	}
}

=head2 test_image_to_text_search

Test image-to-text search.

=cut

sub test_image_to_text_search {
	my ($node, $dbname, $table, $image_embedding, %params) = @_;
	$dbname ||= 'postgres';
	
	my $k = $params{k} || 10;
	my $text_col = $params{text_col} || 'text_embedding';
	
	my $result = $node->psql($dbname, qq{
		SELECT id, $text_col <=> '$image_embedding'::vector AS similarity
		FROM $table
		ORDER BY $text_col <=> '$image_embedding'::vector
		LIMIT $k;
	});
	
	if ($result->{success}) {
		return (1, "Image-to-text search successful");
	} else {
		return (0, "Image-to-text search failed: $result->{stderr}");
	}
}

=head2 test_text_to_image_search

Test text-to-image search.

=cut

sub test_text_to_image_search {
	my ($node, $dbname, $table, $text_embedding, %params) = @_;
	$dbname ||= 'postgres';
	
	my $k = $params{k} || 10;
	my $image_col = $params{image_col} || 'image_embedding';
	
	my $result = $node->psql($dbname, qq{
		SELECT id, $image_col <=> '$text_embedding'::vector AS similarity
		FROM $table
		ORDER BY $image_col <=> '$text_embedding'::vector
		LIMIT $k;
	});
	
	if ($result->{success}) {
		return (1, "Text-to-image search successful");
	} else {
		return (0, "Text-to-image search failed: $result->{stderr}");
	}
}

1;



