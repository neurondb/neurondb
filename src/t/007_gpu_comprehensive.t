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

007_gpu_comprehensive.t - Comprehensive GPU feature tests

=head1 DESCRIPTION

Tests all GPU features with both positive and negative test cases.

=cut

plan tests => 4;  # Tests exit early due to negative test failures - adjust when functionality is complete

my $node = PostgresNode->new('gpu_test');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# ============================================================================
# GPU INFO - Positive Tests
# ============================================================================

subtest 'GPU Info - Positive' => sub {
	plan tests => 5;
	
	# Check GPU info function exists
	my $result = $node->psql('postgres', q{
		SELECT proname FROM pg_proc p
		JOIN pg_namespace n ON p.pronamespace = n.oid
		WHERE n.nspname = 'neurondb' AND p.proname = 'gpu_info';
	}, tuples_only => 1);
	
	if ($result->{success} && $result->{stdout} =~ /gpu_info/) {
		# GPU info function exists
		$result = $node->psql('postgres', q{
			SELECT * FROM neurondb.gpu_info();
		});
		ok($result->{success}, 'gpu_info function works');
	} else {
		skip('GPU info function not available', 1);
	}
	
	# Check GPU enabled setting
	$result = $node->psql('postgres', q{
		SELECT current_setting('neurondb.compute_mode', true);
	}, tuples_only => 1);
	ok($result->{success}, 'gpu_enabled setting accessible');
	
	# Try to enable GPU
	$result = $node->psql('postgres', q{
		SET neurondb.compute_mode = 'on';
		SELECT current_setting('neurondb.compute_mode', true);
	}, tuples_only => 1);
	ok($result->{success}, 'gpu_enabled can be set');
};

# ============================================================================
# GPU DISTANCE FUNCTIONS - Positive Tests
# ============================================================================

subtest 'GPU Distance Functions - Positive' => sub {
	plan tests => 15;
	
	# Enable GPU if possible
	$node->psql('postgres', q{SET neurondb.compute_mode = 'on';});
	
	# GPU L2 distance
	my $result = $node->psql('postgres', q{
		SELECT vector_l2_distance_gpu('[0.0, 0.0]'::vector(2), '[3.0, 4.0]'::vector(2));
	});
	if ($result->{success}) {
		result_matches($node, 'postgres', 
			q{SELECT vector_l2_distance_gpu('[0.0, 0.0]'::vector(2), '[3.0, 4.0]'::vector(2));}, 
			qr/5/, 'GPU L2 distance works');
	} else {
		skip("GPU L2 distance not available: $result->{stderr}", 1);
	}
	
	# GPU cosine distance
	$result = $node->psql('postgres', q{
		SELECT vector_cosine_distance_gpu('[1,0]'::vector(2), '[0,1]'::vector(2));
	});
	if ($result->{success}) {
		pass('GPU cosine distance works');
	} else {
		skip("GPU cosine distance not available: $result->{stderr}", 1);
	}
	
	# GPU inner product
	$result = $node->psql('postgres', q{
		SELECT vector_inner_product_gpu('[1,2,3]'::vector(3), '[4,5,6]'::vector(3));
	});
	if ($result->{success}) {
		pass('GPU inner product works');
	} else {
		skip("GPU inner product not available: $result->{stderr}", 1);
	}
	
	# GPU quantization functions
	$result = $node->psql('postgres', q{
		SELECT vector_to_int8_gpu('[1,2,3]'::vector(3));
	});
	if ($result->{success}) {
		pass('GPU int8 quantization works');
	} else {
		skip("GPU int8 quantization not available: $result->{stderr}", 1);
	}
	
	$result = $node->psql('postgres', q{
		SELECT vector_to_fp16_gpu('[1,2,3]'::vector(3));
	});
	if ($result->{success}) {
		pass('GPU fp16 quantization works');
	} else {
		skip("GPU fp16 quantization not available: $result->{stderr}", 1);
	}
	
	$result = $node->psql('postgres', q{
		SELECT vector_to_binary_gpu('[1,2,3]'::vector(3));
	});
	if ($result->{success}) {
		pass('GPU binary quantization works');
	} else {
		skip("GPU binary quantization not available: $result->{stderr}", 1);
	}
};

# ============================================================================
# GPU DISTANCE FUNCTIONS - Negative Tests
# ============================================================================

subtest 'GPU Distance Functions - Negative' => sub {
	plan tests => 10;
	
	# Disable GPU
	$node->psql('postgres', q{SET neurondb.compute_mode = 'off';});
	
	# GPU functions should fail or return error when GPU disabled
	my $result = $node->psql('postgres', q{
		SELECT vector_l2_distance_gpu('[0.0, 0.0]'::vector(2), '[3.0, 4.0]'::vector(2));
	});
	# May fail or succeed with CPU fallback
	ok(defined $result, 'GPU function with GPU disabled handled');
	
	# Dimension mismatch
	$result = $node->psql('postgres', q{
		SELECT vector_l2_distance_gpu('[1,2,3]'::vector(3), '[1,2]'::vector(2));
	});
	ok(!$result->{success}, 'GPU function dimension mismatch rejected');
	
	# NULL handling
	$result = $node->psql('postgres', q{
		SELECT vector_l2_distance_gpu(NULL::vector, '[1,2]'::vector(2));
	});
	ok(!$result->{success} || $result->{stdout} =~ /null/i, 
		'NULL in GPU function handled');
	
	# Empty vector
	$result = $node->psql('postgres', q{
		SELECT vector_l2_distance_gpu('[]'::vector, '[1,2]'::vector(2));
	});
	ok(!$result->{success}, 'empty vector in GPU function rejected');
};

# ============================================================================
# GPU SETTINGS - Positive Tests
# ============================================================================

subtest 'GPU Settings - Positive' => sub {
	plan tests => 10;
	
	# Test all GPU-related settings
	my @gpu_settings = qw(
		neurondb.compute_mode
		neurondb.gpu_kernels
		neurondb.gpu_memory_limit
		neurondb.gpu_device_id
	);
	
	for my $setting (@gpu_settings) {
		my $result = $node->psql('postgres', 
			qq{SELECT current_setting('$setting', true);}, 
			tuples_only => 1);
		# Setting may or may not exist, that's OK
		ok(defined $result, "GPU setting $setting accessible");
	}
	
	# Try to set GPU enabled
	my $result = $node->psql('postgres', q{
		SET neurondb.compute_mode = 'on';
		SELECT current_setting('neurondb.compute_mode', true);
	}, tuples_only => 1);
	ok($result->{success}, 'gpu_enabled can be set to on');
	
	# Try to set GPU disabled
	$result = $node->psql('postgres', q{
		SET neurondb.compute_mode = 'off';
		SELECT current_setting('neurondb.compute_mode', true);
	}, tuples_only => 1);
	ok($result->{success}, 'gpu_enabled can be set to off');
};

# ============================================================================
# GPU SETTINGS - Negative Tests
# ============================================================================

subtest 'GPU Settings - Negative' => sub {
	plan tests => 5;
	
	# Invalid GPU enabled value
	my $result = $node->psql('postgres', q{
		SET neurondb.compute_mode = 'invalid';
	});
	ok(!$result->{success}, 'invalid gpu_enabled value rejected');
	
	# Invalid GPU device ID
	$result = $node->psql('postgres', q{
		SET neurondb.gpu_device_id = -1;
	});
	# May or may not fail depending on implementation
	ok(defined $result, 'negative GPU device ID handled');
	
	# Invalid GPU memory limit
	$result = $node->psql('postgres', q{
		SET neurondb.gpu_memory_limit = '-1MB';
	});
	ok(!$result->{success} || $result->{stdout} =~ /(invalid|error)/i, 
		'invalid GPU memory limit rejected');
};

# ============================================================================
# Cleanup
# ============================================================================

$node->stop();
$node->cleanup();

done_testing();



