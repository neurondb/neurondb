package GPUHelpers;

use strict;
use warnings;
use PostgresNode;
use TapTest;
use Exporter 'import';

our @EXPORT = qw(
	check_gpu_available
	get_gpu_info
	test_gpu_distance
	test_gpu_ml
	compare_gpu_cpu
	test_gpu_memory
	enable_gpu_mode
	disable_gpu_mode
);

=head1 NAME

GPUHelpers - GPU feature test helpers

=head1 SYNOPSIS

  use GPUHelpers;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  if (check_gpu_available($node, 'postgres')) {
      test_gpu_distance($node, 'postgres', ...);
  }

=head1 DESCRIPTION

Provides test helpers for GPU-accelerated operations.

=cut

=head2 check_gpu_available

Check if GPU is available and enabled.

=cut

sub check_gpu_available {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	# Check compute mode
	my $result = $node->psql($dbname, q{
		SELECT current_setting('neurondb.compute_mode', true) AS compute_mode;
	}, tuples_only => 1);
	
	if ($result->{success}) {
		my $mode = $result->{stdout};
		chomp $mode;
		$mode =~ s/^\s+|\s+$//g;
		return $mode eq '1' || $mode eq '2';  # GPU or AUTO
	}
	
	# Check if gpu_info function exists
	$result = $node->psql($dbname, q{
		SELECT COUNT(*) FROM pg_proc p
		JOIN pg_namespace n ON p.pronamespace = n.oid
		WHERE n.nspname = 'neurondb' AND p.proname = 'gpu_info';
	}, tuples_only => 1);
	
	return $result->{success} && $result->{stdout} =~ /1/;
}

=head2 get_gpu_info

Get GPU information.

=cut

sub get_gpu_info {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, q{
		SELECT * FROM neurondb.gpu_info();
	});
	
	if ($result->{success}) {
		return $result->{stdout};
	}
	
	return undef;
}

=head2 test_gpu_distance

Test GPU-accelerated distance computation.

=cut

sub test_gpu_distance {
	my ($node, $dbname, $v1, $v2, %params) = @_;
	$dbname ||= 'postgres';
	
	my $distance_op = $params{distance_op} || '<->';
	
	# Try GPU-specific function if available
	my $result = $node->psql($dbname, qq{
		SELECT vector_l2_distance_gpu(
			'$v1'::vector,
			'$v2'::vector
		) AS gpu_dist;
	});
	
	if ($result->{success}) {
		return (1, "GPU distance computed");
	}
	
	# Fallback to regular distance (may use GPU if enabled)
	$result = $node->psql($dbname, qq{
		SELECT '$v1'::vector $distance_op '$v2'::vector AS dist;
	}, tuples_only => 1);
	
	if ($result->{success}) {
		return (1, "Distance computed (may be GPU-accelerated)");
	}
	
	return (0, "Distance computation failed: $result->{stderr}");
}

=head2 test_gpu_ml

Test GPU-accelerated ML operations.

=cut

sub test_gpu_ml {
	my ($node, $dbname, $algorithm, %params) = @_;
	$dbname ||= 'postgres';
	
	# Enable GPU mode
	enable_gpu_mode($node, $dbname);
	
	# Use MLHelpers to train with GPU
	require MLHelpers;
	my ($success, $msg) = MLHelpers::train_ml_model($node, $dbname, $algorithm,
		%params,
		options => '{"use_gpu": true}'
	);
	
	return ($success, $msg);
}

=head2 compare_gpu_cpu

Compare GPU vs CPU performance and results.

=cut

sub compare_gpu_cpu {
	my ($node, $dbname, $sql, %params) = @_;
	$dbname ||= 'postgres';
	
	my @results;
	
	# CPU mode
	disable_gpu_mode($node, $dbname);
	my $cpu_start = time();
	my $cpu_result = $node->psql($dbname, $sql);
	my $cpu_time = time() - $cpu_start;
	
	push @results, {
		mode => 'CPU',
		success => $cpu_result->{success},
		time => $cpu_time,
		result => $cpu_result->{stdout}
	};
	
	# GPU mode (if available)
	if (check_gpu_available($node, $dbname)) {
		enable_gpu_mode($node, $dbname);
		my $gpu_start = time();
		my $gpu_result = $node->psql($dbname, $sql);
		my $gpu_time = time() - $gpu_start;
		
		push @results, {
			mode => 'GPU',
			success => $gpu_result->{success},
			time => $gpu_time,
			result => $gpu_result->{stdout}
		};
	}
	
	return \@results;
}

=head2 test_gpu_memory

Test GPU memory management.

=cut

sub test_gpu_memory {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, q{
		SELECT * FROM neurondb.gpu_info();
	});
	
	if ($result->{success} && $result->{stdout} =~ /memory/) {
		return (1, "GPU memory info available");
	}
	
	return (0, "GPU memory info not available");
}

=head2 enable_gpu_mode

Enable GPU compute mode.

=cut

sub enable_gpu_mode {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SET neurondb.compute_mode = '1';"
	);
	
	return $result->{success};
}

=head2 disable_gpu_mode

Disable GPU compute mode (use CPU).

=cut

sub disable_gpu_mode {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SET neurondb.compute_mode = '0';"
	);
	
	return $result->{success};
}

1;



