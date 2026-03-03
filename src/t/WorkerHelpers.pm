package WorkerHelpers;

use strict;
use warnings;
use PostgresNode;
use TapTest;
use Exporter 'import';

our @EXPORT = qw(
	submit_job
	get_job_status
	cancel_job
	get_worker_status
	test_async_operation
	wait_for_job_completion
);

=head1 NAME

WorkerHelpers - Background worker test helpers

=head1 SYNOPSIS

  use WorkerHelpers;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  submit_job($node, 'postgres', 'index_build', ...);
  wait_for_job_completion($node, 'postgres', $job_id);

=head1 DESCRIPTION

Provides test helpers for background workers and job queues.

=cut

=head2 submit_job

Submit a job to the job queue.

=cut

sub submit_job {
	my ($node, $dbname, $job_type, %params) = @_;
	$dbname ||= 'postgres';
	
	my $job_data = $params{job_data} || '{}';
	my $priority = $params{priority} || 5;
	
	my $result = $node->psql($dbname, qq{
		INSERT INTO neurondb.job_queue (job_type, job_data, priority)
		VALUES ('$job_type', '$job_data'::jsonb, $priority)
		RETURNING job_id;
	}, tuples_only => 1);
	
	if ($result->{success} && $result->{stdout}) {
		my $job_id = $result->{stdout};
		chomp $job_id;
		$job_id =~ s/^\s+|\s+$//g;
		return (1, $job_id);
	} else {
		return (0, "Job submission failed: $result->{stderr}");
	}
}

=head2 get_job_status

Get the status of a job.

=cut

sub get_job_status {
	my ($node, $dbname, $job_id) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, qq{
		SELECT status FROM neurondb.job_queue WHERE job_id = $job_id;
	}, tuples_only => 1);
	
	if ($result->{success} && $result->{stdout}) {
		my $status = $result->{stdout};
		chomp $status;
		$status =~ s/^\s+|\s+$//g;
		return (1, $status);
	} else {
		return (0, "Could not get job status");
	}
}

=head2 cancel_job

Cancel a job.

=cut

sub cancel_job {
	my ($node, $dbname, $job_id) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, qq{
		UPDATE neurondb.job_queue 
		SET status = 'cancelled'
		WHERE job_id = $job_id;
	});
	
	if ($result->{success}) {
		return (1, "Job cancelled");
	} else {
		return (0, "Job cancellation failed: $result->{stderr}");
	}
}

=head2 get_worker_status

Get status of background workers.

=cut

sub get_worker_status {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, q{
		SELECT * FROM neurondb.worker_status;
	});
	
	if ($result->{success}) {
		return (1, "Worker status retrieved");
	} else {
		return (0, "Could not get worker status: $result->{stderr}");
	}
}

=head2 test_async_operation

Test an asynchronous operation.

=cut

sub test_async_operation {
	my ($node, $dbname, $operation, %params) = @_;
	$dbname ||= 'postgres';
	
	# Submit async job
	my ($success, $job_id) = submit_job($node, $dbname, $operation,
		job_data => $params{job_data} || '{}',
		priority => $params{priority} || 5
	);
	
	unless ($success) {
		return (0, "Failed to submit async job");
	}
	
	# Wait for completion
	my ($complete, $status) = wait_for_job_completion($node, $dbname, $job_id,
		timeout => $params{timeout} || 60
	);
	
	return ($complete, $status);
}

=head2 wait_for_job_completion

Wait for a job to complete.

=cut

sub wait_for_job_completion {
	my ($node, $dbname, $job_id, %params) = @_;
	$dbname ||= 'postgres';
	
	my $timeout = $params{timeout} || 60;
	my $start_time = time();
	
	while ((time() - $start_time) < $timeout) {
		my ($success, $status) = get_job_status($node, $dbname, $job_id);
		
		if ($success) {
			if ($status eq 'done' || $status eq 'failed' || $status eq 'cancelled') {
				return (1, $status);
			}
		}
		
		sleep(1);
	}
	
	return (0, "Job did not complete within timeout");
}

1;



