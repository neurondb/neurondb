package IndexHelpers;

use strict;
use warnings;
use PostgresNode;
use TapTest;
use Exporter 'import';

our @EXPORT = qw(
	create_hnsw_index
	create_ivf_index
	test_index_creation
	test_knn_query
	test_index_statistics
	test_index_rebuild
	test_index_maintenance
	validate_index_health
	get_index_info
);

=head1 NAME

IndexHelpers - Index operation test helpers

=head1 SYNOPSIS

  use IndexHelpers;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  create_hnsw_index($node, 'postgres', 'test_table', 'vec', ...);
  test_knn_query($node, 'postgres', 'idx_hnsw', ...);

=head1 DESCRIPTION

Provides comprehensive test helpers for HNSW, IVF, and other index types.

=cut

=head2 create_hnsw_index

Create an HNSW index with specified parameters.

=cut

sub create_hnsw_index {
	my ($node, $dbname, $table, $column, %params) = @_;
	$dbname ||= 'postgres';
	
	my $index_name = $params{index_name} || "idx_${table}_${column}_hnsw";
	my $m = $params{m} || 16;
	my $ef_construction = $params{ef_construction} || 200;
	my $operator_class = $params{operator_class} || 'vector_l2_ops';
	
	my $create_sql = qq{
		CREATE INDEX $index_name ON $table 
		USING hnsw ($column $operator_class)
		WITH (m = $m, ef_construction = $ef_construction);
	};
	
	my $result = $node->psql($dbname, $create_sql);
	
	if ($result->{success}) {
		return (1, $index_name);
	} else {
		return (0, "Index creation failed: $result->{stderr}");
	}
}

=head2 create_ivf_index

Create an IVF index with specified parameters.

=cut

sub create_ivf_index {
	my ($node, $dbname, $table, $column, %params) = @_;
	$dbname ||= 'postgres';
	
	my $index_name = $params{index_name} || "idx_${table}_${column}_ivf";
	my $lists = $params{lists} || 100;
	my $operator_class = $params{operator_class} || 'vector_l2_ops';
	
	my $create_sql = qq{
		CREATE INDEX $index_name ON $table 
		USING ivfflat ($column $operator_class)
		WITH (lists = $lists);
	};
	
	my $result = $node->psql($dbname, $create_sql);
	
	if ($result->{success}) {
		return (1, $index_name);
	} else {
		return (0, "Index creation failed: $result->{stderr}");
	}
}

=head2 test_index_creation

Test index creation with various parameters.

=cut

sub test_index_creation {
	my ($node, $dbname, $index_type, %params) = @_;
	$dbname ||= 'postgres';
	
	my $table = $params{table} || 'test_index_table';
	my $column = $params{column} || 'vec';
	
	# Ensure table exists
	my $check = $node->psql($dbname,
		"SELECT COUNT(*) FROM pg_tables WHERE tablename = '$table';",
		tuples_only => 1
	);
	
	unless ($check->{success} && $check->{stdout} =~ /1/) {
		return (0, "Table $table does not exist");
	}
	
	if ($index_type eq 'hnsw') {
		return create_hnsw_index($node, $dbname, $table, $column, %params);
	} elsif ($index_type eq 'ivf') {
		return create_ivf_index($node, $dbname, $table, $column, %params);
	}
	
	return (0, "Unknown index type: $index_type");
}

=head2 test_knn_query

Test KNN query using an index.

=cut

sub test_knn_query {
	my ($node, $dbname, $table, $column, $query_vec, %params) = @_;
	$dbname ||= 'postgres';
	
	my $k = $params{k} || 10;
	my $distance_op = $params{distance_op} || '<->';
	
	my $query_sql = qq{
		SELECT id, $column $distance_op '$query_vec'::vector AS distance
		FROM $table
		ORDER BY $column $distance_op '$query_vec'::vector
		LIMIT $k;
	};
	
	my $result = $node->psql($dbname, $query_sql);
	
	if ($result->{success}) {
		return (1, "KNN query successful");
	} else {
		return (0, "KNN query failed: $result->{stderr}");
	}
}

=head2 test_index_statistics

Get and validate index statistics.

=cut

sub test_index_statistics {
	my ($node, $dbname, $index_name) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT * FROM neurondb.index_metadata WHERE index_oid = '$index_name'::regclass::oid;"
	);
	
	if ($result->{success}) {
		return (1, "Index statistics retrieved");
	} else {
		return (0, "Failed to get index statistics: $result->{stderr}");
	}
}

=head2 test_index_rebuild

Test index rebuild operation.

=cut

sub test_index_rebuild {
	my ($node, $dbname, $index_name) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"REINDEX INDEX $index_name;"
	);
	
	if ($result->{success}) {
		return (1, "Index rebuild successful");
	} else {
		return (0, "Index rebuild failed: $result->{stderr}");
	}
}

=head2 test_index_maintenance

Test index maintenance operations (VACUUM, ANALYZE).

=cut

sub test_index_maintenance {
	my ($node, $dbname, $table) = @_;
	$dbname ||= 'postgres';
	
	my @results;
	
	# VACUUM
	my $result = $node->psql($dbname, "VACUUM $table;");
	push @results, $result->{success};
	
	# ANALYZE
	$result = $node->psql($dbname, "ANALYZE $table;");
	push @results, $result->{success};
	
	# VACUUM ANALYZE
	$result = $node->psql($dbname, "VACUUM ANALYZE $table;");
	push @results, $result->{success};
	
	return \@results;
}

=head2 validate_index_health

Validate index health status.

=cut

sub validate_index_health {
	my ($node, $dbname, $index_name) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT health_status FROM neurondb.index_health 
		 WHERE index_name = '$index_name';",
		tuples_only => 1
	);
	
	if ($result->{success} && $result->{stdout}) {
		my $status = $result->{stdout};
		chomp $status;
		$status =~ s/^\s+|\s+$//g;
		return (1, $status);
	}
	
	return (0, "Could not retrieve index health");
}

=head2 get_index_info

Get detailed information about an index.

=cut

sub get_index_info {
	my ($node, $dbname, $index_name) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT indexname, indexdef FROM pg_indexes WHERE indexname = '$index_name';",
		tuples_only => 1
	);
	
	if ($result->{success}) {
		return $result->{stdout};
	}
	
	return undef;
}

1;



