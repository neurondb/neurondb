package TapTest;

use strict;
use warnings;
use Test::More;
use Exporter 'import';

our @EXPORT = qw(
	neurondb_ok
	query_ok
	query_fails
	result_is
	result_matches
	vector_ok
	ml_result_ok
	extension_ok
	schema_ok
	table_ok
	function_ok
	result_within_tolerance
	result_json_ok
	result_array_contains
	performance_ok
	error_message_matches
	table_row_count_is
	column_exists_ok
	index_exists_ok
	trigger_exists_ok
);

=head1 NAME

TapTest - TAP test utilities for NeuronDB

=head1 SYNOPSIS

  use TapTest;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  query_ok($node, 'postgres', 'SELECT 1', 'Simple query');
  result_is($node, 'postgres', 'SELECT 1', '1', 'Result matches');
  extension_ok($node, 'neurondb', 'NeuronDB extension installed');

=head1 DESCRIPTION

Provides TAP test utilities and NeuronDB-specific assertions for
testing PostgreSQL extensions and NeuronDB functionality.

=cut

=head2 neurondb_ok

Check if NeuronDB extension is properly installed and configured.

=cut

sub neurondb_ok {
	my ($node, $dbname, $test_name) = @_;
	$test_name ||= 'NeuronDB extension is installed';
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, q{
		SELECT extname, extversion 
		FROM pg_extension 
		WHERE extname = 'neurondb';
	});
	
	ok($result->{success}, $test_name);
	
	if ($result->{success}) {
		like($result->{stdout}, qr/neurondb/, "Extension name found");
		like($result->{stdout}, qr/\d+\.\d+/, "Extension version found");
	}
	
	return $result->{success};
}

=head2 query_ok

Execute a SQL query and verify it succeeds.

=cut

sub query_ok {
	my ($node, $dbname, $sql, $test_name) = @_;
	$test_name ||= "Query executes successfully";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql);
	ok($result->{success}, $test_name);
	
	unless ($result->{success}) {
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
	}
	
	return $result->{success};
}

=head2 query_fails

Execute a SQL query and verify it fails (returns non-zero exit code).

=cut

sub query_fails {
	my ($node, $dbname, $sql, $test_name) = @_;
	$test_name ||= "Query fails as expected";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql);
	ok(!$result->{success}, $test_name);
	
	if ($result->{success}) {
		diag("SQL: $sql");
		diag("Expected query to fail but it succeeded");
	}
	
	return !$result->{success};
}

=head2 result_is

Execute a SQL query and verify the result matches expected value.

=cut

sub result_is {
	my ($node, $dbname, $sql, $expected, $test_name) = @_;
	$test_name ||= "Query result matches expected";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql, tuples_only => 1);
	
	unless ($result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $actual = $result->{stdout};
	$actual =~ s/^\s+|\s+$//g;  # Trim leading/trailing whitespace
	chomp $expected;
	
	is($actual, $expected, $test_name);
	
	unless ($actual eq $expected) {
		diag("Expected: $expected");
		diag("Got: $actual");
	}
	
	return $actual eq $expected;
}

=head2 result_matches

Execute a SQL query and verify the result matches a regex pattern.

=cut

sub result_matches {
	my ($node, $dbname, $sql, $pattern, $test_name) = @_;
	$test_name ||= "Query result matches pattern";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql, tuples_only => 1);
	
	unless ($result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $actual = $result->{stdout};
	chomp $actual;
	
	like($actual, $pattern, $test_name);
	
	unless ($actual =~ $pattern) {
		diag("Pattern: $pattern");
		diag("Got: $actual");
	}
	
	return $actual =~ $pattern;
}

=head2 vector_ok

Verify vector type operations work correctly.

=cut

sub vector_ok {
	my ($node, $dbname, $test_name) = @_;
	$test_name ||= 'Vector operations work';
	$dbname ||= 'postgres';
	
	# Test vector creation
	my $result = $node->psql($dbname, q{
		SELECT '[1,2,3]'::vector(3);
	});
	
	unless ($result->{success}) {
		fail($test_name);
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	# Test vector distance
	$result = $node->psql($dbname, q{
		SELECT '[1,2,3]'::vector(3) <-> '[4,5,6]'::vector(3) AS distance;
	});
	
	unless ($result->{success}) {
		fail($test_name);
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	ok(1, $test_name);
	return 1;
}

=head2 ml_result_ok

Verify ML function result structure and validity.

=cut

sub ml_result_ok {
	my ($node, $dbname, $sql, $test_name, %checks) = @_;
	$test_name ||= 'ML function result is valid';
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql);
	
	unless ($result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $output = $result->{stdout};
	
	# Check for expected columns if specified
	if (exists $checks{columns}) {
		for my $col (@{$checks{columns}}) {
			like($output, qr/$col/, "Result contains column: $col");
		}
	}
	
	# Check for numeric results if specified
	if (exists $checks{numeric}) {
		like($output, qr/\d+\.?\d*/, "Result contains numeric values");
	}
	
	ok(1, $test_name);
	return 1;
}

=head2 extension_ok

Verify an extension is installed.

=cut

sub extension_ok {
	my ($node, $dbname, $extname, $test_name) = @_;
	$test_name ||= "Extension $extname is installed";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, 
		"SELECT extname FROM pg_extension WHERE extname = '$extname';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	$output =~ s/^\s+|\s+$//g;  # Trim leading/trailing whitespace
	
	is($output, $extname, $test_name);
	return $output eq $extname;
}

=head2 schema_ok

Verify a schema exists.

=cut

sub schema_ok {
	my ($node, $dbname, $schemaname, $test_name) = @_;
	$test_name ||= "Schema $schemaname exists";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT nspname FROM pg_namespace WHERE nspname = '$schemaname';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	$output =~ s/^\s+|\s+$//g;  # Trim leading/trailing whitespace
	
	is($output, $schemaname, $test_name);
	return $output eq $schemaname;
}

=head2 table_ok

Verify a table exists.

=cut

sub table_ok {
	my ($node, $dbname, $schemaname, $tablename, $test_name) = @_;
	$test_name ||= "Table $schemaname.$tablename exists";
	$dbname ||= 'postgres';
	
	my $schema_part = $schemaname ? "$schemaname." : "";
	my $result = $node->psql($dbname,
		"SELECT tablename FROM pg_tables WHERE schemaname = '$schemaname' AND tablename = '$tablename';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	chomp $output;
	$output =~ s/^\s+|\s+$//g;  # Trim leading/trailing whitespace
	
	is($output, $tablename, $test_name);
	return $output eq $tablename;
}

=head2 function_ok

Verify a function exists.

=cut

sub function_ok {
	my ($node, $dbname, $schemaname, $funcname, $test_name) = @_;
	$test_name ||= "Function $schemaname.$funcname exists";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT proname FROM pg_proc p JOIN pg_namespace n ON p.pronamespace = n.oid WHERE n.nspname = '$schemaname' AND p.proname = '$funcname';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	chomp $output;
	
	is($output, $funcname, $test_name);
	return $output eq $funcname;
}

=head2 result_within_tolerance

Execute a SQL query and verify the result is within a tolerance of the expected value.

=cut

sub result_within_tolerance {
	my ($node, $dbname, $sql, $expected, $tolerance, $test_name) = @_;
	$test_name ||= "Query result within tolerance";
	$dbname ||= 'postgres';
	$tolerance ||= 0.0001;
	
	my $result = $node->psql($dbname, $sql, tuples_only => 1);
	
	unless ($result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $actual = $result->{stdout};
	$actual =~ s/^\s+|\s+$//g;
	chomp $expected;
	
	unless ($actual =~ /^-?\d+\.?\d*$/ && $expected =~ /^-?\d+\.?\d*$/) {
		fail($test_name);
		diag("Expected numeric values, got: actual=$actual, expected=$expected");
		return 0;
	}
	
	my $diff = abs($actual - $expected);
	my $within = $diff <= $tolerance;
	
	ok($within, $test_name);
	
	unless ($within) {
		diag("Expected: $expected ± $tolerance");
		diag("Got: $actual");
		diag("Difference: $diff");
	}
	
	return $within;
}

=head2 result_json_ok

Execute a SQL query returning JSON and verify it's valid JSON with optional structure checks.

=cut

sub result_json_ok {
	my ($node, $dbname, $sql, $test_name, %checks) = @_;
	$test_name ||= "Query returns valid JSON";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql, tuples_only => 1);
	
	unless ($result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $json_str = $result->{stdout};
	$json_str =~ s/^\s+|\s+$//g;
	
	# Try to parse as JSON (basic check)
	unless ($json_str =~ /^[\[\{]/) {
		fail($test_name);
		diag("Result doesn't look like JSON: $json_str");
		return 0;
	}
	
	# Check for required keys if specified
	if (exists $checks{keys}) {
		for my $key (@{$checks{keys}}) {
			unless ($json_str =~ /"$key"/) {
				fail("$test_name - missing key: $key");
				diag("JSON: $json_str");
				return 0;
			}
		}
	}
	
	ok(1, $test_name);
	return 1;
}

=head2 result_array_contains

Execute a SQL query returning an array and verify it contains expected values.

=cut

sub result_array_contains {
	my ($node, $dbname, $sql, $expected_values, $test_name) = @_;
	$test_name ||= "Query result array contains expected values";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql, tuples_only => 1);
	
	unless ($result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $output = $result->{stdout};
	$output =~ s/^\s+|\s+$//g;
	
	# Parse array (handle PostgreSQL array format)
	my @actual = ();
	if ($output =~ /^\{([^}]*)\}$/) {
		@actual = split(/,/, $1);
		@actual = map { s/^"|"$//g; $_ } @actual;
		@actual = map { s/^\s+|\s+$//g; $_ } @actual;
	}
	
	my @expected = ref($expected_values) eq 'ARRAY' ? @$expected_values : ($expected_values);
	
	my $all_found = 1;
	for my $exp (@expected) {
		my $found = 0;
		for my $act (@actual) {
			if ($act eq $exp) {
				$found = 1;
				last;
			}
		}
		unless ($found) {
			$all_found = 0;
			last;
		}
	}
	
	ok($all_found, $test_name);
	
	unless ($all_found) {
		diag("Expected to find: " . join(', ', @expected));
		diag("Got array: " . join(', ', @actual));
	}
	
	return $all_found;
}

=head2 performance_ok

Execute a SQL query and verify it completes within a time limit.

=cut

sub performance_ok {
	my ($node, $dbname, $sql, $max_seconds, $test_name) = @_;
	$test_name ||= "Query completes within time limit";
	$dbname ||= 'postgres';
	$max_seconds ||= 1.0;
	
	my $start_time = time();
	my $result = $node->psql($dbname, $sql);
	my $elapsed = time() - $start_time;
	
	unless ($result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $within_limit = $elapsed <= $max_seconds;
	ok($within_limit, $test_name);
	
	unless ($within_limit) {
		diag("Query took ${elapsed}s, limit was ${max_seconds}s");
	}
	
	return $within_limit;
}

=head2 error_message_matches

Execute a SQL query that should fail and verify the error message matches a pattern.

=cut

sub error_message_matches {
	my ($node, $dbname, $sql, $pattern, $test_name) = @_;
	$test_name ||= "Error message matches pattern";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql);
	
	unless (!$result->{success}) {
		fail($test_name);
		diag("SQL: $sql");
		diag("Expected query to fail but it succeeded");
		return 0;
	}
	
	my $error_msg = $result->{stderr} || '';
	my $matches = $error_msg =~ $pattern;
	
	ok($matches, $test_name);
	
	unless ($matches) {
		diag("Pattern: $pattern");
		diag("Error message: $error_msg");
	}
	
	return $matches;
}

=head2 table_row_count_is

Verify a table has the expected number of rows.

=cut

sub table_row_count_is {
	my ($node, $dbname, $table_name, $expected_count, $test_name) = @_;
	$test_name ||= "Table $table_name has $expected_count rows";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT COUNT(*) FROM $table_name;",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		diag("Error: $result->{stderr}");
		return 0;
	}
	
	my $actual_count = $result->{stdout};
	$actual_count =~ s/^\s+|\s+$//g;
	chomp $actual_count;
	
	is($actual_count, $expected_count, $test_name);
	return $actual_count == $expected_count;
}

=head2 column_exists_ok

Verify a column exists in a table.

=cut

sub column_exists_ok {
	my ($node, $dbname, $schemaname, $tablename, $columnname, $test_name) = @_;
	$test_name ||= "Column $schemaname.$tablename.$columnname exists";
	$dbname ||= 'postgres';
	
	my $schema_part = $schemaname ? "$schemaname." : "";
	my $result = $node->psql($dbname,
		"SELECT column_name FROM information_schema.columns 
		 WHERE table_schema = '$schemaname' AND table_name = '$tablename' AND column_name = '$columnname';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	chomp $output;
	$output =~ s/^\s+|\s+$//g;
	
	is($output, $columnname, $test_name);
	return $output eq $columnname;
}

=head2 index_exists_ok

Verify an index exists.

=cut

sub index_exists_ok {
	my ($node, $dbname, $schemaname, $indexname, $test_name) = @_;
	$test_name ||= "Index $schemaname.$indexname exists";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT indexname FROM pg_indexes 
		 WHERE schemaname = '$schemaname' AND indexname = '$indexname';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	chomp $output;
	$output =~ s/^\s+|\s+$//g;
	
	is($output, $indexname, $test_name);
	return $output eq $indexname;
}

=head2 trigger_exists_ok

Verify a trigger exists.

=cut

sub trigger_exists_ok {
	my ($node, $dbname, $schemaname, $tablename, $triggername, $test_name) = @_;
	$test_name ||= "Trigger $schemaname.$tablename.$triggername exists";
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT trigger_name FROM information_schema.triggers 
		 WHERE trigger_schema = '$schemaname' AND event_object_table = '$tablename' AND trigger_name = '$triggername';",
		tuples_only => 1
	);
	
	unless ($result->{success}) {
		fail($test_name);
		return 0;
	}
	
	my $output = $result->{stdout};
	chomp $output;
	$output =~ s/^\s+|\s+$//g;
	
	is($output, $triggername, $test_name);
	return $output eq $triggername;
}

1;


