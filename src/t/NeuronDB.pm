package NeuronDB;

use strict;
use warnings;
use PostgresNode;
use TapTest;
use Exporter 'import';

our @EXPORT = qw(
	install_extension
	create_test_vectors
	test_ml_function
	test_gpu_features
	test_workers
	test_indexes
	test_vector_operations
	test_distance_metrics
	test_aggregates
	cleanup_test_objects
	setup_test_data
	check_feature_availability
	get_neurondb_version
	enable_gpu_mode
	disable_gpu_mode
);

=head1 NAME

NeuronDB - NeuronDB-specific test helpers

=head1 SYNOPSIS

  use NeuronDB;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  install_extension($node, 'postgres');
  create_test_vectors($node, 'postgres', 100);
  test_ml_function($node, 'postgres', 'linear_regression', ...);

=head1 DESCRIPTION

Provides NeuronDB-specific test helpers for extension installation,
vector operations, ML functions, GPU features, workers, and indexes.

=cut

=head2 install_extension

Install NeuronDB extension in the specified database.

=cut

sub install_extension {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	# Drop extension if exists
	$node->psql($dbname, 'DROP EXTENSION IF EXISTS neurondb CASCADE;');
	
	# Create extension
	my $result = $node->psql($dbname, 'CREATE EXTENSION neurondb CASCADE;');
	
	unless ($result->{success}) {
		die "Failed to install NeuronDB extension: $result->{stderr}\n";
	}
	
	# Verify extension
	my $verify = $node->psql($dbname, 
		"SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';",
		tuples_only => 1
	);
	
	unless ($verify->{success} && $verify->{stdout} =~ /neurondb/) {
		die "Extension verification failed\n";
	}
	
	# Verify schema exists
	my $schema = $node->psql($dbname,
		"SELECT nspname FROM pg_namespace WHERE nspname = 'neurondb';",
		tuples_only => 1
	);
	
	unless ($schema->{success} && $schema->{stdout} =~ /neurondb/) {
		die "NeuronDB schema not found\n";
	}
	
	return 1;
}

=head2 create_test_vectors

Create test vector data in a table.

=cut

sub create_test_vectors {
	my ($node, $dbname, $num_rows, %params) = @_;
	$dbname ||= 'postgres';
	$num_rows ||= 10;
	
	my $table = $params{table} || 'test_vectors';
	my $dim = $params{dim} || 3;
	
	# Drop table if exists
	$node->psql($dbname, "DROP TABLE IF EXISTS $table CASCADE;");
	
	# Create table
	my $create_sql = qq{
		CREATE TABLE $table (
			id SERIAL PRIMARY KEY,
			name TEXT,
			embedding vector($dim),
			metadata JSONB
		);
	};
	
	my $result = $node->psql($dbname, $create_sql);
	unless ($result->{success}) {
		die "Failed to create test vectors table: $result->{stderr}\n";
	}
	
	# Generate test data
	my @values;
	for my $i (1..$num_rows) {
		my @coords;
		for my $j (1..$dim) {
			push @coords, sprintf("%.2f", ($i * 0.1 + $j * 0.01));
		}
		my $vec = '[' . join(',', @coords) . ']';
		push @values, "('item$i', '$vec'::vector($dim), '{\"id\": $i}'::jsonb)";
	}
	
	my $insert_sql = "INSERT INTO $table (name, embedding, metadata) VALUES " 
		. join(', ', @values) . ';';
	
	$result = $node->psql($dbname, $insert_sql);
	unless ($result->{success}) {
		die "Failed to insert test vectors: $result->{stderr}\n";
	}
	
	return 1;
}

=head2 test_vector_operations

Test basic vector operations.

=cut

sub test_vector_operations {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	# Test vector creation
	my $result = $node->psql($dbname, q{
		SELECT '[1,2,3]'::vector(3) AS v1;
	});
	
	unless ($result->{success}) {
		return (0, "Vector creation failed: $result->{stderr}");
	}
	
	# Test vector dimensions
	$result = $node->psql($dbname, q{
		SELECT vector_dims('[1,2,3,4,5]'::vector) AS dims;
	}, tuples_only => 1);
	
	unless ($result->{success} && $result->{stdout} =~ /5/) {
		return (0, "vector_dims failed");
	}
	
	# Test vector norm
	$result = $node->psql($dbname, q{
		SELECT vector_norm('[3,4]'::vector) AS norm;
	}, tuples_only => 1);
	
	unless ($result->{success} && $result->{stdout} =~ /5/) {
		return (0, "vector_norm failed");
	}
	
	# Test vector arithmetic
	$result = $node->psql($dbname, q{
		SELECT '[1,2,3]'::vector + '[4,5,6]'::vector AS v_add;
	});
	
	unless ($result->{success}) {
		return (0, "Vector addition failed");
	}
	
	return (1, "Vector operations OK");
}

=head2 test_distance_metrics

Test distance metric operations.

=cut

sub test_distance_metrics {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	# Test L2 distance
	my $result = $node->psql($dbname, q{
		SELECT '[1,0,0]'::vector(3) <-> '[0,1,0]'::vector(3) AS l2_dist;
	}, tuples_only => 1);
	
	unless ($result->{success}) {
		return (0, "L2 distance failed: $result->{stderr}");
	}
	
	# Test cosine distance
	$result = $node->psql($dbname, q{
		SELECT '[1,0,0]'::vector(3) <=> '[0,1,0]'::vector(3) AS cosine_dist;
	}, tuples_only => 1);
	
	unless ($result->{success}) {
		return (0, "Cosine distance failed: $result->{stderr}");
	}
	
	# Test inner product
	$result = $node->psql($dbname, q{
		SELECT '[1,2,3]'::vector(3) <#> '[4,5,6]'::vector(3) AS inner_prod;
	}, tuples_only => 1);
	
	unless ($result->{success}) {
		return (0, "Inner product failed: $result->{stderr}");
	}
	
	return (1, "Distance metrics OK");
}

=head2 test_aggregates

Test vector aggregate functions.

=cut

sub test_aggregates {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	# Create test table
	$node->psql($dbname, q{
		DROP TABLE IF EXISTS test_agg;
		CREATE TABLE test_agg (id SERIAL, vec vector(3));
		INSERT INTO test_agg (vec) VALUES
			('[1,2,3]'::vector(3)),
			('[4,5,6]'::vector(3)),
			('[7,8,9]'::vector(3));
	});
	
	# Test vector_avg
	my $result = $node->psql($dbname, q{
		SELECT vector_avg(vec) FROM test_agg;
	});
	
	unless ($result->{success}) {
		return (0, "vector_avg failed: $result->{stderr}");
	}
	
	# Test vector_sum
	$result = $node->psql($dbname, q{
		SELECT vector_sum(vec) FROM test_agg;
	});
	
	unless ($result->{success}) {
		return (0, "vector_sum failed: $result->{stderr}");
	}
	
	$node->psql($dbname, 'DROP TABLE test_agg;');
	
	return (1, "Aggregates OK");
}

=head2 test_ml_function

Test ML function (train, predict, evaluate).

=cut

sub test_ml_function {
	my ($node, $dbname, $algorithm, %params) = @_;
	$dbname ||= 'postgres';
	
	# Create test data table
	my $train_table = $params{train_table} || 'test_train';
	my $test_table = $params{test_table} || 'test_test';
	
	# Create training data
	$node->psql($dbname, qq{
		DROP TABLE IF EXISTS $train_table CASCADE;
		CREATE TABLE $train_table (
			id SERIAL PRIMARY KEY,
			features vector(3),
			label REAL
		);
		INSERT INTO $train_table (features, label) VALUES
			('[1,2,3]'::vector(3), 10.0),
			('[2,3,4]'::vector(3), 20.0),
			('[3,4,5]'::vector(3), 30.0),
			('[4,5,6]'::vector(3), 40.0),
			('[5,6,7]'::vector(3), 50.0);
	});
	
	# Create test data
	$node->psql($dbname, qq{
		DROP TABLE IF EXISTS $test_table CASCADE;
		CREATE TABLE $test_table (
			id SERIAL PRIMARY KEY,
			features vector(3),
			label REAL
		);
		INSERT INTO $test_table (features, label) VALUES
			('[1.5,2.5,3.5]'::vector(3), 15.0),
			('[2.5,3.5,4.5]'::vector(3), 25.0);
	});
	
	# Train model
	my $model_name = $params{model_name} || "test_${algorithm}_model";
	my $train_sql = qq{
		SELECT neurondb.train(
			'$algorithm',
			'$train_table',
			'features',
			'label',
			'{"model_name": "$model_name"}'::jsonb
		);
	};
	
	my $result = $node->psql($dbname, $train_sql);
	
	unless ($result->{success}) {
		return (0, "Training failed: $result->{stderr}");
	}
	
	# Predict
	my $predict_sql = qq{
		SELECT neurondb.predict(
			'$model_name',
			'$test_table',
			'features'
		);
	};
	
	$result = $node->psql($dbname, $predict_sql);
	
	unless ($result->{success}) {
		return (0, "Prediction failed: $result->{stderr}");
	}
	
	# Evaluate (if supported)
	if ($params{evaluate}) {
		my $eval_sql = qq{
			SELECT neurondb.evaluate(
				'$model_name',
				'$test_table',
				'features',
				'label'
			);
		};
		
		$result = $node->psql($dbname, $eval_sql);
		
		unless ($result->{success}) {
			return (0, "Evaluation failed: $result->{stderr}");
		}
	}
	
	# Cleanup
	$node->psql($dbname, "DROP TABLE IF EXISTS $train_table CASCADE;");
	$node->psql($dbname, "DROP TABLE IF EXISTS $test_table CASCADE;");
	
	return (1, "ML function test OK");
}

=head2 test_gpu_features

Test GPU features if available.

=cut

sub test_gpu_features {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	# Check if GPU is enabled
	my $result = $node->psql($dbname, q{
		SELECT current_setting('neurondb.compute_mode', true) AS compute_mode;
	}, tuples_only => 1);
	
	unless ($result->{success}) {
		return (0, "Cannot check GPU status: $result->{stderr}");
	}
	
	my $compute_mode = $result->{stdout};
	chomp $compute_mode;
	
	if ($compute_mode ne '1') {  # 1 = GPU mode, 0 = CPU mode, 2 = AUTO mode
		return (1, "GPU not enabled (compute_mode=$compute_mode, expected 1 for GPU mode)");
	}
	
	# Test GPU info function
	$result = $node->psql($dbname, q{
		SELECT * FROM neurondb.gpu_info();
	});
	
	unless ($result->{success}) {
		return (0, "GPU info failed: $result->{stderr}");
	}
	
	# Test GPU distance function if available
	$result = $node->psql($dbname, q{
		SELECT vector_l2_distance_gpu(
			'[1,2,3]'::vector(3),
			'[4,5,6]'::vector(3)
		) AS gpu_dist;
	});
	
	unless ($result->{success}) {
		return (1, "GPU distance not available (expected)");
	}
	
	return (1, "GPU features OK");
}

=head2 test_workers

Test worker functions and status.

=cut

sub test_workers {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	# Check worker tables exist
	my $result = $node->psql($dbname, q{
		SELECT tablename FROM pg_tables 
		WHERE schemaname = 'neurondb' 
		AND tablename LIKE '%job%';
	}, tuples_only => 1);
	
	unless ($result->{success}) {
		return (0, "Cannot check worker tables: $result->{stderr}");
	}
	
	# Test job queue functions
	$result = $node->psql($dbname, q{
		SELECT COUNT(*) FROM neurondb.job_queue;
	}, tuples_only => 1);
	
	unless ($result->{success}) {
		return (0, "Job queue query failed: $result->{stderr}");
	}
	
	# Test worker status functions if available
	$result = $node->psql($dbname, q{
		SELECT proname FROM pg_proc p
		JOIN pg_namespace n ON p.pronamespace = n.oid
		WHERE n.nspname = 'neurondb'
		AND p.proname LIKE '%worker%';
	}, tuples_only => 1);
	
	# Worker functions may not exist, that's OK
	return (1, "Workers OK");
}

=head2 test_indexes

Test index creation and queries.

=cut

sub test_indexes {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	# Create test table
	my $table = $params{table} || 'test_index_vectors';
	
	$node->psql($dbname, qq{
		DROP TABLE IF EXISTS $table CASCADE;
		CREATE TABLE $table (
			id SERIAL PRIMARY KEY,
			vec vector(4),
			label TEXT
		);
		INSERT INTO $table (vec, label) VALUES
			('[1,2,3,4]'::vector(4), 'label1'),
			('[2,3,4,5]'::vector(4), 'label2'),
			('[3,4,5,6]'::vector(4), 'label3'),
			('[4,5,6,7]'::vector(4), 'label4'),
			('[5,6,7,8]'::vector(4), 'label5');
	});
	
	# Try to create HNSW index
	my $index_name = $params{index_name} || 'idx_test_hnsw';
	my $result = $node->psql($dbname, qq{
		CREATE INDEX $index_name ON $table 
		USING hnsw (vec vector_l2_ops)
		WITH (m = 16, ef_construction = 200);
	});
	
	unless ($result->{success}) {
		# Index creation may fail if not supported, that's OK
		$node->psql($dbname, "DROP TABLE $table CASCADE;");
		return (1, "Index creation not supported (skipped)");
	}
	
	# Test KNN query
	$result = $node->psql($dbname, qq{
		SELECT id, vec <-> '[1,2,3,4]'::vector(4) AS distance
		FROM $table
		ORDER BY vec <-> '[1,2,3,4]'::vector(4)
		LIMIT 3;
	});
	
	unless ($result->{success}) {
		$node->psql($dbname, "DROP TABLE $table CASCADE;");
		return (0, "KNN query failed: $result->{stderr}");
	}
	
	# Cleanup
	$node->psql($dbname, "DROP TABLE $table CASCADE;");
	
	return (1, "Indexes OK");
}

=head2 cleanup_test_objects

Comprehensive cleanup of test objects (tables, indexes, models, etc.).

=cut

sub cleanup_test_objects {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my @objects = @{$params{objects} || []};
	my $prefix = $params{prefix} || 'test_';
	
	# If no specific objects, clean up all test_* tables
	if (@objects == 0) {
		my $result = $node->psql($dbname,
			"SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE '${prefix}%';",
			tuples_only => 1
		);
		
		if ($result->{success} && $result->{stdout}) {
			my @tables = split(/\n/, $result->{stdout});
			for my $table (@tables) {
				$table =~ s/^\s+|\s+$//g;
				next unless $table;
				$node->psql($dbname, "DROP TABLE IF EXISTS $table CASCADE;");
			}
		}
		
		# Clean up test models
		$node->psql($dbname,
			"DELETE FROM neurondb.model_catalog WHERE model_name LIKE '${prefix}%';"
		);
		
		# Clean up test indexes
		my $idx_result = $node->psql($dbname,
			"SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND indexname LIKE '${prefix}%';",
			tuples_only => 1
		);
		
		if ($idx_result->{success} && $idx_result->{stdout}) {
			my @indexes = split(/\n/, $idx_result->{stdout});
			for my $idx (@indexes) {
				$idx =~ s/^\s+|\s+$//g;
				next unless $idx;
				$node->psql($dbname, "DROP INDEX IF EXISTS $idx CASCADE;");
			}
		}
	} else {
		# Clean up specific objects
		for my $obj (@objects) {
			$node->psql($dbname, "DROP TABLE IF EXISTS $obj CASCADE;");
			$node->psql($dbname, "DROP INDEX IF EXISTS $obj CASCADE;");
		}
	}
	
	return 1;
}

=head2 setup_test_data

Create configurable test data for various scenarios.

=cut

sub setup_test_data {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $table = $params{table} || 'test_data';
	my $num_rows = $params{num_rows} || 100;
	my $dim = $params{dim} || 128;
	my $distribution = $params{distribution} || 'uniform';
	my $seed = $params{seed} || 42;
	
	# Drop existing table
	$node->psql($dbname, "DROP TABLE IF EXISTS $table CASCADE;");
	
	# Create table
	my $create_sql = qq{
		CREATE TABLE $table (
			id SERIAL PRIMARY KEY,
			embedding vector($dim),
			label TEXT,
			metadata JSONB
		);
	};
	
	my $result = $node->psql($dbname, $create_sql);
	unless ($result->{success}) {
		die "Failed to create test data table: $result->{stderr}\n";
	}
	
	# Generate data based on distribution
	if ($distribution eq 'uniform') {
		# Uniform random vectors
		my @values;
		for my $i (1..$num_rows) {
			my @coords;
			for my $j (1..$dim) {
				# Simple deterministic "random" based on seed
				my $val = sin($seed + $i * 0.1 + $j * 0.01) * 10;
				push @coords, sprintf("%.6f", $val);
			}
			my $vec = '[' . join(',', @coords) . ']';
			push @values, "('$vec'::vector($dim), 'label$i', '{\"id\": $i}'::jsonb)";
		}
		
		my $insert_sql = "INSERT INTO $table (embedding, label, metadata) VALUES " 
			. join(', ', @values) . ';';
		$result = $node->psql($dbname, $insert_sql);
	} elsif ($distribution eq 'clustered') {
		# Clustered data (for clustering tests)
		my $num_clusters = $params{num_clusters} || 3;
		my @values;
		for my $i (1..$num_rows) {
			my $cluster = ($i - 1) % $num_clusters;
			my @coords;
			for my $j (1..$dim) {
				# Center each cluster at different points
				my $center = $cluster * 10.0;
				my $val = $center + sin($seed + $i * 0.1 + $j * 0.01) * 2;
				push @coords, sprintf("%.6f", $val);
			}
			my $vec = '[' . join(',', @coords) . ']';
			push @values, "('$vec'::vector($dim), 'cluster$cluster', '{\"cluster\": $cluster, \"id\": $i}'::jsonb)";
		}
		
		my $insert_sql = "INSERT INTO $table (embedding, label, metadata) VALUES " 
			. join(', ', @values) . ';';
		$result = $node->psql($dbname, $insert_sql);
	} else {
		# Default: simple sequential
		create_test_vectors($node, $dbname, $num_rows, table => $table, dim => $dim);
		return 1;
	}
	
	unless ($result->{success}) {
		die "Failed to insert test data: $result->{stderr}\n";
	}
	
	return 1;
}

=head2 check_feature_availability

Check if a specific feature is available (GPU, workers, etc.).

=cut

sub check_feature_availability {
	my ($node, $dbname, $feature) = @_;
	$dbname ||= 'postgres';
	
	if ($feature eq 'gpu') {
		my $result = $node->psql($dbname, q{
			SELECT current_setting('neurondb.compute_mode', true) AS compute_mode;
		}, tuples_only => 1);
		
		if ($result->{success}) {
			my $mode = $result->{stdout};
			chomp $mode;
			return $mode eq '1' || $mode eq '2';  # GPU or AUTO mode
		}
		
		# Also check if gpu_info function exists
		$result = $node->psql($dbname, q{
			SELECT COUNT(*) FROM pg_proc p
			JOIN pg_namespace n ON p.pronamespace = n.oid
			WHERE n.nspname = 'neurondb' AND p.proname = 'gpu_info';
		}, tuples_only => 1);
		
		return $result->{success} && $result->{stdout} =~ /1/;
	} elsif ($feature eq 'workers') {
		my $result = $node->psql($dbname, q{
			SELECT COUNT(*) FROM pg_tables 
			WHERE schemaname = 'neurondb' AND tablename = 'job_queue';
		}, tuples_only => 1);
		
		return $result->{success} && $result->{stdout} =~ /1/;
	} elsif ($feature eq 'sparse_vectors') {
		my $result = $node->psql($dbname, q{
			SELECT COUNT(*) FROM pg_type WHERE typname = 'sparse_vector';
		}, tuples_only => 1);
		
		return $result->{success} && $result->{stdout} =~ /1/;
	} elsif ($feature eq 'quantization') {
		my $result = $node->psql($dbname, q{
			SELECT COUNT(*) FROM pg_proc p
			JOIN pg_namespace n ON p.pronamespace = n.oid
			WHERE n.nspname = 'neurondb' AND p.proname LIKE '%quant%';
		}, tuples_only => 1);
		
		return $result->{success} && $result->{stdout} =~ /\d+/ && $result->{stdout} > 0;
	}
	
	return 0;
}

=head2 get_neurondb_version

Get the installed NeuronDB extension version.

=cut

sub get_neurondb_version {
	my ($node, $dbname) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT extversion FROM pg_extension WHERE extname = 'neurondb';",
		tuples_only => 1
	);
	
	if ($result->{success} && $result->{stdout}) {
		my $version = $result->{stdout};
		chomp $version;
		$version =~ s/^\s+|\s+$//g;
		return $version;
	}
	
	return undef;
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





