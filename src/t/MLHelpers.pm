package MLHelpers;

use strict;
use warnings;
use PostgresNode;
use TapTest;
use Exporter 'import';

our @EXPORT = qw(
	train_ml_model
	predict_ml_model
	evaluate_ml_model
	create_regression_dataset
	create_classification_dataset
	create_clustering_dataset
	test_ml_algorithm
	validate_ml_result
	get_model_info
	cleanup_ml_models
	test_cross_validation
	test_hyperparameters
);

=head1 NAME

MLHelpers - Machine Learning test helpers

=head1 SYNOPSIS

  use MLHelpers;
  use PostgresNode;

  my $node = PostgresNode->new('test');
  $node->init();
  $node->start();

  train_ml_model($node, 'postgres', 'linear_regression', ...);
  predict_ml_model($node, 'postgres', 'my_model', ...);

=head1 DESCRIPTION

Provides comprehensive test helpers for ML algorithms including
training, prediction, evaluation, and dataset generation.

=cut

=head2 train_ml_model

Train an ML model with given parameters.

=cut

sub train_ml_model {
	my ($node, $dbname, $algorithm, %params) = @_;
	$dbname ||= 'postgres';
	
	my $train_table = $params{train_table} || 'train_data';
	my $feature_col = $params{feature_col} || 'features';
	my $label_col = $params{label_col} || 'label';
	my $model_name = $params{model_name} || "test_${algorithm}_model";
	my $options = $params{options} || '{}';
	
	my $train_sql = qq{
		SELECT neurondb.train(
			'$algorithm',
			'$train_table',
			'$feature_col',
			'$label_col',
			'$options'::jsonb
		);
	};
	
	my $result = $node->psql($dbname, $train_sql);
	
	if ($result->{success}) {
		return (1, $model_name);
	} else {
		return (0, "Training failed: $result->{stderr}");
	}
}

=head2 predict_ml_model

Make predictions using a trained model.

=cut

sub predict_ml_model {
	my ($node, $dbname, $model_name, %params) = @_;
	$dbname ||= 'postgres';
	
	my $test_table = $params{test_table} || 'test_data';
	my $feature_col = $params{feature_col} || 'features';
	
	my $predict_sql = qq{
		SELECT neurondb.predict(
			'$model_name',
			'$test_table',
			'$feature_col'
		);
	};
	
	my $result = $node->psql($dbname, $predict_sql);
	
	if ($result->{success}) {
		return (1, "Prediction successful");
	} else {
		return (0, "Prediction failed: $result->{stderr}");
	}
}

=head2 evaluate_ml_model

Evaluate a trained model.

=cut

sub evaluate_ml_model {
	my ($node, $dbname, $model_name, %params) = @_;
	$dbname ||= 'postgres';
	
	my $test_table = $params{test_table} || 'test_data';
	my $feature_col = $params{feature_col} || 'features';
	my $label_col = $params{label_col} || 'label';
	
	my $eval_sql = qq{
		SELECT neurondb.evaluate(
			'$model_name',
			'$test_table',
			'$feature_col',
			'$label_col'
		);
	};
	
	my $result = $node->psql($dbname, $eval_sql);
	
	if ($result->{success}) {
		return (1, "Evaluation successful");
	} else {
		return (0, "Evaluation failed: $result->{stderr}");
	}
}

=head2 create_regression_dataset

Create a dataset for regression testing.

=cut

sub create_regression_dataset {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $table = $params{table} || 'regression_data';
	my $num_rows = $params{num_rows} || 100;
	my $dim = $params{dim} || 3;
	my $noise = $params{noise} || 0.1;
	
	# Drop existing table
	$node->psql($dbname, "DROP TABLE IF EXISTS $table CASCADE;");
	
	# Create table
	my $create_sql = qq{
		CREATE TABLE $table (
			id SERIAL PRIMARY KEY,
			features vector($dim),
			label REAL
		);
	};
	
	my $result = $node->psql($dbname, $create_sql);
	unless ($result->{success}) {
		die "Failed to create regression table: $result->{stderr}\n";
	}
	
	# Generate data: label = sum of features + noise
	my @values;
	for my $i (1..$num_rows) {
		my @coords;
		my $label_sum = 0;
		for my $j (1..$dim) {
			my $val = ($i * 0.1 + $j * 0.01) + rand($noise);
			push @coords, sprintf("%.6f", $val);
			$label_sum += $val;
		}
		my $vec = '[' . join(',', @coords) . ']';
		my $label = $label_sum + rand($noise);
		push @values, "('$vec'::vector($dim), $label)";
	}
	
	my $insert_sql = "INSERT INTO $table (features, label) VALUES " 
		. join(', ', @values) . ';';
	
	$result = $node->psql($dbname, $insert_sql);
	unless ($result->{success}) {
		die "Failed to insert regression data: $result->{stderr}\n";
	}
	
	return 1;
}

=head2 create_classification_dataset

Create a dataset for classification testing.

=cut

sub create_classification_dataset {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $table = $params{table} || 'classification_data';
	my $num_rows = $params{num_rows} || 100;
	my $dim = $params{dim} || 3;
	my $num_classes = $params{num_classes} || 2;
	
	# Drop existing table
	$node->psql($dbname, "DROP TABLE IF EXISTS $table CASCADE;");
	
	# Create table
	my $create_sql = qq{
		CREATE TABLE $table (
			id SERIAL PRIMARY KEY,
			features vector($dim),
			label INTEGER
		);
	};
	
	my $result = $node->psql($dbname, $create_sql);
	unless ($result->{success}) {
		die "Failed to create classification table: $result->{stderr}\n";
	}
	
	# Generate data: clusters for each class
	my @values;
	for my $i (1..$num_rows) {
		my $class = ($i - 1) % $num_classes;
		my @coords;
		for my $j (1..$dim) {
			# Center each class at different points
			my $center = $class * 10.0;
			my $val = $center + ($i * 0.1 + $j * 0.01) + rand(2.0);
			push @coords, sprintf("%.6f", $val);
		}
		my $vec = '[' . join(',', @coords) . ']';
		push @values, "('$vec'::vector($dim), $class)";
	}
	
	my $insert_sql = "INSERT INTO $table (features, label) VALUES " 
		. join(', ', @values) . ';';
	
	$result = $node->psql($dbname, $insert_sql);
	unless ($result->{success}) {
		die "Failed to insert classification data: $result->{stderr}\n";
	}
	
	return 1;
}

=head2 create_clustering_dataset

Create a dataset for clustering testing.

=cut

sub create_clustering_dataset {
	my ($node, $dbname, %params) = @_;
	$dbname ||= 'postgres';
	
	my $table = $params{table} || 'clustering_data';
	my $num_rows = $params{num_rows} || 100;
	my $dim = $params{dim} || 3;
	my $num_clusters = $params{num_clusters} || 3;
	
	# Drop existing table
	$node->psql($dbname, "DROP TABLE IF EXISTS $table CASCADE;");
	
	# Create table
	my $create_sql = qq{
		CREATE TABLE $table (
			id SERIAL PRIMARY KEY,
			features vector($dim)
		);
	};
	
	my $result = $node->psql($dbname, $create_sql);
	unless ($result->{success}) {
		die "Failed to create clustering table: $result->{stderr}\n";
	}
	
	# Generate clustered data
	my @values;
	for my $i (1..$num_rows) {
		my $cluster = ($i - 1) % $num_clusters;
		my @coords;
		for my $j (1..$dim) {
			# Center each cluster at different points
			my $center = $cluster * 10.0;
			my $val = $center + sin($i * 0.1 + $j * 0.01) * 2 + rand(1.0);
			push @coords, sprintf("%.6f", $val);
		}
		my $vec = '[' . join(',', @coords) . ']';
		push @values, "('$vec'::vector($dim))";
	}
	
	my $insert_sql = "INSERT INTO $table (features) VALUES " 
		. join(', ', @values) . ';';
	
	$result = $node->psql($dbname, $insert_sql);
	unless ($result->{success}) {
		die "Failed to insert clustering data: $result->{stderr}\n";
	}
	
	return 1;
}

=head2 test_ml_algorithm

Test a complete ML algorithm workflow.

=cut

sub test_ml_algorithm {
	my ($node, $dbname, $algorithm, %params) = @_;
	$dbname ||= 'postgres';
	
	# Create datasets
	if ($algorithm =~ /regression|ridge|lasso/) {
		create_regression_dataset($node, $dbname, 
			table => 'train_reg',
			num_rows => $params{train_rows} || 50,
			dim => $params{dim} || 3
		);
		create_regression_dataset($node, $dbname,
			table => 'test_reg',
			num_rows => $params{test_rows} || 10,
			dim => $params{dim} || 3
		);
		
		# Train
		my ($success, $msg) = train_ml_model($node, $dbname, $algorithm,
			train_table => 'train_reg',
			feature_col => 'features',
			label_col => 'label',
			model_name => "test_${algorithm}",
			options => $params{options} || '{}'
		);
		
		unless ($success) {
			return (0, $msg);
		}
		
		# Predict
		($success, $msg) = predict_ml_model($node, $dbname, "test_${algorithm}",
			test_table => 'test_reg',
			feature_col => 'features'
		);
		
		unless ($success) {
			return (0, $msg);
		}
		
		# Evaluate
		if ($params{evaluate}) {
			($success, $msg) = evaluate_ml_model($node, $dbname, "test_${algorithm}",
				test_table => 'test_reg',
				feature_col => 'features',
				label_col => 'label'
			);
			
			unless ($success) {
				return (0, $msg);
			}
		}
		
		# Cleanup
		$node->psql($dbname, "DROP TABLE IF EXISTS train_reg CASCADE;");
		$node->psql($dbname, "DROP TABLE IF EXISTS test_reg CASCADE;");
		
		return (1, "Algorithm $algorithm tested successfully");
	} elsif ($algorithm =~ /classification|logistic|svm|naive_bayes/) {
		create_classification_dataset($node, $dbname,
			table => 'train_cls',
			num_rows => $params{train_rows} || 50,
			dim => $params{dim} || 3,
			num_classes => $params{num_classes} || 2
		);
		create_classification_dataset($node, $dbname,
			table => 'test_cls',
			num_rows => $params{test_rows} || 10,
			dim => $params{dim} || 3,
			num_classes => $params{num_classes} || 2
		);
		
		# Similar workflow as regression
		my ($success, $msg) = train_ml_model($node, $dbname, $algorithm,
			train_table => 'train_cls',
			feature_col => 'features',
			label_col => 'label',
			model_name => "test_${algorithm}",
			options => $params{options} || '{}'
		);
		
		unless ($success) {
			return (0, $msg);
		}
		
		($success, $msg) = predict_ml_model($node, $dbname, "test_${algorithm}",
			test_table => 'test_cls',
			feature_col => 'features'
		);
		
		unless ($success) {
			return (0, $msg);
		}
		
		$node->psql($dbname, "DROP TABLE IF EXISTS train_cls CASCADE;");
		$node->psql($dbname, "DROP TABLE IF EXISTS test_cls CASCADE;");
		
		return (1, "Algorithm $algorithm tested successfully");
	}
	
	return (0, "Unknown algorithm type: $algorithm");
}

=head2 validate_ml_result

Validate ML result structure and values.

=cut

sub validate_ml_result {
	my ($node, $dbname, $sql, %checks) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname, $sql);
	
	unless ($result->{success}) {
		return (0, "Query failed: $result->{stderr}");
	}
	
	my $output = $result->{stdout};
	
	# Check for expected columns
	if (exists $checks{columns}) {
		for my $col (@{$checks{columns}}) {
			unless ($output =~ /$col/) {
				return (0, "Missing column: $col");
			}
		}
	}
	
	# Check for numeric values
	if (exists $checks{numeric}) {
		unless ($output =~ /\d+\.?\d*/) {
			return (0, "No numeric values found");
		}
	}
	
	return (1, "Validation passed");
}

=head2 get_model_info

Get information about a trained model.

=cut

sub get_model_info {
	my ($node, $dbname, $model_name) = @_;
	$dbname ||= 'postgres';
	
	my $result = $node->psql($dbname,
		"SELECT * FROM neurondb.model_catalog WHERE model_name = '$model_name';",
		tuples_only => 1
	);
	
	if ($result->{success}) {
		return $result->{stdout};
	}
	
	return undef;
}

=head2 cleanup_ml_models

Clean up test ML models.

=cut

sub cleanup_ml_models {
	my ($node, $dbname, $prefix) = @_;
	$dbname ||= 'postgres';
	$prefix ||= 'test_';
	
	$node->psql($dbname,
		"DELETE FROM neurondb.model_catalog WHERE model_name LIKE '${prefix}%';"
	);
	
	return 1;
}

=head2 test_cross_validation

Test cross-validation for an ML algorithm.

=cut

sub test_cross_validation {
	my ($node, $dbname, $algorithm, %params) = @_;
	$dbname ||= 'postgres';
	
	# Create dataset
	create_regression_dataset($node, $dbname,
		table => 'cv_data',
		num_rows => $params{num_rows} || 100,
		dim => $params{dim} || 3
	);
	
	# Train with cross-validation options
	my $options = '{"cv_folds": 5, "cv_metric": "mse"}' unless $params{options};
	$options ||= $params{options};
	
	my ($success, $msg) = train_ml_model($node, $dbname, $algorithm,
		train_table => 'cv_data',
		feature_col => 'features',
		label_col => 'label',
		model_name => "test_${algorithm}_cv",
		options => $options
	);
	
	$node->psql($dbname, "DROP TABLE IF EXISTS cv_data CASCADE;");
	
	return ($success, $msg);
}

=head2 test_hyperparameters

Test different hyperparameter combinations.

=cut

sub test_hyperparameters {
	my ($node, $dbname, $algorithm, %params) = @_;
	$dbname ||= 'postgres';
	
	# Create dataset
	create_regression_dataset($node, $dbname,
		table => 'hp_data',
		num_rows => $params{num_rows} || 50,
		dim => $params{dim} || 3
	);
	
	my @results;
	my $hyperparams = $params{hyperparams} || [];
	
	for my $hp (@$hyperparams) {
		my $options = $hp->{options} || '{}';
		my $model_name = "test_${algorithm}_hp_" . ($hp->{name} || 'default');
		
		my ($success, $msg) = train_ml_model($node, $dbname, $algorithm,
			train_table => 'hp_data',
			feature_col => 'features',
			label_col => 'label',
			model_name => $model_name,
			options => $options
		);
		
		push @results, {
			hyperparams => $hp,
			success => $success,
			message => $msg
		};
	}
	
	$node->psql($dbname, "DROP TABLE IF EXISTS hp_data CASCADE;");
	
	return \@results;
}

1;



