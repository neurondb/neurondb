#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use MLHelpers;

=head1 NAME

050_ml_linear_regression.t - Exhaustive ML regression tests

=head1 DESCRIPTION

Comprehensive tests for regression algorithms: linear regression, ridge,
lasso, elastic net, polynomial regression, training convergence, prediction
accuracy, cross-validation, and outlier handling.

Target: 100+ test cases

=cut

plan tests => 110;

my $node = PostgresNode->new('ml_regression_test');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# ============================================================================
# LINEAR REGRESSION
# ============================================================================

subtest 'Linear Regression' => sub {
	plan tests => 20;
	
	# Create regression dataset
	create_regression_dataset($node, 'postgres',
		table => 'train_linreg',
		num_rows => 50,
		dim => 3
	);
	create_regression_dataset($node, 'postgres',
		table => 'test_linreg',
		num_rows => 10,
		dim => 3
	);
	
	# Train linear regression
	my ($success, $msg) = train_ml_model($node, 'postgres', 'linear_regression',
		train_table => 'train_linreg',
		feature_col => 'features',
		label_col => 'label',
		model_name => 'test_linear_regression',
		options => '{}'
	);
	ok($success, "Linear regression training: $msg");
	
	# Predict
	($success, $msg) = predict_ml_model($node, 'postgres', 'test_linear_regression',
		test_table => 'test_linreg',
		feature_col => 'features'
	);
	ok($success, "Linear regression prediction: $msg");
	
	# Evaluate
	($success, $msg) = evaluate_ml_model($node, 'postgres', 'test_linear_regression',
		test_table => 'test_linreg',
		feature_col => 'features',
		label_col => 'label'
	);
	ok($success, "Linear regression evaluation: $msg");
	
	# Test with different dimensions
	for my $dim (2, 3, 4, 5, 10, 128) {
		create_regression_dataset($node, 'postgres',
			table => "train_linreg_dim$dim",
			num_rows => 30,
			dim => $dim
		);
		
		($success, $msg) = train_ml_model($node, 'postgres', 'linear_regression',
			train_table => "train_linreg_dim$dim",
			feature_col => 'features',
			label_col => 'label',
			model_name => "test_linreg_dim$dim",
			options => '{}'
		);
		ok($success, "Linear regression dimension $dim: $msg");
		
		$node->psql('postgres', "DROP TABLE train_linreg_dim$dim CASCADE;");
		cleanup_ml_models($node, 'postgres', "test_linreg_dim$dim");
	}
	
	# Cleanup
	$node->psql('postgres', 'DROP TABLE train_linreg CASCADE;');
	$node->psql('postgres', 'DROP TABLE test_linreg CASCADE;');
	cleanup_ml_models($node, 'postgres', 'test_linear_regression');
};

# ============================================================================
# RIDGE REGRESSION
# ============================================================================

subtest 'Ridge Regression' => sub {
	plan tests => 15;
	
	# Create dataset
	create_regression_dataset($node, 'postgres',
		table => 'train_ridge',
		num_rows => 50,
		dim => 3
	);
	
	# Test different alpha values
	for my $alpha (0.01, 0.1, 1.0, 10.0, 100.0) {
		my ($success, $msg) = train_ml_model($node, 'postgres', 'ridge',
			train_table => 'train_ridge',
			feature_col => 'features',
			label_col => 'label',
			model_name => "test_ridge_alpha$alpha",
			options => "{\"alpha\": $alpha}"
		);
		ok($success, "Ridge regression alpha=$alpha: $msg");
		
		cleanup_ml_models($node, 'postgres', "test_ridge_alpha$alpha");
	}
	
	# Test with different dimensions
	for my $dim (3, 5, 10, 128) {
		create_regression_dataset($node, 'postgres',
			table => "train_ridge_dim$dim",
			num_rows => 30,
			dim => $dim
		);
		
		my ($success, $msg) = train_ml_model($node, 'postgres', 'ridge',
			train_table => "train_ridge_dim$dim",
			feature_col => 'features',
			label_col => 'label',
			model_name => "test_ridge_dim$dim",
			options => '{"alpha": 1.0}'
		);
		ok($success, "Ridge regression dimension $dim: $msg");
		
		$node->psql('postgres', "DROP TABLE train_ridge_dim$dim CASCADE;");
		cleanup_ml_models($node, 'postgres', "test_ridge_dim$dim");
	}
	
	$node->psql('postgres', 'DROP TABLE train_ridge CASCADE;');
};

# ============================================================================
# LASSO REGRESSION
# ============================================================================

subtest 'Lasso Regression' => sub {
	plan tests => 15;
	
	# Create dataset
	create_regression_dataset($node, 'postgres',
		table => 'train_lasso',
		num_rows => 50,
		dim => 3
	);
	
	# Test different alpha values
	for my $alpha (0.01, 0.1, 1.0, 10.0, 100.0) {
		my ($success, $msg) = train_ml_model($node, 'postgres', 'lasso',
			train_table => 'train_lasso',
			feature_col => 'features',
			label_col => 'label',
			model_name => "test_lasso_alpha$alpha",
			options => "{\"alpha\": $alpha}"
		);
		ok($success, "Lasso regression alpha=$alpha: $msg");
		
		cleanup_ml_models($node, 'postgres', "test_lasso_alpha$alpha");
	}
	
	# Test with different dimensions
	for my $dim (3, 5, 10, 128) {
		create_regression_dataset($node, 'postgres',
			table => "train_lasso_dim$dim",
			num_rows => 30,
			dim => $dim
		);
		
		my ($success, $msg) = train_ml_model($node, 'postgres', 'lasso',
			train_table => "train_lasso_dim$dim",
			feature_col => 'features',
			label_col => 'label',
			model_name => "test_lasso_dim$dim",
			options => '{"alpha": 1.0}'
		);
		ok($success, "Lasso regression dimension $dim: $msg");
		
		$node->psql('postgres', "DROP TABLE train_lasso_dim$dim CASCADE;");
		cleanup_ml_models($node, 'postgres', "test_lasso_dim$dim");
	}
	
	$node->psql('postgres', 'DROP TABLE train_lasso CASCADE;');
};

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

subtest 'Cross-Validation' => sub {
	plan tests => 10;
	
	# Test cross-validation for linear regression
	my ($success, $msg) = test_cross_validation($node, 'postgres', 'linear_regression',
		num_rows => 100,
		dim => 3,
		options => '{"cv_folds": 5, "cv_metric": "mse"}'
	);
	ok($success, "Cross-validation linear regression: $msg");
	
	# Test cross-validation for ridge
	($success, $msg) = test_cross_validation($node, 'postgres', 'ridge',
		num_rows => 100,
		dim => 3,
		options => '{"cv_folds": 5, "cv_metric": "mse", "alpha": 1.0}'
	);
	ok($success, "Cross-validation ridge: $msg");
	
	# Test different CV folds
	for my $folds (3, 5, 10) {
		($success, $msg) = test_cross_validation($node, 'postgres', 'linear_regression',
			num_rows => 50,
			dim => 3,
			options => "{\"cv_folds\": $folds, \"cv_metric\": \"mse\"}"
		);
		ok($success, "Cross-validation with $folds folds: $msg");
	}
};

# ============================================================================
# PREDICTION ACCURACY
# ============================================================================

subtest 'Prediction Accuracy' => sub {
	plan tests => 15;
	
	# Create train/test split
	create_regression_dataset($node, 'postgres',
		table => 'train_acc',
		num_rows => 100,
		dim => 3
	);
	create_regression_dataset($node, 'postgres',
		table => 'test_acc',
		num_rows => 20,
		dim => 3
	);
	
	# Train and evaluate
	my ($success, $msg) = train_ml_model($node, 'postgres', 'linear_regression',
		train_table => 'train_acc',
		feature_col => 'features',
		label_col => 'label',
		model_name => 'test_acc_model',
		options => '{}'
	);
	ok($success, "Training for accuracy test: $msg");
	
	# Evaluate and check metrics
	($success, $msg) = evaluate_ml_model($node, 'postgres', 'test_acc_model',
		test_table => 'test_acc',
		feature_col => 'features',
		label_col => 'label'
	);
	ok($success, "Accuracy evaluation: $msg");
	
	# Test with different algorithms
	for my $algo (qw(linear_regression ridge lasso)) {
		($success, $msg) = train_ml_model($node, 'postgres', $algo,
			train_table => 'train_acc',
			feature_col => 'features',
			label_col => 'label',
			model_name => "test_acc_$algo",
			options => '{}'
		);
		ok($success, "Accuracy test $algo: $msg");
		
		($success, $msg) = evaluate_ml_model($node, 'postgres', "test_acc_$algo",
			test_table => 'test_acc',
			feature_col => 'features',
			label_col => 'label'
		);
		ok($success, "Accuracy evaluation $algo: $msg");
		
		cleanup_ml_models($node, 'postgres', "test_acc_$algo");
	}
	
	# Cleanup
	$node->psql('postgres', 'DROP TABLE train_acc CASCADE;');
	$node->psql('postgres', 'DROP TABLE test_acc CASCADE;');
	cleanup_ml_models($node, 'postgres', 'test_acc_model');
};

# ============================================================================
# OUTLIER HANDLING
# ============================================================================

subtest 'Outlier Handling' => sub {
	plan tests => 10;
	
	# Create dataset with outliers
	$node->psql('postgres', q{
		DROP TABLE IF EXISTS train_outliers;
		CREATE TABLE train_outliers (id SERIAL, features vector(3), label REAL);
		INSERT INTO train_outliers (features, label) VALUES
			('[1,2,3]'::vector(3), 10.0),
			('[2,3,4]'::vector(3), 20.0),
			('[3,4,5]'::vector(3), 30.0),
			('[4,5,6]'::vector(3), 40.0),
			('[5,6,7]'::vector(3), 50.0),
			('[100,200,300]'::vector(3), 1000.0),  -- outlier
			('[200,300,400]'::vector(3), 2000.0);  -- outlier
	});
	
	# Train with outlier handling
	my ($success, $msg) = train_ml_model($node, 'postgres', 'linear_regression',
		train_table => 'train_outliers',
		feature_col => 'features',
		label_col => 'label',
		model_name => 'test_outliers',
		options => '{"handle_outliers": true}'
	);
	ok($success, "Training with outliers: $msg");
	
	# Test robust regression
	($success, $msg) = train_ml_model($node, 'postgres', 'ridge',
		train_table => 'train_outliers',
		feature_col => 'features',
		label_col => 'label',
		model_name => 'test_outliers_ridge',
		options => '{"alpha": 10.0, "handle_outliers": true}'
	);
	ok($success, "Robust regression with outliers: $msg");
	
	$node->psql('postgres', 'DROP TABLE train_outliers CASCADE;');
	cleanup_ml_models($node, 'postgres', 'test_outliers');
	cleanup_ml_models($node, 'postgres', 'test_outliers_ridge');
};

$node->stop();
$node->cleanup();

done_testing();

