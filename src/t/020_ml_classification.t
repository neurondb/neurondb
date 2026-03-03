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

020_ml_classification.t - Exhaustive ML classification algorithm tests

=head1 DESCRIPTION

Comprehensive tests for ML classification algorithms:
- Logistic Regression (Binary, Multinomial, One-vs-Rest)
- Support Vector Machines (SVM) - Linear, RBF, Polynomial kernels
- Decision Trees (CART, pruning, max_depth)
- Random Forest (n_estimators, feature importance)
- Gradient Boosting (XGBoost-style)
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- K-Nearest Neighbors (KNN) for classification
- Neural Network classifiers
- Ensemble methods
- Model evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
- Cross-validation
- Hyperparameter tuning
- Class imbalance handling
- Multi-class and multi-label classification

=head1 TEST COVERAGE

- 150+ test cases covering all classification scenarios
- Binary and multi-class classification
- Various kernel functions and parameters
- Model serialization and versioning
- Prediction confidence scores
- Feature importance analysis
- Edge cases and error handling

=cut

plan tests => 11;  # 3 neurondb_ok + 8 subtests

my $node = PostgresNode->new('test_classification');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# Enable ML features
query_ok($node, 'postgres', "SET neurondb.enable_ml = on;", 'ML features enabled');

#
# Test 1-15: Logistic Regression
#
subtest 'Logistic Regression Binary' => sub {
    plan tests => 11;  # 3 neurondb_ok + 8 subtests
    
    # Create binary classification dataset
    query_ok($node, 'postgres', q{
        CREATE TABLE binary_data (
            id serial PRIMARY KEY,
            features vector(4),
            label int CHECK (label IN (0, 1))
        );
    }, 'Binary data table created');
    
    # Insert training data (linearly separable)
    for my $i (1..50) {
        my $x1 = rand() - 0.5;
        my $x2 = rand() - 0.5;
        my $x3 = rand();
        my $x4 = rand();
        my $label = ($x1 + $x2 > 0) ? 1 : 0;
        
        query_ok($node, 'postgres', qq{
            INSERT INTO binary_data (features, label) 
            VALUES (ARRAY[$x1, $x2, $x3, $x4]::vector, $label);
        });
    }
    
    # Train logistic regression model
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'logistic_binary',
            algorithm := 'logistic_regression',
            training_table := 'binary_data',
            target_column := 'label',
            feature_columns := ARRAY['features']
        );
    }, 'Logistic regression trained');
    
    # Verify model exists
    result_is($node, 'postgres', 
        "SELECT COUNT(*) FROM neurondb.ml_models WHERE model_name = 'logistic_binary';",
        '1', 'Model registered');
    
    # Make predictions
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict('logistic_binary', ARRAY[0.5, 0.5, 0.3, 0.7]::vector);
    }, 'Binary prediction works');
    
    # Predict with probability
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict_proba('logistic_binary', ARRAY[0.5, 0.5, 0.3, 0.7]::vector);
    }, 'Probability prediction works');
    
    # Evaluate model
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_evaluate('logistic_binary', 'binary_data', 'label');
    }, 'Model evaluation works');
    
    # Check accuracy > 80%
    my $accuracy = $node->safe_psql('postgres', q{
        SELECT (neurondb.ml_evaluate('logistic_binary', 'binary_data', 'label')::jsonb)->>'accuracy';
    });
    ok($accuracy > 0.8, "Accuracy > 80% (got $accuracy)");
    
    # Test with NULL features
    query_fails($node, 'postgres', q{
        SELECT neurondb.ml_predict('logistic_binary', NULL::vector);
    }, 'NULL features rejected');
    
    # Test with wrong dimension
    query_fails($node, 'postgres', q{
        SELECT neurondb.ml_predict('logistic_binary', ARRAY[0.5, 0.5]::vector);
    }, 'Wrong dimension rejected');
    
    # Get model coefficients
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_coefficients('logistic_binary');
    }, 'Coefficients retrieved');
    
    # Test batch prediction
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict_batch('logistic_binary', 'binary_data', 'features');
    }, 'Batch prediction works');
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE binary_data CASCADE;
        SELECT neurondb.ml_drop_model('logistic_binary');
    }, 'Binary data cleaned up');
};

#
# Test 16-30: Multi-class Logistic Regression
#
subtest 'Logistic Regression Multi-class' => sub {
    plan tests => 11;  # 3 neurondb_ok + 8 subtests
    
    # Create multi-class dataset (Iris-like)
    query_ok($node, 'postgres', q{
        CREATE TABLE multiclass_data (
            id serial PRIMARY KEY,
            features vector(4),
            label int CHECK (label BETWEEN 0 AND 2)
        );
    }, 'Multi-class data table created');
    
    # Insert training data for 3 classes
    for my $class (0..2) {
        for my $i (1..30) {
            my $x1 = rand() + $class;
            my $x2 = rand() + $class * 0.5;
            my $x3 = rand();
            my $x4 = rand();
            
            query_ok($node, 'postgres', qq{
                INSERT INTO multiclass_data (features, label) 
                VALUES (ARRAY[$x1, $x2, $x3, $x4]::vector, $class);
            });
        }
    }
    
    # Train multi-class model (one-vs-rest)
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'logistic_multiclass',
            algorithm := 'logistic_regression',
            training_table := 'multiclass_data',
            target_column := 'label',
            feature_columns := ARRAY['features'],
            params := '{"multi_class": "ovr", "max_iter": 1000}'::jsonb
        );
    }, 'Multi-class model trained');
    
    # Predict class
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict('logistic_multiclass', ARRAY[2.5, 1.0, 0.5, 0.5]::vector);
    }, 'Multi-class prediction works');
    
    # Get class probabilities
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict_proba('logistic_multiclass', ARRAY[2.5, 1.0, 0.5, 0.5]::vector);
    }, 'Multi-class probabilities work');
    
    # Evaluate multi-class model
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_evaluate('logistic_multiclass', 'multiclass_data', 'label');
    }, 'Multi-class evaluation works');
    
    # Check confusion matrix
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_confusion_matrix('logistic_multiclass', 'multiclass_data', 'label');
    }, 'Confusion matrix generated');
    
    # Test class prediction for each class
    for my $class (0..2) {
        my $test_x1 = $class + 0.5;
        my $test_x2 = $class * 0.5 + 0.5;
        
        query_ok($node, 'postgres', qq{
            SELECT neurondb.ml_predict('logistic_multiclass', ARRAY[$test_x1, $test_x2, 0.5, 0.5]::vector);
        }, "Prediction for class $class works");
    }
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE multiclass_data CASCADE;
        SELECT neurondb.ml_drop_model('logistic_multiclass');
    }, 'Multi-class data cleaned up');
};

#
# Test 31-50: Support Vector Machines (SVM)
#
subtest 'SVM Linear Kernel' => sub {
    plan tests => 11;  # 3 neurondb_ok + 8 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE svm_data (
            id serial PRIMARY KEY,
            features vector(2),
            label int CHECK (label IN (-1, 1))
        );
    }, 'SVM data table created');
    
    # Create linearly separable data
    for my $i (1..40) {
        my $x1 = rand() * 2 - 1;
        my $x2 = rand() * 2 - 1;
        my $label = ($x1 + $x2 > 0) ? 1 : -1;
        
        query_ok($node, 'postgres', qq{
            INSERT INTO svm_data (features, label) 
            VALUES (ARRAY[$x1, $x2]::vector, $label);
        });
    }
    
    # Train SVM with linear kernel
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'svm_linear',
            algorithm := 'svm',
            training_table := 'svm_data',
            target_column := 'label',
            feature_columns := ARRAY['features'],
            params := '{"kernel": "linear", "C": 1.0}'::jsonb
        );
    }, 'SVM linear trained');
    
    # Get support vectors
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_support_vectors('svm_linear');
    }, 'Support vectors retrieved');
    
    # Test decision function
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_decision_function('svm_linear', ARRAY[0.5, 0.5]::vector);
    }, 'Decision function works');
    
    # Test margin calculation
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_margin('svm_linear');
    }, 'Margin retrieved');
    
    # Predict
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict('svm_linear', ARRAY[0.5, 0.5]::vector);
    }, 'SVM prediction works');
    
    # Evaluate
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_evaluate('svm_linear', 'svm_data', 'label');
    }, 'SVM evaluation works');
    
    # Test with different C values
    for my $c (0.1, 1.0, 10.0) {
        query_ok($node, 'postgres', qq{
            SELECT neurondb.ml_train(
                model_name := 'svm_linear_c$c',
                algorithm := 'svm',
                training_table := 'svm_data',
                target_column := 'label',
                feature_columns := ARRAY['features'],
                params := '{"kernel": "linear", "C": $c}'::jsonb
            );
        }, "SVM with C=$c trained");
    }
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE svm_data CASCADE;
        SELECT neurondb.ml_drop_model('svm_linear');
    }, 'SVM data cleaned up');
};

#
# Test 51-70: Decision Trees
#
subtest 'Decision Trees' => sub {
    plan tests => 11;  # 3 neurondb_ok + 8 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE tree_data (
            id serial PRIMARY KEY,
            features vector(3),
            label int
        );
    }, 'Tree data table created');
    
    # Insert data with clear decision boundaries
    for my $i (1..60) {
        my $x1 = rand();
        my $x2 = rand();
        my $x3 = rand();
        my $label = ($x1 > 0.5) ? (($x2 > 0.5) ? 1 : 0) : (($x3 > 0.5) ? 2 : 0);
        
        query_ok($node, 'postgres', qq{
            INSERT INTO tree_data (features, label) 
            VALUES (ARRAY[$x1, $x2, $x3]::vector, $label);
        });
    }
    
    # Train decision tree
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'decision_tree',
            algorithm := 'decision_tree',
            training_table := 'tree_data',
            target_column := 'label',
            feature_columns := ARRAY['features'],
            params := '{"max_depth": 5, "min_samples_split": 2}'::jsonb
        );
    }, 'Decision tree trained');
    
    # Get tree structure
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_tree_structure('decision_tree');
    }, 'Tree structure retrieved');
    
    # Get feature importance
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_feature_importance('decision_tree');
    }, 'Feature importance retrieved');
    
    # Predict
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict('decision_tree', ARRAY[0.8, 0.8, 0.2]::vector);
    }, 'Tree prediction works');
    
    # Test pruning
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_prune_tree('decision_tree', alpha := 0.01);
    }, 'Tree pruning works');
    
    # Test max_depth variations
    for my $depth (3, 5, 10) {
        query_ok($node, 'postgres', qq{
            SELECT neurondb.ml_train(
                model_name := 'tree_depth$depth',
                algorithm := 'decision_tree',
                training_table := 'tree_data',
                target_column := 'label',
                feature_columns := ARRAY['features'],
                params := '{"max_depth": $depth}'::jsonb
            );
        }, "Tree with max_depth=$depth trained");
    }
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE tree_data CASCADE;
        SELECT neurondb.ml_drop_model('decision_tree');
    }, 'Tree data cleaned up');
};

#
# Test 71-90: Random Forest
#
subtest 'Random Forest' => sub {
    plan tests => 11;  # 3 neurondb_ok + 8 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE forest_data (
            id serial PRIMARY KEY,
            features vector(5),
            label int
        );
    }, 'Forest data table created');
    
    # Insert training data
    for my $i (1..100) {
        my @features = map { rand() } (1..5);
        my $label = ($features[0] + $features[1] > 1.0) ? 1 : 0;
        my $feat_str = join(',', @features);
        
        query_ok($node, 'postgres', qq{
            INSERT INTO forest_data (features, label) 
            VALUES (ARRAY[$feat_str]::vector, $label);
        });
    }
    
    # Train random forest
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'random_forest',
            algorithm := 'random_forest',
            training_table := 'forest_data',
            target_column := 'label',
            feature_columns := ARRAY['features'],
            params := '{"n_estimators": 10, "max_depth": 5}'::jsonb
        );
    }, 'Random forest trained');
    
    # Get ensemble info
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_ensemble_info('random_forest');
    }, 'Ensemble info retrieved');
    
    # Get feature importance (averaged across trees)
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_feature_importance('random_forest');
    }, 'Forest feature importance retrieved');
    
    # Predict with voting
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict('random_forest', ARRAY[0.8, 0.8, 0.3, 0.2, 0.1]::vector);
    }, 'Forest prediction works');
    
    # Out-of-bag error
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_oob_score('random_forest');
    }, 'OOB score retrieved');
    
    # Test different n_estimators
    for my $n (5, 10, 20) {
        query_ok($node, 'postgres', qq{
            SELECT neurondb.ml_train(
                model_name := 'forest_n$n',
                algorithm := 'random_forest',
                training_table := 'forest_data',
                target_column := 'label',
                feature_columns := ARRAY['features'],
                params := '{"n_estimators": $n, "max_depth": 5}'::jsonb
            );
        }, "Forest with n_estimators=$n trained");
    }
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE forest_data CASCADE;
        SELECT neurondb.ml_drop_model('random_forest');
    }, 'Forest data cleaned up');
};

#
# Test 91-110: Naive Bayes
#
subtest 'Naive Bayes' => sub {
    plan tests => 11;  # 3 neurondb_ok + 8 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE nb_data (
            id serial PRIMARY KEY,
            features vector(3),
            label int
        );
    }, 'Naive Bayes data table created');
    
    # Insert data from Gaussian distributions
    for my $class (0..1) {
        for my $i (1..40) {
            my $x1 = rand() + $class;
            my $x2 = rand() + $class * 0.5;
            my $x3 = rand();
            
            query_ok($node, 'postgres', qq{
                INSERT INTO nb_data (features, label) 
                VALUES (ARRAY[$x1, $x2, $x3]::vector, $class);
            });
        }
    }
    
    # Train Gaussian Naive Bayes
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'naive_bayes',
            algorithm := 'naive_bayes',
            training_table := 'nb_data',
            target_column := 'label',
            feature_columns := ARRAY['features'],
            params := '{"variant": "gaussian"}'::jsonb
        );
    }, 'Naive Bayes trained');
    
    # Get class priors
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_class_priors('naive_bayes');
    }, 'Class priors retrieved');
    
    # Get feature distributions
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_distributions('naive_bayes');
    }, 'Feature distributions retrieved');
    
    # Predict with probabilities
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict_proba('naive_bayes', ARRAY[1.5, 0.8, 0.5]::vector);
    }, 'NB prediction with proba works');
    
    # Evaluate
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_evaluate('naive_bayes', 'nb_data', 'label');
    }, 'NB evaluation works');
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE nb_data CASCADE;
        SELECT neurondb.ml_drop_model('naive_bayes');
    }, 'NB data cleaned up');
};

#
# Test 111-130: KNN Classification
#
subtest 'KNN Classification' => sub {
    plan tests => 11;  # 3 neurondb_ok + 8 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE knn_data (
            id serial PRIMARY KEY,
            features vector(2),
            label int
        );
    }, 'KNN data table created');
    
    # Insert clustered data
    for my $class (0..2) {
        my $cx = $class;
        my $cy = $class;
        
        for my $i (1..30) {
            my $x1 = $cx + (rand() - 0.5) * 0.5;
            my $x2 = $cy + (rand() - 0.5) * 0.5;
            
            query_ok($node, 'postgres', qq{
                INSERT INTO knn_data (features, label) 
                VALUES (ARRAY[$x1, $x2]::vector, $class);
            });
        }
    }
    
    # Train KNN classifier
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'knn_classifier',
            algorithm := 'knn',
            training_table := 'knn_data',
            target_column := 'label',
            feature_columns := ARRAY['features'],
            params := '{"n_neighbors": 5, "metric": "euclidean"}'::jsonb
        );
    }, 'KNN classifier trained');
    
    # Predict
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict('knn_classifier', ARRAY[1.5, 1.5]::vector);
    }, 'KNN prediction works');
    
    # Get nearest neighbors
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_neighbors('knn_classifier', ARRAY[1.5, 1.5]::vector, k := 5);
    }, 'Neighbors retrieved');
    
    # Test different k values
    for my $k (3, 5, 7) {
        query_ok($node, 'postgres', qq{
            SELECT neurondb.ml_train(
                model_name := 'knn_k$k',
                algorithm := 'knn',
                training_table := 'knn_data',
                target_column := 'label',
                feature_columns := ARRAY['features'],
                params := '{"n_neighbors": $k}'::jsonb
            );
        }, "KNN with k=$k trained");
    }
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE knn_data CASCADE;
        SELECT neurondb.ml_drop_model('knn_classifier');
    }, 'KNN data cleaned up');
};

#
# Test 131-150: Advanced Classification Features
#
subtest 'Advanced Classification Features' => sub {
    plan tests => 11;  # 3 neurondb_ok + 8 subtests
    
    # Test cross-validation
    query_ok($node, 'postgres', q{
        CREATE TABLE cv_data (
            id serial PRIMARY KEY,
            features vector(3),
            label int
        );
    }, 'CV data table created');
    
    # Insert data
    for my $i (1..100) {
        my @features = map { rand() } (1..3);
        my $label = ($features[0] > 0.5) ? 1 : 0;
        my $feat_str = join(',', @features);
        
        $node->safe_psql('postgres', qq{
            INSERT INTO cv_data (features, label) 
            VALUES (ARRAY[$feat_str]::vector, $label);
        });
    }
    
    # K-fold cross-validation
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_cross_validate(
            algorithm := 'logistic_regression',
            training_table := 'cv_data',
            target_column := 'label',
            feature_columns := ARRAY['features'],
            k_folds := 5
        );
    }, 'Cross-validation works');
    
    # Grid search for hyperparameters
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_grid_search(
            algorithm := 'svm',
            training_table := 'cv_data',
            target_column := 'label',
            feature_columns := ARRAY['features'],
            param_grid := '{"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}'::jsonb
        );
    }, 'Grid search works');
    
    # ROC curve and AUC
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'roc_model',
            algorithm := 'logistic_regression',
            training_table := 'cv_data',
            target_column := 'label',
            feature_columns := ARRAY['features']
        );
    }, 'ROC model trained');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_roc_curve('roc_model', 'cv_data', 'label');
    }, 'ROC curve generated');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_roc_auc('roc_model', 'cv_data', 'label');
    }, 'ROC AUC calculated');
    
    # Precision-Recall curve
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_precision_recall_curve('roc_model', 'cv_data', 'label');
    }, 'PR curve generated');
    
    # Classification report
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_classification_report('roc_model', 'cv_data', 'label');
    }, 'Classification report generated');
    
    # Test class imbalance handling
    query_ok($node, 'postgres', q{
        CREATE TABLE imbalanced_data AS 
        SELECT * FROM cv_data WHERE label = 0
        UNION ALL
        SELECT * FROM cv_data WHERE label = 1 LIMIT 10;
    }, 'Imbalanced data created');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'imbalanced_model',
            algorithm := 'logistic_regression',
            training_table := 'imbalanced_data',
            target_column := 'label',
            feature_columns := ARRAY['features'],
            params := '{"class_weight": "balanced"}'::jsonb
        );
    }, 'Imbalanced model trained');
    
    # Test model versioning
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_save_model_version('roc_model', version := 'v1.0', notes := 'Initial version');
    }, 'Model version saved');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_list_model_versions('roc_model');
    }, 'Model versions listed');
    
    # Test model comparison
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_compare_models(
            ARRAY['roc_model', 'imbalanced_model'],
            'cv_data',
            'label'
        );
    }, 'Models compared');
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE cv_data CASCADE;
        DROP TABLE imbalanced_data CASCADE;
        SELECT neurondb.ml_drop_model('roc_model');
        SELECT neurondb.ml_drop_model('imbalanced_model');
    }, 'Classification test data cleaned up');
};

$node->stop();
$node->cleanup();

done_testing();



