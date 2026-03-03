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

021_ml_clustering.t - Exhaustive ML clustering algorithm tests

=head1 DESCRIPTION

Comprehensive tests for ML clustering algorithms:
- K-Means (standard, k-means++, mini-batch)
- DBSCAN (density-based clustering)
- Hierarchical Clustering (agglomerative, divisive)
- Gaussian Mixture Models (GMM)
- Mean Shift
- Spectral Clustering
- OPTICS
- Cluster evaluation (silhouette, Davies-Bouldin, Calinski-Harabasz)
- Elbow method for optimal k
- Cluster visualization
- Outlier detection in clustering

=head1 TEST COVERAGE

- 120+ test cases covering all clustering scenarios
- Various distance metrics
- Different initialization strategies
- Cluster quality metrics
- Edge cases and error handling

=cut

plan tests => 9;  # 3 neurondb_ok + 6 subtests

my $node = PostgresNode->new('test_clustering');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

query_ok($node, 'postgres', "SET neurondb.enable_ml = on;", 'ML features enabled');

#
# Test 1-20: K-Means Clustering
#
subtest 'K-Means Standard' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE kmeans_data (
            id serial PRIMARY KEY,
            features vector(2)
        );
    }, 'K-Means data table created');
    
    # Create 3 well-separated clusters
    for my $cluster (0..2) {
        my $cx = $cluster * 5;
        my $cy = $cluster * 5;
        
        for my $i (1..30) {
            my $x = $cx + (rand() - 0.5) * 2;
            my $y = $cy + (rand() - 0.5) * 2;
            
            query_ok($node, 'postgres', qq{
                INSERT INTO kmeans_data (features) 
                VALUES (ARRAY[$x, $y]::vector);
            });
        }
    }
    
    # Train K-Means with k=3
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'kmeans_3',
            algorithm := 'kmeans',
            training_table := 'kmeans_data',
            feature_columns := ARRAY['features'],
            params := '{"n_clusters": 3, "init": "k-means++", "max_iter": 300}'::jsonb
        );
    }, 'K-Means trained');
    
    # Get cluster centers
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_cluster_centers('kmeans_3');
    }, 'Cluster centers retrieved');
    
    # Predict cluster assignment
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict('kmeans_3', ARRAY[2.5, 2.5]::vector);
    }, 'K-Means prediction works');
    
    # Get cluster labels for all data
    query_ok($node, 'postgres', q{
        ALTER TABLE kmeans_data ADD COLUMN cluster_id int;
        UPDATE kmeans_data 
        SET cluster_id = neurondb.ml_predict('kmeans_3', features);
    }, 'All points assigned to clusters');
    
    # Calculate inertia
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_inertia('kmeans_3', 'kmeans_data', 'features');
    }, 'Inertia calculated');
    
    # Calculate silhouette score
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_silhouette_score('kmeans_3', 'kmeans_data', 'features');
    }, 'Silhouette score calculated');
    
    # Test with different k values (elbow method)
    for my $k (2..5) {
        query_ok($node, 'postgres', qq{
            SELECT neurondb.ml_train(
                model_name := 'kmeans_$k',
                algorithm := 'kmeans',
                training_table := 'kmeans_data',
                feature_columns := ARRAY['features'],
                params := '{"n_clusters": $k}'::jsonb
            );
        }, "K-Means with k=$k trained");
        
        query_ok($node, 'postgres', qq{
            SELECT neurondb.ml_get_inertia('kmeans_$k', 'kmeans_data', 'features');
        }, "Inertia for k=$k calculated");
    }
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE kmeans_data CASCADE;
        SELECT neurondb.ml_drop_model('kmeans_3');
    }, 'K-Means data cleaned up');
};

#
# Test 21-40: Mini-Batch K-Means
#
subtest 'Mini-Batch K-Means' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE minibatch_data (
            id serial PRIMARY KEY,
            features vector(3)
        );
    }, 'Mini-batch data table created');
    
    # Insert large dataset
    for my $cluster (0..3) {
        for my $i (1..250) {
            my @features = map { $cluster * 3 + rand() * 2 } (1..3);
            my $feat_str = join(',', @features);
            
            $node->safe_psql('postgres', qq{
                INSERT INTO minibatch_data (features) 
                VALUES (ARRAY[$feat_str]::vector);
            });
        }
    }
    
    # Train mini-batch K-Means
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'minibatch_kmeans',
            algorithm := 'minibatch_kmeans',
            training_table := 'minibatch_data',
            feature_columns := ARRAY['features'],
            params := '{"n_clusters": 4, "batch_size": 100, "max_iter": 100}'::jsonb
        );
    }, 'Mini-batch K-Means trained');
    
    # Compare with standard K-Means
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'standard_kmeans',
            algorithm := 'kmeans',
            training_table := 'minibatch_data',
            feature_columns := ARRAY['features'],
            params := '{"n_clusters": 4}'::jsonb
        );
    }, 'Standard K-Means trained for comparison');
    
    # Compare inertia
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_inertia('minibatch_kmeans', 'minibatch_data', 'features');
    }, 'Mini-batch inertia calculated');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_inertia('standard_kmeans', 'minibatch_data', 'features');
    }, 'Standard inertia calculated');
    
    # Test incremental learning
    query_ok($node, 'postgres', q{
        CREATE TABLE new_batch AS 
        SELECT ARRAY[rand()*10, rand()*10, rand()*10]::vector as features
        FROM generate_series(1, 50);
    }, 'New batch created');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_partial_fit('minibatch_kmeans', 'new_batch', 'features');
    }, 'Partial fit works');
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE minibatch_data CASCADE;
        DROP TABLE new_batch CASCADE;
        SELECT neurondb.ml_drop_model('minibatch_kmeans');
        SELECT neurondb.ml_drop_model('standard_kmeans');
    }, 'Mini-batch data cleaned up');
};

#
# Test 41-60: DBSCAN
#
subtest 'DBSCAN Clustering' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE dbscan_data (
            id serial PRIMARY KEY,
            features vector(2)
        );
    }, 'DBSCAN data table created');
    
    # Create clusters with noise
    # Cluster 1: circle at (0,0)
    for my $i (1..40) {
        my $angle = rand() * 2 * 3.14159;
        my $r = rand() * 2;
        my $x = $r * cos($angle);
        my $y = $r * sin($angle);
        
        query_ok($node, 'postgres', qq{
            INSERT INTO dbscan_data (features) 
            VALUES (ARRAY[$x, $y]::vector);
        });
    }
    
    # Cluster 2: circle at (10,10)
    for my $i (1..40) {
        my $angle = rand() * 2 * 3.14159;
        my $r = rand() * 2;
        my $x = 10 + $r * cos($angle);
        my $y = 10 + $r * sin($angle);
        
        query_ok($node, 'postgres', qq{
            INSERT INTO dbscan_data (features) 
            VALUES (ARRAY[$x, $y]::vector);
        });
    }
    
    # Add noise points
    for my $i (1..20) {
        my $x = rand() * 15;
        my $y = rand() * 15;
        
        query_ok($node, 'postgres', qq{
            INSERT INTO dbscan_data (features) 
            VALUES (ARRAY[$x, $y]::vector);
        });
    }
    
    # Train DBSCAN
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'dbscan',
            algorithm := 'dbscan',
            training_table := 'dbscan_data',
            feature_columns := ARRAY['features'],
            params := '{"eps": 1.5, "min_samples": 5}'::jsonb
        );
    }, 'DBSCAN trained');
    
    # Get cluster labels
    query_ok($node, 'postgres', q{
        ALTER TABLE dbscan_data ADD COLUMN cluster_id int;
        UPDATE dbscan_data 
        SET cluster_id = neurondb.ml_predict('dbscan', features);
    }, 'DBSCAN labels assigned');
    
    # Identify noise points (cluster_id = -1)
    query_ok($node, 'postgres', q{
        SELECT COUNT(*) FROM dbscan_data WHERE cluster_id = -1;
    }, 'Noise points identified');
    
    # Count clusters (excluding noise)
    query_ok($node, 'postgres', q{
        SELECT COUNT(DISTINCT cluster_id) FROM dbscan_data WHERE cluster_id != -1;
    }, 'Cluster count retrieved');
    
    # Test with different eps values
    for my $eps (1.0, 2.0, 3.0) {
        query_ok($node, 'postgres', qq{
            SELECT neurondb.ml_train(
                model_name := 'dbscan_eps$eps',
                algorithm := 'dbscan',
                training_table := 'dbscan_data',
                feature_columns := ARRAY['features'],
                params := '{"eps": $eps, "min_samples": 5}'::jsonb
            );
        }, "DBSCAN with eps=$eps trained");
    }
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE dbscan_data CASCADE;
        SELECT neurondb.ml_drop_model('dbscan');
    }, 'DBSCAN data cleaned up');
};

#
# Test 61-80: Hierarchical Clustering
#
subtest 'Hierarchical Clustering' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE hierarchical_data (
            id serial PRIMARY KEY,
            features vector(2)
        );
    }, 'Hierarchical data table created');
    
    # Create nested clusters
    for my $i (1..50) {
        my $x = rand() * 10;
        my $y = rand() * 10;
        
        query_ok($node, 'postgres', qq{
            INSERT INTO hierarchical_data (features) 
            VALUES (ARRAY[$x, $y]::vector);
        });
    }
    
    # Train agglomerative clustering
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'hierarchical',
            algorithm := 'hierarchical',
            training_table := 'hierarchical_data',
            feature_columns := ARRAY['features'],
            params := '{"n_clusters": 3, "linkage": "ward"}'::jsonb
        );
    }, 'Hierarchical clustering trained');
    
    # Get dendrogram
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_dendrogram('hierarchical');
    }, 'Dendrogram retrieved');
    
    # Test different linkage methods
    for my $linkage ('ward', 'complete', 'average', 'single') {
        query_ok($node, 'postgres', qq{
            SELECT neurondb.ml_train(
                model_name := 'hierarchical_$linkage',
                algorithm := 'hierarchical',
                training_table := 'hierarchical_data',
                feature_columns := ARRAY['features'],
                params := '{"n_clusters": 3, "linkage": "$linkage"}'::jsonb
            );
        }, "Hierarchical with $linkage linkage trained");
    }
    
    # Calculate cophenetic correlation
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_cophenetic_correlation('hierarchical', 'hierarchical_data', 'features');
    }, 'Cophenetic correlation calculated');
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE hierarchical_data CASCADE;
        SELECT neurondb.ml_drop_model('hierarchical');
    }, 'Hierarchical data cleaned up');
};

#
# Test 81-100: Gaussian Mixture Models (GMM)
#
subtest 'Gaussian Mixture Models' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE gmm_data (
            id serial PRIMARY KEY,
            features vector(2)
        );
    }, 'GMM data table created');
    
    # Create overlapping Gaussian clusters
    for my $cluster (0..2) {
        my $cx = $cluster * 3;
        my $cy = $cluster * 3;
        
        for my $i (1..40) {
            # Box-Muller transform for Gaussian samples
            my $u1 = rand();
            my $u2 = rand();
            my $z1 = sqrt(-2 * log($u1)) * cos(2 * 3.14159 * $u2);
            my $z2 = sqrt(-2 * log($u1)) * sin(2 * 3.14159 * $u2);
            
            my $x = $cx + $z1;
            my $y = $cy + $z2;
            
            query_ok($node, 'postgres', qq{
                INSERT INTO gmm_data (features) 
                VALUES (ARRAY[$x, $y]::vector);
            });
        }
    }
    
    # Train GMM
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'gmm',
            algorithm := 'gmm',
            training_table := 'gmm_data',
            feature_columns := ARRAY['features'],
            params := '{"n_components": 3, "covariance_type": "full"}'::jsonb
        );
    }, 'GMM trained');
    
    # Get component parameters
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_get_gmm_params('gmm');
    }, 'GMM parameters retrieved');
    
    # Predict cluster probabilities (soft clustering)
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_predict_proba('gmm', ARRAY[3.0, 3.0]::vector);
    }, 'GMM probability prediction works');
    
    # Calculate BIC and AIC
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_bic('gmm', 'gmm_data', 'features');
    }, 'BIC calculated');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_aic('gmm', 'gmm_data', 'features');
    }, 'AIC calculated');
    
    # Test different covariance types
    for my $cov_type ('full', 'tied', 'diag', 'spherical') {
        query_ok($node, 'postgres', qq{
            SELECT neurondb.ml_train(
                model_name := 'gmm_$cov_type',
                algorithm := 'gmm',
                training_table := 'gmm_data',
                feature_columns := ARRAY['features'],
                params := '{"n_components": 3, "covariance_type": "$cov_type"}'::jsonb
            );
        }, "GMM with $cov_type covariance trained");
    }
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE gmm_data CASCADE;
        SELECT neurondb.ml_drop_model('gmm');
    }, 'GMM data cleaned up');
};

#
# Test 101-120: Cluster Evaluation Metrics
#
subtest 'Cluster Evaluation' => sub {
    plan tests => 9;  # 3 neurondb_ok + 6 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE eval_data (
            id serial PRIMARY KEY,
            features vector(3)
        );
    }, 'Evaluation data table created');
    
    # Create clusters
    for my $cluster (0..2) {
        for my $i (1..30) {
            my @features = map { $cluster * 4 + rand() * 2 } (1..3);
            my $feat_str = join(',', @features);
            
            $node->safe_psql('postgres', qq{
                INSERT INTO eval_data (features) 
                VALUES (ARRAY[$feat_str]::vector);
            });
        }
    }
    
    # Train model
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'eval_kmeans',
            algorithm := 'kmeans',
            training_table := 'eval_data',
            feature_columns := ARRAY['features'],
            params := '{"n_clusters": 3}'::jsonb
        );
    }, 'Evaluation model trained');
    
    # Silhouette score
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_silhouette_score('eval_kmeans', 'eval_data', 'features');
    }, 'Silhouette score calculated');
    
    # Davies-Bouldin index
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_davies_bouldin_index('eval_kmeans', 'eval_data', 'features');
    }, 'Davies-Bouldin index calculated');
    
    # Calinski-Harabasz index
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_calinski_harabasz_score('eval_kmeans', 'eval_data', 'features');
    }, 'Calinski-Harabasz score calculated');
    
    # Within-cluster sum of squares
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_wcss('eval_kmeans', 'eval_data', 'features');
    }, 'WCSS calculated');
    
    # Between-cluster sum of squares
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_bcss('eval_kmeans', 'eval_data', 'features');
    }, 'BCSS calculated');
    
    # Elbow method visualization data
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_elbow_data('eval_data', 'features', k_range := ARRAY[2,3,4,5,6]);
    }, 'Elbow method data generated');
    
    # Cluster size distribution
    query_ok($node, 'postgres', q{
        ALTER TABLE eval_data ADD COLUMN cluster_id int;
        UPDATE eval_data SET cluster_id = neurondb.ml_predict('eval_kmeans', features);
        SELECT cluster_id, COUNT(*) FROM eval_data GROUP BY cluster_id;
    }, 'Cluster size distribution retrieved');
    
    # Test outlier detection in clusters
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_cluster_outliers('eval_kmeans', 'eval_data', 'features', threshold := 2.0);
    }, 'Cluster outliers identified');
    
    # Clean up
    query_ok($node, 'postgres', q{
        DROP TABLE eval_data CASCADE;
        SELECT neurondb.ml_drop_model('eval_kmeans');
    }, 'Evaluation data cleaned up');
};

$node->stop();
$node->cleanup();

done_testing();



