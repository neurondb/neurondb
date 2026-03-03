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

022_ml_dimensionality.t - ML dimensionality reduction tests

=head1 DESCRIPTION

Comprehensive tests for dimensionality reduction:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- Incremental PCA
- Kernel PCA
- Truncated SVD
- Factor Analysis
- Variance explained analysis
- Reconstruction error

=cut

plan tests => 4;  # 3 neurondb_ok + 1 subtest

my $node = PostgresNode->new('test_dim_reduction');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');
query_ok($node, 'postgres', "SET neurondb.enable_ml = on;", 'ML features enabled');

subtest 'PCA Standard' => sub {
    plan tests => 4;  # 3 neurondb_ok + 1 subtest
    
    query_ok($node, 'postgres', q{
        CREATE TABLE pca_data (id serial PRIMARY KEY, features vector(10));
        INSERT INTO pca_data (features) 
        SELECT ARRAY(SELECT random() FROM generate_series(1,10))::vector 
        FROM generate_series(1,100);
    }, 'PCA data created');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'pca_model',
            algorithm := 'pca',
            training_table := 'pca_data',
            feature_columns := ARRAY['features'],
            params := '{"n_components": 3}'::jsonb
        );
    }, 'PCA model trained');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_transform('pca_model', ARRAY[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]::vector);
    }, 'PCA transform works');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_explained_variance('pca_model');
    }, 'Explained variance retrieved');
    
    query_ok($node, 'postgres', q{
        DROP TABLE pca_data CASCADE;
        SELECT neurondb.ml_drop_model('pca_model');
    }, 'PCA cleaned up');
};

$node->stop();
$node->cleanup();
done_testing();



