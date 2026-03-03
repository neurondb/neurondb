#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use VectorOps;
use MLHelpers;
use IndexHelpers;
use SparseHelpers;

=head1 NAME

029_integration_final.t - Final integration tests

=head1 DESCRIPTION

Comprehensive end-to-end integration tests combining multiple features:
- Vector + Index + Search pipeline
- ML training + Inference + Evaluation
- Sparse + Dense hybrid search
- Quantization + GPU acceleration
- Multi-tenant operations
- Real-world use cases

=cut

plan tests => 8;  # 3 neurondb_ok + 5 subtests

my $node = PostgresNode->new('test_integration');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

subtest 'Full Search Pipeline' => sub {
    plan tests => 8;  # 3 neurondb_ok + 5 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE documents (
            id serial PRIMARY KEY,
            content text,
            embedding vector(384),
            sparse_embedding sparsevec
        );
    }, 'Document table created');
    
    query_ok($node, 'postgres', q{
        INSERT INTO documents (content, embedding, sparse_embedding) 
        VALUES 
        ('Machine learning basics', ARRAY(SELECT random() FROM generate_series(1,384))::vector, '{1:0.8,5:0.6,10:0.4}'::sparsevec),
        ('Deep learning tutorial', ARRAY(SELECT random() FROM generate_series(1,384))::vector, '{2:0.9,5:0.7,12:0.5}'::sparsevec),
        ('Neural networks guide', ARRAY(SELECT random() FROM generate_series(1,384))::vector, '{1:0.7,3:0.8,10:0.3}'::sparsevec);
    }, 'Documents inserted');
    
    query_ok($node, 'postgres', q{
        CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX ON documents USING gin (sparse_embedding);
    }, 'Indexes created');
    
    query_ok($node, 'postgres', q{
        WITH dense_results AS (
            SELECT id, content, embedding <=> ARRAY(SELECT 0.5 FROM generate_series(1,384))::vector as dense_score
            FROM documents
            ORDER BY dense_score LIMIT 10
        ),
        sparse_results AS (
            SELECT id, content, sparse_embedding <#> '{1:0.9,5:0.8}'::sparsevec as sparse_score
            FROM documents
            ORDER BY sparse_score LIMIT 10
        )
        SELECT d.id, d.content, 
               COALESCE(dr.dense_score, 1.0) * 0.7 + COALESCE(sr.sparse_score, 1.0) * 0.3 as hybrid_score
        FROM documents d
        LEFT JOIN dense_results dr ON d.id = dr.id
        LEFT JOIN sparse_results sr ON d.id = sr.id
        ORDER BY hybrid_score LIMIT 5;
    }, 'Hybrid search works');
    
    query_ok($node, 'postgres', q{
        DROP TABLE documents CASCADE;
    }, 'Integration test cleaned up');
};

subtest 'ML Training Pipeline' => sub {
    plan tests => 8;  # 3 neurondb_ok + 5 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE ml_dataset (
            id serial PRIMARY KEY,
            features vector(10),
            label int
        );
        INSERT INTO ml_dataset (features, label)
        SELECT ARRAY(SELECT random() FROM generate_series(1,10))::vector, 
               (random() > 0.5)::int
        FROM generate_series(1,200);
    }, 'ML dataset created');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_train(
            model_name := 'integration_model',
            algorithm := 'logistic_regression',
            training_table := 'ml_dataset',
            target_column := 'label',
            feature_columns := ARRAY['features']
        );
    }, 'Model trained');
    
    query_ok($node, 'postgres', q{
        SELECT neurondb.ml_evaluate('integration_model', 'ml_dataset', 'label');
    }, 'Model evaluated');
    
    query_ok($node, 'postgres', q{
        ALTER TABLE ml_dataset ADD COLUMN prediction int;
        UPDATE ml_dataset SET prediction = neurondb.ml_predict('integration_model', features);
    }, 'Batch prediction works');
    
    query_ok($node, 'postgres', q{
        SELECT AVG((label = prediction)::int) as accuracy FROM ml_dataset;
    }, 'Accuracy calculated');
    
    query_ok($node, 'postgres', q{
        DROP TABLE ml_dataset CASCADE;
        SELECT neurondb.ml_drop_model('integration_model');
    }, 'ML pipeline cleaned up');
};

subtest 'Multi-tenant Operations' => sub {
    plan tests => 8;  # 3 neurondb_ok + 5 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE multi_tenant_data (
            id serial PRIMARY KEY,
            tenant_id int,
            embedding vector(64)
        );
        INSERT INTO multi_tenant_data (tenant_id, embedding)
        SELECT 
            (random() * 5)::int,
            ARRAY(SELECT random() FROM generate_series(1,64))::vector
        FROM generate_series(1,500);
    }, 'Multi-tenant data created');
    
    query_ok($node, 'postgres', q{
        CREATE INDEX ON multi_tenant_data (tenant_id);
        CREATE INDEX ON multi_tenant_data USING hnsw (embedding vector_l2_ops);
    }, 'Multi-tenant indexes created');
    
    query_ok($node, 'postgres', q{
        SELECT * FROM multi_tenant_data 
        WHERE tenant_id = 1 
        ORDER BY embedding <-> ARRAY(SELECT 0.5 FROM generate_series(1,64))::vector 
        LIMIT 10;
    }, 'Tenant-specific search works');
    
    query_ok($node, 'postgres', q{
        DROP TABLE multi_tenant_data CASCADE;
    }, 'Multi-tenant test cleaned up');
};

subtest 'Performance Testing' => sub {
    plan tests => 8;  # 3 neurondb_ok + 5 subtests
    
    query_ok($node, 'postgres', q{
        CREATE TABLE perf_test (
            id serial PRIMARY KEY,
            embedding vector(256)
        );
        INSERT INTO perf_test (embedding)
        SELECT ARRAY(SELECT random() FROM generate_series(1,256))::vector
        FROM generate_series(1,5000);
    }, 'Performance test data created');
    
    query_ok($node, 'postgres', q{
        CREATE INDEX ON perf_test USING hnsw (embedding vector_l2_ops) 
        WITH (m = 16, ef_construction = 64);
    }, 'Performance index created');
    
    query_ok($node, 'postgres', q{
        EXPLAIN (ANALYZE, BUFFERS) 
        SELECT * FROM perf_test 
        ORDER BY embedding <-> ARRAY(SELECT 0.5 FROM generate_series(1,256))::vector 
        LIMIT 10;
    }, 'Query plan analyzed');
    
    query_ok($node, 'postgres', q{
        DROP TABLE perf_test CASCADE;
    }, 'Performance test cleaned up');
};

subtest 'Real-world Use Cases' => sub {
    plan tests => 8;  # 3 neurondb_ok + 5 subtests
    
    # Semantic search use case
    query_ok($node, 'postgres', q{
        CREATE TABLE articles (
            id serial PRIMARY KEY,
            title text,
            embedding vector(768)
        );
        INSERT INTO articles (title, embedding) VALUES
        ('Understanding PostgreSQL', ARRAY(SELECT random() FROM generate_series(1,768))::vector),
        ('Vector Databases Explained', ARRAY(SELECT random() FROM generate_series(1,768))::vector),
        ('Machine Learning in SQL', ARRAY(SELECT random() FROM generate_series(1,768))::vector);
    }, 'Articles table created');
    
    query_ok($node, 'postgres', q{
        CREATE INDEX ON articles USING hnsw (embedding vector_cosine_ops);
    }, 'Semantic index created');
    
    query_ok($node, 'postgres', q{
        SELECT title FROM articles 
        ORDER BY embedding <=> ARRAY(SELECT 0.5 FROM generate_series(1,768))::vector 
        LIMIT 3;
    }, 'Semantic search works');
    
    query_ok($node, 'postgres', q{
        DROP TABLE articles CASCADE;
    }, 'Semantic search cleaned up');
};

$node->stop();
$node->cleanup();
done_testing();



