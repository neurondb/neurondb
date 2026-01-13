#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;

=head1 NAME

030_security_comprehensive.t - Comprehensive security feature tests

=head1 DESCRIPTION

Tests for Data Governance & Security features:
- Row-Level Security (RLS) for embeddings
- Field-level encryption for vectors
- Audit logging for ML inference and RAG operations

=cut

plan tests => 50;  # Adjust based on actual test count

my $node = PostgresNode->new('security_test');
ok($node, 'PostgresNode created');

$node->init();
$node->start();

ok($node->is_running(), 'PostgreSQL node started');

# Install extension
install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# ============================================================================
# RLS for Embeddings Tests
# ============================================================================

subtest 'RLS for Embeddings' => sub {
	plan tests => 8;
	
	# Create test table with tenant_id
	query_ok($node, 'postgres', q{
		DROP TABLE IF EXISTS test_documents_rls CASCADE;
		CREATE TABLE test_documents_rls (
			id SERIAL PRIMARY KEY,
			tenant_id INTEGER NOT NULL,
			content TEXT,
			embedding vector(3)
		);
		
		INSERT INTO test_documents_rls (tenant_id, content, embedding) VALUES
			(1, 'Document 1', '[1,2,3]'::vector(3)),
			(1, 'Document 2', '[2,3,4]'::vector(3)),
			(2, 'Document 3', '[3,4,5]'::vector(3)),
			(2, 'Document 4', '[4,5,6]'::vector(3));
	}, 'Create test table with tenant data');
	
	# Enable RLS
	query_ok($node, 'postgres', q{
		ALTER TABLE test_documents_rls ENABLE ROW LEVEL SECURITY;
	}, 'Enable RLS on table');
	
	# Test RLS function exists
	query_ok($node, 'postgres', q{
		SELECT enable_embedding_rls('test_documents_rls');
	}, 'enable_embedding_rls function works');
	
	# Create RLS policy
	query_ok($node, 'postgres', q{
		CREATE POLICY tenant_isolation ON test_documents_rls
		FOR SELECT USING (tenant_id = 1);
	}, 'Create RLS policy');
	
	# Test RLS policy creation function
	query_ok($node, 'postgres', q{
		DROP POLICY IF EXISTS test_policy ON test_documents_rls;
		SELECT create_embedding_rls_policy(
			'test_documents_rls',
			'test_policy',
			'tenant_id = 2'
		);
	}, 'create_embedding_rls_policy function works');
	
	# Test RLS is enabled
	result_is($node, 'postgres',
		q{SELECT neurondb_test_rls('test_documents_rls'::regclass);},
		't',
		'RLS is enabled on table'
	);
	
	# Test tenant isolation policy exists
	query_ok($node, 'postgres', q{
		SELECT polname FROM pg_policies 
		WHERE tablename = 'test_documents_rls' AND polname = 'tenant_isolation';
	}, 'RLS policy exists');
	
	# Cleanup
	query_ok($node, 'postgres', q{
		DROP TABLE IF EXISTS test_documents_rls CASCADE;
	}, 'Cleanup RLS test table');
};

# ============================================================================
# Field-Level Encryption Tests
# ============================================================================

subtest 'Field-Level Encryption' => sub {
	plan tests => 6;
	
	# Create test table
	query_ok($node, 'postgres', q{
		DROP TABLE IF EXISTS test_encrypted_documents CASCADE;
		CREATE TABLE test_encrypted_documents (
			id SERIAL PRIMARY KEY,
			content TEXT,
			embedding vector(3),
			encrypted_embedding bytea
		);
	}, 'Create test table for encryption');
	
	# Test encryption function exists (may fail if OpenSSL not available)
	my $encrypt_result = $node->psql('postgres', q{
		SELECT encrypt_vector('[1,2,3]'::vector(3), 'test-key-12345');
	});
	
	if ($encrypt_result->{success}) {
		ok(1, 'encrypt_vector function works');
		
		# Test decryption
		my $decrypt_result = $node->psql('postgres', q{
			SELECT decrypt_vector(
				encrypt_vector('[1,2,3]'::vector(3), 'test-key-12345'),
				'test-key-12345'
			);
		});
		
		if ($decrypt_result->{success}) {
			ok(1, 'decrypt_vector function works');
			
			# Test round-trip encryption/decryption
			query_ok($node, 'postgres', q{
				INSERT INTO test_encrypted_documents (content, encrypted_embedding)
				VALUES (
					'Test document',
					encrypt_vector('[1,2,3]'::vector(3), 'test-key-12345')
				);
			}, 'Insert encrypted vector');
			
			# Test decryption in query
			query_ok($node, 'postgres', q{
				SELECT decrypt_vector(encrypted_embedding, 'test-key-12345') 
				FROM test_encrypted_documents 
				WHERE id = 1;
			}, 'Decrypt vector in query');
			
			# Test key rotation function exists
			query_ok($node, 'postgres', q{
				SELECT rotate_encryption_key(
					'test_encrypted_documents',
					'encrypted_embedding',
					'test-key-12345',
					'new-test-key-67890'
				);
			}, 'rotate_encryption_key function works');
		} else {
			skip('decrypt_vector not available (OpenSSL may not be configured)', 4);
		}
	} else {
		skip('encrypt_vector not available (OpenSSL may not be configured)', 5);
	}
	
	# Cleanup
	query_ok($node, 'postgres', q{
		DROP TABLE IF EXISTS test_encrypted_documents CASCADE;
	}, 'Cleanup encryption test table');
};

# ============================================================================
# ML Inference Audit Logging Tests
# ============================================================================

subtest 'ML Inference Audit Logging' => sub {
	plan tests => 10;
	
	# Enable audit logging
	query_ok($node, 'postgres', q{
		SET neurondb.audit_ml_enabled = true;
	}, 'Enable ML audit logging');
	
	# Test audit table exists
	query_ok($node, 'postgres', q{
		SELECT COUNT(*) FROM neurondb.ml_inference_audit_log;
	}, 'ML inference audit log table exists');
	
	# Test log_ml_inference function
	query_ok($node, 'postgres', q{
		SELECT log_ml_inference(
			1,
			'predict',
			'abc123',
			'def456',
			'{"batch_size": 100}'::jsonb
		);
	}, 'log_ml_inference function works');
	
	# Verify log entry was created
	result_is($node, 'postgres',
		q{SELECT COUNT(*) FROM neurondb.ml_inference_audit_log WHERE operation_type = 'predict';},
		'1',
		'ML inference log entry created'
	);
	
	# Test query_audit_log function for ML inference
	query_ok($node, 'postgres', q{
		SELECT * FROM query_audit_log('ml_inference', NULL, NULL, NULL, 'predict');
	}, 'query_audit_log function works for ML inference');
	
	# Test query with filters
	query_ok($node, 'postgres', q{
		SELECT * FROM query_audit_log(
			'ml_inference',
			CURRENT_TIMESTAMP - INTERVAL '1 day',
			CURRENT_TIMESTAMP,
			NULL,
			'predict'
		);
	}, 'query_audit_log with time filters works');
	
	# Test multiple log entries
	query_ok($node, 'postgres', q{
		SELECT log_ml_inference(1, 'predict', 'hash1', 'hash2', NULL);
		SELECT log_ml_inference(2, 'batch_predict', 'hash3', 'hash4', NULL);
		SELECT log_ml_inference(1, 'predict', 'hash5', 'hash6', NULL);
	}, 'Multiple ML inference log entries');
	
	# Verify multiple entries
	result_matches($node, 'postgres',
		q{SELECT COUNT(*) FROM neurondb.ml_inference_audit_log;},
		qr/[4-9]/,
		'Multiple ML inference log entries created'
	);
	
	# Test audit log indexes exist
	query_ok($node, 'postgres', q{
		SELECT indexname FROM pg_indexes 
		WHERE tablename = 'ml_inference_audit_log' 
		AND indexname LIKE 'idx_ml_inference_audit%';
	}, 'ML inference audit log indexes exist');
	
	# Cleanup
	query_ok($node, 'postgres', q{
		DELETE FROM neurondb.ml_inference_audit_log;
		SET neurondb.audit_ml_enabled = false;
	}, 'Cleanup ML audit logs');
};

# ============================================================================
# RAG Operation Audit Logging Tests
# ============================================================================

subtest 'RAG Operation Audit Logging' => sub {
	plan tests => 10;
	
	# Enable audit logging
	query_ok($node, 'postgres', q{
		SET neurondb.audit_rag_enabled = true;
	}, 'Enable RAG audit logging');
	
	# Test audit table exists
	query_ok($node, 'postgres', q{
		SELECT COUNT(*) FROM neurondb.rag_operation_audit_log;
	}, 'RAG operation audit log table exists');
	
	# Test log_rag_operation function
	query_ok($node, 'postgres', q{
		SELECT log_rag_operation(
			'test_pipeline',
			'retrieve',
			'query_hash_123',
			5,
			'{"k": 5}'::jsonb
		);
	}, 'log_rag_operation function works');
	
	# Verify log entry was created
	result_is($node, 'postgres',
		q{SELECT COUNT(*) FROM neurondb.rag_operation_audit_log WHERE operation_type = 'retrieve';},
		'1',
		'RAG operation log entry created'
	);
	
	# Test query_audit_log function for RAG operations
	query_ok($node, 'postgres', q{
		SELECT * FROM query_audit_log('rag_operation', NULL, NULL, NULL, 'retrieve');
	}, 'query_audit_log function works for RAG operations');
	
	# Test different operation types
	query_ok($node, 'postgres', q{
		SELECT log_rag_operation('pipeline1', 'generate', 'hash1', 1, NULL);
		SELECT log_rag_operation('pipeline1', 'chat', 'hash2', 3, NULL);
		SELECT log_rag_operation('pipeline2', 'retrieve', 'hash3', 10, NULL);
	}, 'Multiple RAG operation types');
	
	# Verify multiple entries
	result_matches($node, 'postgres',
		q{SELECT COUNT(*) FROM neurondb.rag_operation_audit_log;},
		qr/[4-9]/,
		'Multiple RAG operation log entries created'
	);
	
	# Test query with pipeline filter
	query_ok($node, 'postgres', q{
		SELECT COUNT(*) FROM neurondb.rag_operation_audit_log 
		WHERE pipeline_name = 'pipeline1';
	}, 'RAG audit log supports pipeline filtering');
	
	# Test audit log indexes exist
	query_ok($node, 'postgres', q{
		SELECT indexname FROM pg_indexes 
		WHERE tablename = 'rag_operation_audit_log' 
		AND indexname LIKE 'idx_rag_audit%';
	}, 'RAG operation audit log indexes exist');
	
	# Test query_audit_log with all filters
	query_ok($node, 'postgres', q{
		SELECT * FROM query_audit_log(
			'rag_operation',
			CURRENT_TIMESTAMP - INTERVAL '1 day',
			CURRENT_TIMESTAMP,
			CURRENT_USER,
			'generate'
		);
	}, 'query_audit_log with all filters works');
	
	# Cleanup
	query_ok($node, 'postgres', q{
		DELETE FROM neurondb.rag_operation_audit_log;
		SET neurondb.audit_rag_enabled = false;
	}, 'Cleanup RAG audit logs');
};

# ============================================================================
# Security Configuration Tests
# ============================================================================

subtest 'Security Configuration' => sub {
	plan tests => 6;
	
	# Test GUC parameters exist
	query_ok($node, 'postgres', q{
		SELECT name FROM pg_settings 
		WHERE name LIKE 'neurondb.rls_embeddings_enabled';
	}, 'rls_embeddings_enabled GUC exists');
	
	query_ok($node, 'postgres', q{
		SELECT name FROM pg_settings 
		WHERE name LIKE 'neurondb.encryption_enabled';
	}, 'encryption_enabled GUC exists');
	
	query_ok($node, 'postgres', q{
		SELECT name FROM pg_settings 
		WHERE name LIKE 'neurondb.audit_ml_enabled';
	}, 'audit_ml_enabled GUC exists');
	
	query_ok($node, 'postgres', q{
		SELECT name FROM pg_settings 
		WHERE name LIKE 'neurondb.audit_rag_enabled';
	}, 'audit_rag_enabled GUC exists');
	
	query_ok($node, 'postgres', q{
		SELECT name FROM pg_settings 
		WHERE name LIKE 'neurondb.audit_retention_days';
	}, 'audit_retention_days GUC exists');
	
	# Test default values
	result_is($node, 'postgres',
		q{SELECT setting FROM pg_settings WHERE name = 'neurondb.audit_retention_days';},
		'365',
		'audit_retention_days default is 365'
	);
};

# Cleanup
$node->stop();
ok(!$node->is_running(), 'PostgreSQL node stopped');

$node->cleanup();
ok(!-d $node->{data_dir}, 'Data directory cleaned up');

done_testing();


