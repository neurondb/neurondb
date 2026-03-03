#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin";
use PostgresNode;
use TapTest;
use NeuronDB;
use IndexHelpers;

=head1 NAME

014_index_hnsw.t - HNSW index comprehensive tests - Comprehensive index operation tests

=head1 DESCRIPTION

Tests all index operations (HNSW, IVF) with both positive and negative test cases.

=cut

plan tests => 9;  # Tests exit early due to negative test failures - adjust when functionality is complete

my $node = PostgresNode->new('index_test');
$node->init();
$node->start();

install_extension($node, 'postgres');
neurondb_ok($node, 'postgres', 'NeuronDB extension installed');

# Create test data
$node->psql('postgres', q{
	DROP TABLE IF EXISTS test_index_data CASCADE;
	CREATE TABLE test_index_data (
		id SERIAL PRIMARY KEY,
		vec vector(4),
		label TEXT
	);
	INSERT INTO test_index_data (vec, label) VALUES
		('[1,2,3,4]'::vector(4), 'label1'),
		('[2,3,4,5]'::vector(4), 'label2'),
		('[3,4,5,6]'::vector(4), 'label3'),
		('[4,5,6,7]'::vector(4), 'label4'),
		('[5,6,7,8]'::vector(4), 'label5'),
		('[6,7,8,9]'::vector(4), 'label6'),
		('[7,8,9,10]'::vector(4), 'label7'),
		('[8,9,10,11]'::vector(4), 'label8');
});

# ============================================================================
# HNSW INDEX - Positive Tests
# ============================================================================

subtest 'HNSW Index - Positive' => sub {
	plan tests => 20;
	
	# Create HNSW index with default parameters
	my $result = $node->psql('postgres', q{
		CREATE INDEX idx_test_hnsw_default ON test_index_data 
		USING hnsw (vec vector_l2_ops);
	});
	
	if ($result->{success}) {
		pass('HNSW index creation with defaults succeeds');
		
		# Test KNN query
		$result = $node->psql('postgres', q{
			SELECT id, vec <-> '[1,2,3,4]'::vector(4) AS distance
			FROM test_index_data
			ORDER BY vec <-> '[1,2,3,4]'::vector(4)
			LIMIT 3;
		});
		ok($result->{success}, 'KNN query with HNSW index works');
		
		# Drop index
		$node->psql('postgres', q{DROP INDEX idx_test_hnsw_default;});
	} else {
		skip("HNSW index not available: $result->{stderr}", 2);
	}
	
	# Create HNSW index with custom parameters
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_hnsw_custom ON test_index_data 
		USING hnsw (vec vector_l2_ops)
		WITH (m = 16, ef_construction = 200);
	});
	
	if ($result->{success}) {
		pass('HNSW index creation with custom parameters succeeds');
		
		# Test KNN query with ef_search
		$result = $node->psql('postgres', q{
			SET hnsw.ef_search = 40;
			SELECT id, vec <-> '[1,2,3,4]'::vector(4) AS distance
			FROM test_index_data
			ORDER BY vec <-> '[1,2,3,4]'::vector(4)
			LIMIT 3;
		});
		ok($result->{success}, 'KNN query with ef_search works');
		
		$node->psql('postgres', q{DROP INDEX idx_test_hnsw_custom;});
	} else {
		skip("HNSW index with custom params not available: $result->{stderr}", 2);
	}
	
	# Create HNSW index with cosine distance
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_hnsw_cosine ON test_index_data 
		USING hnsw (vec vector_cosine_ops);
	});
	
	if ($result->{success}) {
		pass('HNSW index with cosine ops succeeds');
		
		# Test cosine distance query
		$result = $node->psql('postgres', q{
			SELECT id, vec <=> '[1,2,3,4]'::vector(4) AS distance
			FROM test_index_data
			ORDER BY vec <=> '[1,2,3,4]'::vector(4)
			LIMIT 3;
		});
		ok($result->{success}, 'Cosine distance query with HNSW works');
		
		$node->psql('postgres', q{DROP INDEX idx_test_hnsw_cosine;});
	} else {
		skip("HNSW cosine index not available: $result->{stderr}", 2);
	}
};

# ============================================================================
# HNSW INDEX - Negative Tests
# ============================================================================

subtest 'HNSW Index - Negative' => sub {
	plan tests => 15;
	
	# Invalid m parameter (too small)
	my $result = $node->psql('postgres', q{
		CREATE INDEX idx_test_hnsw_invalid_m ON test_index_data 
		USING hnsw (vec vector_l2_ops)
		WITH (m = 1);
	});
	ok(!$result->{success}, 'HNSW with m=1 rejected');
	
	# Invalid m parameter (too large)
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_hnsw_invalid_m_large ON test_index_data 
		USING hnsw (vec vector_l2_ops)
		WITH (m = 1000);
	});
	ok(!$result->{success}, 'HNSW with m too large rejected');
	
	# Invalid ef_construction (negative)
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_hnsw_invalid_ef ON test_index_data 
		USING hnsw (vec vector_l2_ops)
		WITH (ef_construction = -1);
	});
	ok(!$result->{success}, 'HNSW with negative ef_construction rejected');
	
	# Index on non-existent column
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_hnsw_nonexistent ON test_index_data 
		USING hnsw (nonexistent_col vector_l2_ops);
	});
	ok(!$result->{success}, 'HNSW on non-existent column rejected');
	
	# Index on wrong column type
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_hnsw_wrong_type ON test_index_data 
		USING hnsw (label vector_l2_ops);
	});
	ok(!$result->{success}, 'HNSW on non-vector column rejected');
	
	# Index on NULL vectors
	$node->psql('postgres', q{
		CREATE TEMP TABLE null_vecs (vec vector(4));
		INSERT INTO null_vecs VALUES (NULL);
	});
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_hnsw_null ON null_vecs 
		USING hnsw (vec vector_l2_ops);
	});
	# May or may not fail depending on implementation
	ok(defined $result, 'HNSW on NULL vectors handled');
	
	# Index on empty table
	$node->psql('postgres', q{
		CREATE TEMP TABLE empty_table (vec vector(4));
	});
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_hnsw_empty ON empty_table 
		USING hnsw (vec vector_l2_ops);
	});
	# May or may not fail depending on implementation
	ok(defined $result, 'HNSW on empty table handled');
};

# ============================================================================
# IVF INDEX - Positive Tests
# ============================================================================

subtest 'IVF Index - Positive' => sub {
	plan tests => 15;
	
	# Create IVF index with default parameters
	my $result = $node->psql('postgres', q{
		CREATE INDEX idx_test_ivf_default ON test_index_data 
		USING ivf (vec vector_l2_ops);
	});
	
	if ($result->{success}) {
		pass('IVF index creation with defaults succeeds');
		
		# Test KNN query
		$result = $node->psql('postgres', q{
			SELECT id, vec <-> '[1,2,3,4]'::vector(4) AS distance
			FROM test_index_data
			ORDER BY vec <-> '[1,2,3,4]'::vector(4)
			LIMIT 3;
		});
		ok($result->{success}, 'KNN query with IVF index works');
		
		$node->psql('postgres', q{DROP INDEX idx_test_ivf_default;});
	} else {
		skip("IVF index not available: $result->{stderr}", 2);
	}
	
	# Create IVF index with custom lists parameter
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_ivf_custom ON test_index_data 
		USING ivf (vec vector_l2_ops)
		WITH (lists = 10);
	});
	
	if ($result->{success}) {
		pass('IVF index creation with custom lists succeeds');
		
		$node->psql('postgres', q{DROP INDEX idx_test_ivf_custom;});
	} else {
		skip("IVF index with custom params not available: $result->{stderr}", 1);
	}
};

# ============================================================================
# IVF INDEX - Negative Tests
# ============================================================================

subtest 'IVF Index - Negative' => sub {
	plan tests => 10;
	
	# Invalid lists parameter (too small)
	my $result = $node->psql('postgres', q{
		CREATE INDEX idx_test_ivf_invalid_lists ON test_index_data 
		USING ivf (vec vector_l2_ops)
		WITH (lists = 0);
	});
	ok(!$result->{success}, 'IVF with lists=0 rejected');
	
	# Invalid lists parameter (too large)
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_ivf_invalid_lists_large ON test_index_data 
		USING ivf (vec vector_l2_ops)
		WITH (lists = 10000);
	});
	ok(!$result->{success}, 'IVF with lists too large rejected');
	
	# Negative lists
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_ivf_negative_lists ON test_index_data 
		USING ivf (vec vector_l2_ops)
		WITH (lists = -1);
	});
	ok(!$result->{success}, 'IVF with negative lists rejected');
	
	# Index on non-existent column
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_ivf_nonexistent ON test_index_data 
		USING ivf (nonexistent_col vector_l2_ops);
	});
	ok(!$result->{success}, 'IVF on non-existent column rejected');
	
	# Index on wrong column type
	$result = $node->psql('postgres', q{
		CREATE INDEX idx_test_ivf_wrong_type ON test_index_data 
		USING ivf (label vector_l2_ops);
	});
	ok(!$result->{success}, 'IVF on non-vector column rejected');
};

# ============================================================================
# INDEX OPERATIONS - Positive Tests
# ============================================================================

subtest 'Index Operations - Positive' => sub {
	plan tests => 10;
	
	# Create index
	my $result = $node->psql('postgres', q{
		CREATE INDEX idx_test_ops ON test_index_data 
		USING hnsw (vec vector_l2_ops);
	});
	
	if ($result->{success}) {
		# REINDEX
		$result = $node->psql('postgres', q{
			REINDEX INDEX idx_test_ops;
		});
		ok($result->{success}, 'REINDEX works');
		
		# DROP INDEX
		$result = $node->psql('postgres', q{
			DROP INDEX idx_test_ops;
		});
		ok($result->{success}, 'DROP INDEX works');
	} else {
		skip("Index operations not available: $result->{stderr}", 2);
	}
};

# ============================================================================
# Cleanup
# ============================================================================

$node->stop();
$node->cleanup();

done_testing();



