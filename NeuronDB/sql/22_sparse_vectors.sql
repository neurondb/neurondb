-- ============================================================================
-- NeurondB: Sparse Vectors & Learned Sparse Retrieval
-- ============================================================================
-- Implements sparse_vector type for SPLADE/ColBERTv2/BM25 retrieval
-- with inverted index support and hybrid dense+sparse search.
--
-- Copyright (c) 2024-2025, neurondb, Inc.
-- ============================================================================

-- ============================================================================
-- SPARSE VECTOR TYPE
-- ============================================================================

CREATE TYPE sparse_vector (
	INPUT = sparse_vector_in,
	OUTPUT = sparse_vector_out,
	RECEIVE = sparse_vector_recv,
	SEND = sparse_vector_send,
	INTERNALLENGTH = VARIABLE,
	ALIGNMENT = double,
	STORAGE = extended
);

COMMENT ON TYPE sparse_vector IS
	'Learned sparse vector type for SPLADE/ColBERTv2/BM25 retrieval';

-- ============================================================================
-- SPARSE VECTOR OPERATORS
-- ============================================================================

CREATE FUNCTION sparse_vector_dot_product(sparse_vector, sparse_vector)
RETURNS float4
AS 'MODULE_PATHNAME', 'sparse_vector_dot_product'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION sparse_vector_dot_product(sparse_vector, sparse_vector) IS
	'Compute dot product between two sparse vectors';

CREATE OPERATOR <*> (
	LEFTARG = sparse_vector,
	RIGHTARG = sparse_vector,
	PROCEDURE = sparse_vector_dot_product
);

COMMENT ON OPERATOR <*>(sparse_vector, sparse_vector) IS
	'Dot product operator for sparse vectors';

-- ============================================================================
-- SPARSE INDEX FUNCTIONS
-- ============================================================================

CREATE FUNCTION sparse_index_create(
	table_name text,
	sparse_col text,
	index_name text,
	min_freq int4 DEFAULT 1
)
RETURNS bool
AS 'MODULE_PATHNAME', 'sparse_index_create'
LANGUAGE C STRICT;

COMMENT ON FUNCTION sparse_index_create(text, text, text, int4) IS
	'Create inverted index for sparse vectors';

CREATE FUNCTION sparse_index_search(
	index_name text,
	query_vec sparse_vector,
	k int4
)
RETURNS TABLE(doc_id int4, score float4)
AS 'MODULE_PATHNAME', 'sparse_index_search'
LANGUAGE C STRICT;

COMMENT ON FUNCTION sparse_index_search(text, sparse_vector, int4) IS
	'Search sparse index with query vector';

CREATE FUNCTION sparse_index_update(
	index_name text,
	doc_id int4,
	sparse_vec sparse_vector
)
RETURNS bool
AS 'MODULE_PATHNAME', 'sparse_index_update'
LANGUAGE C STRICT;

COMMENT ON FUNCTION sparse_index_update(text, int4, sparse_vector) IS
	'Update sparse index with new document';

-- ============================================================================
-- SPARSE SEARCH FUNCTIONS
-- ============================================================================

CREATE FUNCTION sparse_search(
	table_name text,
	sparse_col text,
	query_vec sparse_vector,
	k int4
)
RETURNS TABLE(doc_id int4, score float4)
AS 'MODULE_PATHNAME', 'sparse_search'
LANGUAGE C STRICT;

COMMENT ON FUNCTION sparse_search(text, text, sparse_vector, int4) IS
	'Search table using sparse vector query';

CREATE FUNCTION splade_embed(input_text text)
RETURNS sparse_vector
AS 'MODULE_PATHNAME', 'splade_embed'
LANGUAGE C STRICT;

COMMENT ON FUNCTION splade_embed(text) IS
	'Generate SPLADE sparse embedding from text';

CREATE FUNCTION colbertv2_embed(input_text text)
RETURNS sparse_vector
AS 'MODULE_PATHNAME', 'colbertv2_embed'
LANGUAGE C STRICT;

COMMENT ON FUNCTION colbertv2_embed(text) IS
	'Generate ColBERTv2 sparse embedding from text';

CREATE FUNCTION bm25_score(
	query_text text,
	doc_text text,
	k1 float4 DEFAULT 1.5,
	b float4 DEFAULT 0.75
)
RETURNS float4
AS 'MODULE_PATHNAME', 'bm25_score'
LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

COMMENT ON FUNCTION bm25_score(text, text, float4, float4) IS
	'Compute BM25 score between query and document';

-- ============================================================================
-- HYBRID DENSE + SPARSE SEARCH
-- ============================================================================

CREATE FUNCTION hybrid_dense_sparse_search(
	table_name text,
	dense_col text,
	sparse_col text,
	dense_query vector,
	sparse_query sparse_vector,
	k int4,
	dense_weight float4 DEFAULT 0.5,
	sparse_weight float4 DEFAULT 0.5
)
RETURNS TABLE(doc_id int4, fused_score float4)
AS 'MODULE_PATHNAME', 'hybrid_dense_sparse_search'
LANGUAGE C STRICT;

COMMENT ON FUNCTION hybrid_dense_sparse_search(text, text, text, vector, sparse_vector, int4, float4, float4) IS
	'Hybrid search combining dense vector (HNSW) and sparse vector (inverted index)';

CREATE FUNCTION rrf_fusion(
	k int4,
	dense_rank float4,
	sparse_rank float4,
	k_param float4 DEFAULT 60.0
)
RETURNS float4
AS 'MODULE_PATHNAME', 'rrf_fusion'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION rrf_fusion(int4, float4, float4, float4) IS
	'Reciprocal Rank Fusion for combining dense and sparse search results';

-- ============================================================================
-- EXAMPLE USAGE
-- ============================================================================

/*
-- Create table with both dense and sparse vectors
CREATE TABLE documents (
	id serial PRIMARY KEY,
	content text,
	dense_embedding vector(768),
	sparse_embedding sparse_vector
);

-- Create indexes
CREATE INDEX ON documents USING hnsw (dense_embedding vector_l2_ops);
SELECT sparse_index_create('documents', 'sparse_embedding', 'idx_sparse_docs');

-- Generate sparse embeddings
UPDATE documents SET sparse_embedding = splade_embed(content);

-- Hybrid search
SELECT * FROM hybrid_dense_sparse_search(
	'documents',
	'dense_embedding',
	'sparse_embedding',
	'[0.1,0.2,...]'::vector,
	'{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.5,0.8]}'::sparse_vector,
	10,
	0.6,  -- dense weight
	0.4   -- sparse weight
);
*/

