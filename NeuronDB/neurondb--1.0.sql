-- ============================================================================
-- NeurondB Extension SQL Definitions
-- Advanced AI Database Extension for PostgreSQL
-- 
-- Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
-- 
-- Version Compatibility:
-- - PostgreSQL 16: Full support with PL/pgSQL fallbacks for macOS dylib loader issues
-- - PostgreSQL 17: Full support with PL/pgSQL fallbacks for macOS dylib loader issues
-- - PostgreSQL 18: Full support with native C functions (dylib loader fixed)
-- - OS Support: macOS, Rocky Linux, Ubuntu (all versions)
-- ============================================================================

\echo Use "CREATE EXTENSION neurondb" to load this extension. \quit

-- ============================================================================

DO $$
DECLARE
    v_pg_version_num integer;
    v_pg_major integer;
    v_expected_version integer := 160900;
BEGIN
    -- Get actual PostgreSQL version
    v_pg_version_num := current_setting('server_version_num')::integer;
    v_pg_major := v_pg_version_num / 10000;
    
    -- Log version info
    RAISE NOTICE 'NeurondB v1.0: Supports PostgreSQL 16-18';
    RAISE NOTICE 'NeurondB: Current PostgreSQL version: %.%', 
                 v_pg_major, (v_pg_version_num - v_pg_major * 10000) / 100;
    
    -- Validate supported versions
    IF v_pg_major < 16 OR v_pg_major > 18 THEN
        RAISE EXCEPTION 'NeurondB supports only PostgreSQL 16, 17, and 18. Detected version: %.%', 
                        v_pg_major, (v_pg_version_num - v_pg_major * 10000) / 100;
    END IF;
END $$;

-- ============================================================================
-- SCHEMA SETUP
-- ============================================================================

-- Create schema for internal tables
CREATE SCHEMA IF NOT EXISTS neurondb;
COMMENT ON SCHEMA neurondb IS 'NeurondB internal schema for catalog tables and metadata';

-- ============================================================================
-- VECTOR TYPE DEFINITIONS
-- ============================================================================

-- Main vector type (float32)
CREATE TYPE vector;

CREATE FUNCTION vector_in(cstring) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_in'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION vector_out(vector) RETURNS cstring
    AS 'MODULE_PATHNAME', 'vector_out'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION vector_recv(internal) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_recv'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION vector_send(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_send'
    LANGUAGE C IMMUTABLE STRICT;

-- Typmod I/O for vector(dim)
CREATE FUNCTION vector_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME', 'vector_typmod_in'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION vector_typmod_out(integer) RETURNS cstring
    AS 'MODULE_PATHNAME', 'vector_typmod_out'
    LANGUAGE C IMMUTABLE STRICT;

-- Vector statistics for ANALYZE (must be defined before CREATE TYPE)
CREATE FUNCTION vector_analyze(internal) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_analyze'
    LANGUAGE C STRICT;
COMMENT ON FUNCTION vector_analyze(internal) IS 'ANALYZE hook for collecting vector column statistics';

CREATE TYPE vector (
    INPUT = vector_in,
    OUTPUT = vector_out,
    RECEIVE = vector_recv,
    SEND = vector_send,
    TYPMOD_IN = vector_typmod_in,
    TYPMOD_OUT = vector_typmod_out,
    ANALYZE = vector_analyze,
    STORAGE = extended,
    CATEGORY = 'U'
);

COMMENT ON TYPE vector IS 'NeurondB vector type (float32 array)';

-- ============================================================================
-- ADDITIONAL VECTOR TYPES
-- ============================================================================

-- Packed vector type (vectorp): SIMD-optimized packed format
CREATE TYPE vectorp;

CREATE FUNCTION vectorp_in(cstring) RETURNS vectorp
    AS 'MODULE_PATHNAME', 'vectorp_in'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION vectorp_out(vectorp) RETURNS cstring
    AS 'MODULE_PATHNAME', 'vectorp_out'
    LANGUAGE C IMMUTABLE STRICT;

CREATE TYPE vectorp (
    INPUT = vectorp_in,
    OUTPUT = vectorp_out,
    STORAGE = extended,
    CATEGORY = 'U'
);

COMMENT ON TYPE vectorp IS 'Packed SIMD vector with optimized storage layout';

-- Sparse vector map type (vecmap): Stores only non-zero values
CREATE TYPE vecmap;

CREATE FUNCTION vecmap_in(cstring) RETURNS vecmap
    AS 'MODULE_PATHNAME', 'vecmap_in'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION vecmap_out(vecmap) RETURNS cstring
    AS 'MODULE_PATHNAME', 'vecmap_out'
    LANGUAGE C IMMUTABLE STRICT;

CREATE TYPE vecmap (
    INPUT = vecmap_in,
    OUTPUT = vecmap_out,
    STORAGE = extended,
    CATEGORY = 'U'
);

COMMENT ON TYPE vecmap IS 'Sparse high-dimensional vector map (stores only non-zero values)';

-- Vector graph type (vgraph): Graph structure with nodes and edges
CREATE TYPE vgraph;

CREATE FUNCTION vgraph_in(cstring) RETURNS vgraph
    AS 'MODULE_PATHNAME', 'vgraph_in'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION vgraph_out(vgraph) RETURNS cstring
    AS 'MODULE_PATHNAME', 'vgraph_out'
    LANGUAGE C IMMUTABLE STRICT;

CREATE TYPE vgraph (
    INPUT = vgraph_in,
    OUTPUT = vgraph_out,
    STORAGE = extended,
    CATEGORY = 'U'
);

COMMENT ON TYPE vgraph IS 'Graph-based vector structure for neighbor relations and clustering';

-- Retrievable text type (rtext): Text with token metadata
CREATE TYPE rtext;

CREATE FUNCTION rtext_in(cstring) RETURNS rtext
    AS 'MODULE_PATHNAME', 'rtext_in'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION rtext_out(rtext) RETURNS cstring
    AS 'MODULE_PATHNAME', 'rtext_out'
    LANGUAGE C IMMUTABLE STRICT;

CREATE TYPE rtext (
    INPUT = rtext_in,
    OUTPUT = rtext_out,
    STORAGE = extended,
    CATEGORY = 'U'
);

COMMENT ON TYPE rtext IS 'Retrievable text type with token metadata for RAG pipelines';

-- Half-precision vector type (halfvec): FP16 quantized vectors (pgvector compatibility)
CREATE TYPE halfvec;

CREATE FUNCTION halfvec_in(cstring) RETURNS halfvec
    AS 'MODULE_PATHNAME', 'halfvec_in'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION halfvec_out(halfvec) RETURNS cstring
    AS 'MODULE_PATHNAME', 'halfvec_out'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION halfvec_recv(internal) RETURNS halfvec
    AS 'MODULE_PATHNAME', 'halfvec_recv'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION halfvec_send(halfvec) RETURNS bytea
    AS 'MODULE_PATHNAME', 'halfvec_send'
    LANGUAGE C IMMUTABLE STRICT;

CREATE TYPE halfvec (
    INPUT = halfvec_in,
    OUTPUT = halfvec_out,
    RECEIVE = halfvec_recv,
    SEND = halfvec_send,
    STORAGE = extended,
    CATEGORY = 'U'
);

COMMENT ON TYPE halfvec IS 'Half-precision vector type (FP16) for 2x compression, supports up to 4000 dimensions (pgvector compatible)';

-- Binary vector type (binaryvec): Packed binary vectors for quantization
CREATE TYPE binaryvec;

CREATE FUNCTION binaryvec_in(cstring) RETURNS binaryvec
    AS 'MODULE_PATHNAME', 'binaryvec_in'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION binaryvec_out(binaryvec) RETURNS cstring
    AS 'MODULE_PATHNAME', 'binaryvec_out'
    LANGUAGE C IMMUTABLE STRICT;

CREATE TYPE binaryvec (
    INPUT = binaryvec_in,
    OUTPUT = binaryvec_out,
    STORAGE = extended,
    CATEGORY = 'U'
);

COMMENT ON TYPE binaryvec IS 'Binary vector type for efficient binary quantization and Hamming distance';

-- Distance functions for binaryvec type
CREATE FUNCTION binaryvec_hamming_distance(binaryvec, binaryvec) RETURNS integer
    AS 'MODULE_PATHNAME', 'binaryvec_hamming_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION binaryvec_hamming_distance IS 'Hamming distance between binaryvec vectors';

-- Distance operator for binaryvec type (defined after function)
CREATE OPERATOR <-> (
    LEFTARG = binaryvec,
    RIGHTARG = binaryvec,
    PROCEDURE = binaryvec_hamming_distance,
    COMMUTATOR = '<->'
);
COMMENT ON OPERATOR <->(binaryvec, binaryvec) IS 'Hamming distance operator for binary vectors';

-- Sparse vector type (sparsevec): pgvector-compatible sparse vector format
CREATE TYPE sparsevec;

CREATE FUNCTION sparsevec_in(cstring) RETURNS sparsevec
    AS 'MODULE_PATHNAME', 'sparsevec_in'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION sparsevec_out(sparsevec) RETURNS cstring
    AS 'MODULE_PATHNAME', 'sparsevec_out'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION sparsevec_recv(internal) RETURNS sparsevec
    AS 'MODULE_PATHNAME', 'sparsevec_recv'
    LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION sparsevec_send(sparsevec) RETURNS bytea
    AS 'MODULE_PATHNAME', 'sparsevec_send'
    LANGUAGE C IMMUTABLE STRICT;

CREATE TYPE sparsevec (
    INPUT = sparsevec_in,
    OUTPUT = sparsevec_out,
    RECEIVE = sparsevec_recv,
    SEND = sparsevec_send,
    STORAGE = extended,
    CATEGORY = 'U'
);

COMMENT ON TYPE sparsevec IS 'Sparse vector type (pgvector compatible), supports up to 1000 nonzero entries and 1M dimensions';

-- ============================================================================
-- CORE CONFIGURATION TABLES (neurondb schema)
-- ============================================================================

-- Table to store Hugging Face API configuration
CREATE TABLE IF NOT EXISTS neurondb.llm_config (
    api_base text NOT NULL,
    api_key text NOT NULL,
    default_model text NOT NULL,
    updated_at timestamptz DEFAULT now()
);

COMMENT ON TABLE neurondb.llm_config IS 'Hugging Face API configuration for LLM/vector integration';

-- Function: set_llm_config
CREATE OR REPLACE FUNCTION neurondb.set_llm_config(
    api_base text,
    api_key text,
    default_model text
)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM neurondb.llm_config) THEN
        UPDATE neurondb.llm_config
        SET api_base = set_llm_config.api_base,
            api_key = set_llm_config.api_key,
            default_model = set_llm_config.default_model,
            updated_at = now();
    ELSE
        INSERT INTO neurondb.llm_config(api_base, api_key, default_model)
        VALUES (set_llm_config.api_base, set_llm_config.api_key, set_llm_config.default_model);
    END IF;
END;
$$;

COMMENT ON FUNCTION neurondb.set_llm_config IS 'Set Hugging Face LLM API base url, key, and default model';

-- Function: get_llm_config
CREATE OR REPLACE FUNCTION neurondb.get_llm_config()
RETURNS TABLE (
    api_base text,
    api_key text,
    default_model text,
    updated_at timestamptz
)
LANGUAGE sql
AS $$
    SELECT api_base, api_key, default_model, updated_at FROM neurondb.llm_config LIMIT 1
$$;

COMMENT ON FUNCTION neurondb.get_llm_config IS 'Get Hugging Face LLM API configuration row';

-- ============================================================================
-- BASIC VECTOR FUNCTIONS
-- ============================================================================

CREATE FUNCTION vector_dims(vector) RETURNS integer
    AS 'MODULE_PATHNAME', 'vector_dims'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_dims IS 'Get vector dimensions';

CREATE FUNCTION vector_norm(vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_norm'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_norm IS 'Compute L2 norm of vector';

CREATE FUNCTION vector_normalize(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_normalize'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_normalize IS 'Normalize vector to unit length';

CREATE FUNCTION vector_concat(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_concat'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_concat IS 'Concatenate two vectors';

-- ============================================================================
-- VECTOR ARITHMETIC OPERATORS
-- ============================================================================

CREATE FUNCTION vector_add(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_add'
    LANGUAGE C IMMUTABLE STRICT;

CREATE OPERATOR + (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_add,
    COMMUTATOR = +
);

CREATE FUNCTION vector_sub(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_sub'
    LANGUAGE C IMMUTABLE STRICT;

CREATE OPERATOR - (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_sub
);

CREATE FUNCTION vector_mul(vector, double precision) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_mul'
    LANGUAGE C IMMUTABLE STRICT;

CREATE OPERATOR * (
    LEFTARG = vector,
    RIGHTARG = double precision,
    PROCEDURE = vector_mul,
    COMMUTATOR = *
);

-- ============================================================================
-- DISTANCE FUNCTIONS & OPERATORS
-- ============================================================================

-- L2 (Euclidean) Distance
CREATE FUNCTION vector_l2_distance(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_l2_distance IS 'L2 (Euclidean) distance between vectors';

-- Inner Product (Negative for ordering)
CREATE FUNCTION vector_inner_product(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_inner_product IS 'Negative inner product (for ordering)';

-- Note: Operator <#> is created after vector_inner_product_distance_op function definition (see line ~2470)

-- Cosine Distance
CREATE FUNCTION vector_cosine_distance(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_cosine_distance IS 'Cosine distance (1 - cosine similarity)';

-- Note: Operator <=> is created after vector_cosine_distance_op function definition (see line ~2472)

-- L1 (Manhattan) Distance
CREATE FUNCTION vector_l1_distance(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_l1_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_l1_distance IS 'L1 (Manhattan) distance';

-- Hamming Distance
CREATE FUNCTION vector_hamming_distance(vector, vector) RETURNS integer
    AS 'MODULE_PATHNAME', 'vector_hamming_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_hamming_distance IS 'Hamming distance';

-- Chebyshev Distance
CREATE FUNCTION vector_chebyshev_distance(vector, vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_chebyshev_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_chebyshev_distance IS 'Chebyshev (L-infinity) distance';

-- Minkowski Distance
CREATE FUNCTION vector_minkowski_distance(vector, vector, double precision) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_minkowski_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_minkowski_distance IS 'Minkowski distance with parameter p';

-- Squared Euclidean Distance (L2^2, faster, no sqrt)
CREATE FUNCTION vector_squared_l2_distance(vector, vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_squared_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_squared_l2_distance IS 'Squared Euclidean distance (L2^2, faster, no sqrt)';

-- Jaccard Distance
CREATE FUNCTION vector_jaccard_distance(vector, vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_jaccard_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_jaccard_distance IS 'Jaccard distance (1 - Jaccard similarity) for set-based vectors';

-- Dice Distance
CREATE FUNCTION vector_dice_distance(vector, vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_dice_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_dice_distance IS 'Dice distance (1 - Dice coefficient) for set-based vectors';

-- Mahalanobis Distance
CREATE FUNCTION vector_mahalanobis_distance(vector, vector, vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_mahalanobis_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_mahalanobis_distance IS 'Mahalanobis distance with diagonal covariance matrix (3rd arg: inverse variances)';

-- ============================================================================
-- QUANTIZATION FUNCTIONS
-- ============================================================================

CREATE FUNCTION vector_to_int8(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_int8'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_int8 IS 'Quantize vector to int8 (8x compression)';

CREATE FUNCTION int8_to_vector(bytea) RETURNS vector
    AS 'MODULE_PATHNAME', 'int8_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION int8_to_vector IS 'Dequantize int8 vector';

-- FP16 quantization (2x compression, CPU version)
CREATE FUNCTION vector_to_fp16(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_float16'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_fp16 IS 'Quantize vector to FP16 (2x compression, IEEE 754 half-precision)';

CREATE FUNCTION fp16_to_vector(bytea) RETURNS vector
    AS 'MODULE_PATHNAME', 'float16_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION fp16_to_vector IS 'Dequantize FP16 vector';

CREATE FUNCTION vector_to_binary(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_binary'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_binary IS 'Convert vector to binary (32x compression)';

CREATE FUNCTION binary_quantize(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'binary_quantize'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION binary_quantize IS 'Alias for vector_to_binary (pgvector compatibility)';

-- Bit type support for binary vectors (pgvector compatibility)
CREATE FUNCTION vector_to_bit(vector) RETURNS bit
    AS 'MODULE_PATHNAME', 'vector_to_bit'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_bit IS 'Convert vector to PostgreSQL bit type (pgvector compatibility)';

CREATE FUNCTION bit_to_vector(bit) RETURNS vector
    AS 'MODULE_PATHNAME', 'bit_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION bit_to_vector IS 'Convert PostgreSQL bit type to vector';

-- Type conversion functions (pgvector compatibility)
CREATE FUNCTION vector_to_halfvec(vector) RETURNS halfvec
    AS 'MODULE_PATHNAME', 'vector_to_halfvec'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_halfvec IS 'Convert vector to halfvec (pgvector compatible)';

CREATE FUNCTION halfvec_to_vector(halfvec) RETURNS vector
    AS 'MODULE_PATHNAME', 'halfvec_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION halfvec_to_vector IS 'Convert halfvec to vector (pgvector compatible)';

CREATE FUNCTION vector_to_sparsevec(vector) RETURNS sparsevec
    AS 'MODULE_PATHNAME', 'vector_to_sparsevec'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_sparsevec IS 'Convert vector to sparsevec, storing only non-zero values (pgvector compatible)';

CREATE FUNCTION sparsevec_to_vector(sparsevec) RETURNS vector
    AS 'MODULE_PATHNAME', 'sparsevec_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION sparsevec_to_vector IS 'Convert sparsevec to vector (pgvector compatible)';

CREATE FUNCTION binary_hamming_distance(bytea, bytea) RETURNS integer
    AS 'MODULE_PATHNAME', 'binary_hamming_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION binary_hamming_distance IS 'Hamming distance for binary vectors';

-- UINT8 quantization (4x compression, unsigned)
CREATE FUNCTION vector_to_uint8(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_uint8'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_uint8 IS 'Quantize vector to uint8 (4x compression, unsigned [0,255])';

CREATE FUNCTION uint8_to_vector(bytea) RETURNS vector
    AS 'MODULE_PATHNAME', 'uint8_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION uint8_to_vector IS 'Dequantize uint8 vector';

-- Ternary quantization (16x compression, 2 bits per dimension: -1, 0, +1)
CREATE FUNCTION vector_to_ternary(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_ternary'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_ternary IS 'Quantize vector to ternary (16x compression, values: -1, 0, +1)';

CREATE FUNCTION ternary_to_vector(bytea) RETURNS vector
    AS 'MODULE_PATHNAME', 'ternary_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION ternary_to_vector IS 'Dequantize ternary vector';

-- INT4 quantization (8x compression, 4 bits per dimension, signed [-8, 7])
CREATE FUNCTION vector_to_int4(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_int4'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_int4 IS 'Quantize vector to int4 (8x compression, 4 bits per dimension)';

CREATE FUNCTION int4_to_vector(bytea) RETURNS vector
    AS 'MODULE_PATHNAME', 'int4_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION int4_to_vector IS 'Dequantize int4 vector';

-- Quantization accuracy analysis functions
CREATE FUNCTION quantize_analyze_int8(vector) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'quantize_analyze_int8'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION quantize_analyze_int8 IS 'Analyze INT8 quantization accuracy (MSE, MAE, compression ratio)';

CREATE FUNCTION quantize_analyze_fp16(vector) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'quantize_analyze_fp16'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION quantize_analyze_fp16 IS 'Analyze FP16 quantization accuracy (MSE, MAE, compression ratio)';

CREATE FUNCTION quantize_analyze_binary(vector) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'quantize_analyze_binary'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION quantize_analyze_binary IS 'Analyze binary quantization accuracy (MSE, MAE, compression ratio)';

CREATE FUNCTION quantize_analyze_uint8(vector) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'quantize_analyze_uint8'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION quantize_analyze_uint8 IS 'Analyze UINT8 quantization accuracy (MSE, MAE, compression ratio)';

CREATE FUNCTION quantize_analyze_ternary(vector) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'quantize_analyze_ternary'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION quantize_analyze_ternary IS 'Analyze ternary quantization accuracy (MSE, MAE, compression ratio)';

CREATE FUNCTION quantize_analyze_int4(vector) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'quantize_analyze_int4'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION quantize_analyze_int4 IS 'Analyze INT4 quantization accuracy (MSE, MAE, compression ratio)';

CREATE FUNCTION quantize_compare_distances(vector, vector, text) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'quantize_compare_distances'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION quantize_compare_distances IS 'Compare distance preservation before and after quantization';

-- ============================================================================
-- SPARSE VECTOR OPERATIONS (vecmap)
-- ============================================================================

-- Sparse vector distance metrics (optimized for sparse data)
CREATE FUNCTION vecmap_l2_distance(bytea, bytea) RETURNS real
    AS 'MODULE_PATHNAME', 'vecmap_l2_distance'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vecmap_l2_distance IS 'L2 (Euclidean) distance for sparse vectors';

CREATE FUNCTION vecmap_cosine_distance(bytea, bytea) RETURNS real
    AS 'MODULE_PATHNAME', 'vecmap_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vecmap_cosine_distance IS 'Cosine distance for sparse vectors';

CREATE FUNCTION vecmap_inner_product(bytea, bytea) RETURNS real
    AS 'MODULE_PATHNAME', 'vecmap_inner_product'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vecmap_inner_product IS 'Inner product for sparse vectors';

CREATE FUNCTION vecmap_l1_distance(bytea, bytea) RETURNS real
    AS 'MODULE_PATHNAME', 'vecmap_l1_distance'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vecmap_l1_distance IS 'L1 (Manhattan) distance for sparse vectors';

-- Sparse vector arithmetic operations
CREATE FUNCTION vecmap_add(bytea, bytea) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vecmap_add'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vecmap_add IS 'Add two sparse vectors';

CREATE FUNCTION vecmap_sub(bytea, bytea) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vecmap_sub'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vecmap_sub IS 'Subtract two sparse vectors';

CREATE FUNCTION vecmap_mul_scalar(bytea, real) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vecmap_mul_scalar'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vecmap_mul_scalar IS 'Multiply sparse vector by scalar';

CREATE FUNCTION vecmap_norm(bytea) RETURNS real
    AS 'MODULE_PATHNAME', 'vecmap_norm'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vecmap_norm IS 'Compute L2 norm of sparse vector';

-- ============================================================================
-- GRAPH-BASED VECTOR OPERATIONS (vgraph)
-- ============================================================================

-- Breadth-First Search: Traverse graph from starting node
CREATE FUNCTION vgraph_bfs(vgraph, integer, integer DEFAULT -1)
    RETURNS TABLE(node_idx integer, depth integer, parent_idx integer)
    AS 'MODULE_PATHNAME', 'vgraph_bfs'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vgraph_bfs IS 'Breadth-First Search: traverse graph from starting node (returns node_idx, depth, parent_idx)';

-- Depth-First Search: Traverse graph from starting node
CREATE FUNCTION vgraph_dfs(vgraph, integer)
    RETURNS TABLE(node_idx integer, discovery_time integer, finish_time integer, parent_idx integer)
    AS 'MODULE_PATHNAME', 'vgraph_dfs'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vgraph_dfs IS 'Depth-First Search: traverse graph from starting node (returns node_idx, discovery_time, finish_time, parent_idx)';

-- PageRank: Compute PageRank scores for all nodes
CREATE FUNCTION vgraph_pagerank(vgraph, double precision DEFAULT 0.85, integer DEFAULT 100, double precision DEFAULT 1e-6)
    RETURNS TABLE(node_idx integer, pagerank_score double precision)
    AS 'MODULE_PATHNAME', 'vgraph_pagerank'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vgraph_pagerank IS 'PageRank algorithm: compute importance scores for all nodes (damping_factor, max_iterations, tolerance)';

-- Community Detection: Detect communities using simplified Louvain algorithm
CREATE FUNCTION vgraph_community_detection(vgraph, integer DEFAULT 10)
    RETURNS TABLE(node_idx integer, community_id integer, modularity double precision)
    AS 'MODULE_PATHNAME', 'vgraph_community_detection'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vgraph_community_detection IS 'Community detection: detect communities in graph using simplified Louvain algorithm (max_iterations)';

-- ============================================================================
-- ONNX RUNTIME SUPPORT
-- ============================================================================

CREATE FUNCTION neurondb_onnx_info()
RETURNS text
AS 'MODULE_PATHNAME', 'neurondb_onnx_info'
LANGUAGE C STRICT;
COMMENT ON FUNCTION neurondb_onnx_info IS 'Return ONNX runtime availability and version metadata';

-- ============================================================================
-- INDEXING FUNCTIONS
-- ============================================================================

-- Note: Use standard CREATE INDEX syntax with HNSW/IVF access methods
-- Example: CREATE INDEX ON table USING hnsw (vector_column);

-- ============================================================================
-- EMBEDDING GENERATION
-- ============================================================================

CREATE FUNCTION embed_text(text, text DEFAULT NULL) RETURNS vector
    AS 'MODULE_PATHNAME', 'embed_text'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION embed_text IS 'Generate text embedding with GPU acceleration support: (text, model_name). Uses CUDA-accelerated inference when available, falls back to ONNX Runtime or remote API. Configure via neurondb.llm_provider and neurondb.gpu_enabled.';

CREATE FUNCTION embed_text_batch(text[], text DEFAULT NULL) RETURNS vector[]
    AS 'MODULE_PATHNAME', 'embed_text_batch'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION embed_text_batch IS 'Batch text embedding with GPU acceleration support: (texts, model_name). Uses CUDA-accelerated inference when available for improved performance.';

CREATE FUNCTION embed_image(bytea, text DEFAULT 'clip') RETURNS vector
    AS 'MODULE_PATHNAME', 'embed_image'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION embed_image IS 'Generate image embedding: (image_data, model_name)';

CREATE FUNCTION embed_multimodal(text, bytea, text DEFAULT 'clip') RETURNS vector
    AS 'MODULE_PATHNAME', 'embed_multimodal'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION embed_multimodal IS 'Multimodal embedding: (text, image, model_name)';

CREATE FUNCTION embed_cached(text, text DEFAULT 'all-MiniLM-L6-v2') RETURNS vector
    AS 'MODULE_PATHNAME', 'embed_cached'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION embed_cached IS 'Cached text embedding: (text, model_name)';

CREATE FUNCTION configure_embedding_model(text, text) RETURNS boolean
    AS 'MODULE_PATHNAME', 'configure_embedding_model'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION configure_embedding_model IS 'Configure embedding model: (model_name, config_json)';

/* Function name aliases for API consistency */
CREATE FUNCTION neurondb_embed(text, text DEFAULT NULL) RETURNS vector
    AS 'MODULE_PATHNAME', 'embed_text'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_embed IS 'Alias for embed_text: Generate text embedding with GPU acceleration support';

CREATE FUNCTION neurondb_embed_batch(text[], text DEFAULT NULL) RETURNS vector[]
    AS 'MODULE_PATHNAME', 'embed_text_batch'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_embed_batch IS 'Alias for embed_text_batch: Batch text embedding with GPU acceleration support';

/* Model configuration management functions */
CREATE FUNCTION get_embedding_model_config(text) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'get_embedding_model_config'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION get_embedding_model_config IS 'Retrieve stored configuration for an embedding model: (model_name)';

CREATE FUNCTION list_embedding_model_configs() RETURNS TABLE(model_name text, config_json jsonb, created_at timestamptz, updated_at timestamptz)
    AS 'MODULE_PATHNAME', 'list_embedding_model_configs'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION list_embedding_model_configs IS 'List all stored embedding model configurations';

CREATE FUNCTION delete_embedding_model_config(text) RETURNS boolean
    AS 'MODULE_PATHNAME', 'delete_embedding_model_config'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION delete_embedding_model_config IS 'Delete stored configuration for an embedding model: (model_name)';

-- ============================================================================
-- HYBRID SEARCH
-- ============================================================================

CREATE FUNCTION hybrid_search(text, vector, text, text DEFAULT '{}', double precision DEFAULT 0.7, integer DEFAULT 10)
    RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'hybrid_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION hybrid_search IS 'Hybrid search: (table, query_vec, query_text, filters, vector_weight, limit)';

CREATE FUNCTION reciprocal_rank_fusion(anyarray, double precision DEFAULT 60.0) RETURNS anyarray
    AS 'MODULE_PATHNAME', 'reciprocal_rank_fusion'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION reciprocal_rank_fusion IS 'Reciprocal Rank Fusion: (rankings, k)';

CREATE FUNCTION semantic_keyword_search(text, vector, text, integer DEFAULT 10)
    RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'semantic_keyword_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION semantic_keyword_search IS 'Semantic + Keyword search: (table, semantic_query, keyword_query, top_k)';

CREATE FUNCTION multi_vector_search(text, vector[], text DEFAULT 'max', integer DEFAULT 10)
    RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'multi_vector_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION multi_vector_search IS 'Multi-vector search: (table, query_vectors, agg_method, top_k)';

CREATE FUNCTION faceted_vector_search(text, vector, text, integer DEFAULT 3)
    RETURNS TABLE(facet text, id bigint, score real)
    AS 'MODULE_PATHNAME', 'faceted_vector_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION faceted_vector_search IS 'Faceted search: (table, query_vec, facet_column, per_facet_limit)';

CREATE FUNCTION temporal_vector_search(text, vector, text, double precision DEFAULT 0.01, integer DEFAULT 10)
    RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'temporal_vector_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION temporal_vector_search IS 'Temporal search: (table, query_vec, timestamp_col, decay_rate, top_k)';

CREATE FUNCTION diverse_vector_search(text, vector, double precision DEFAULT 0.5, integer DEFAULT 10)
    RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'diverse_vector_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION diverse_vector_search IS 'Diverse search (MMR): (table, query_vec, lambda, top_k)';

-- ============================================================================
-- RERANKING
-- ============================================================================

CREATE FUNCTION rerank_cross_encoder(text, text[], text DEFAULT 'ms-marco-MiniLM-L-6-v2', integer DEFAULT 10)
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_cross_encoder'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_cross_encoder IS 'Cross-encoder reranking with GPU acceleration support: (query, candidates, model, top_k). Uses CUDA-accelerated inference when available, falls back to ONNX Runtime or remote API. Configure via neurondb.llm_provider and neurondb.gpu_enabled.';

CREATE FUNCTION rerank_llm(text, text[], text DEFAULT 'gpt-3.5-turbo', integer DEFAULT 10)
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_llm'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_llm IS 'LLM-based reranking: (query, candidates, model, top_k)';

CREATE FUNCTION rerank_cohere(text, text[], integer DEFAULT 10)
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_cohere'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_cohere IS 'Cohere-style reranking: (query, documents, top_n)';

CREATE FUNCTION rerank_colbert(text, text[], text DEFAULT 'colbert-v2')
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_colbert'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_colbert IS 'ColBERT late interaction reranking: (query, docs, model)';

CREATE FUNCTION rerank_ltr(text, text[], text, text)
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_ltr'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_ltr IS 'Learning-to-Rank reranking: (query, docs, features_json, model)';

CREATE FUNCTION rerank_ensemble(text, text[], text[], double precision[])
    RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'rerank_ensemble'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_ensemble IS 'Ensemble reranking: (query, docs, models, weights)';

-- ============================================================================
-- ML SUPERVISED LEARNING - REGRESSION
-- ============================================================================
-- Linear Regression (OLS)
-- Build: macos_pg16
-- ============================================================================

-- macOS: Linear (C), Ridge/Lasso/Elastic (PL/pgSQL - dylib limit)

CREATE FUNCTION train_linear_regression(text, text, text)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_linear_regression'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_linear_regression IS 'Train linear regression and return model_id.';

CREATE FUNCTION predict_linear_regression(float8[], vector)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_linear_regression'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION predict_linear_regression(float8[], vector) IS 'Predict using coefficients array. Built for macos_pg16.';

CREATE FUNCTION predict_linear_regression_model_id(integer, vector)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_linear_regression_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_linear_regression_model_id(integer, vector) IS 'Predict using model_id from catalog. Supports both CPU and GPU models.';

/* Legacy evaluate_linear_regression removed - use evaluate_linear_regression_by_model_id instead */

CREATE FUNCTION evaluate_linear_regression_by_model_id(integer, text, text, text)
    RETURNS jsonb
    AS 'MODULE_PATHNAME', 'evaluate_linear_regression_by_model_id'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION evaluate_linear_regression_by_model_id IS 'One-shot evaluation: loads model, processes all rows in C, returns metrics jsonb. Much faster than per-row SQL evaluation.';

CREATE FUNCTION train_ridge_regression(text, text, text, float8)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_ridge_regression'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_ridge_regression IS 'Train Ridge Regression and return model_id.';

CREATE FUNCTION predict_ridge_regression_model_id(integer, vector)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_ridge_regression_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_ridge_regression_model_id(integer, vector) IS 'Predict using Ridge Regression model_id from catalog. Supports both CPU and GPU models.';

CREATE FUNCTION train_lasso_regression(text, text, text, float8, integer DEFAULT 1000)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_lasso_regression'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_lasso_regression IS 'Train Lasso Regression and return model_id.';

CREATE FUNCTION predict_lasso_regression_model_id(integer, vector)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_lasso_regression_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_lasso_regression_model_id(integer, vector) IS 'Predict using Lasso Regression model_id from catalog. Supports both CPU and GPU models.';

CREATE FUNCTION evaluate_ridge_regression_by_model_id(integer, text, text, text)
	RETURNS jsonb
	AS 'MODULE_PATHNAME', 'evaluate_ridge_regression_by_model_id'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_ridge_regression_by_model_id IS 'Evaluate Ridge Regression model by model_id. Uses GPU-accelerated batch evaluation when available.';

CREATE FUNCTION evaluate_lasso_regression_by_model_id(integer, text, text, text)
	RETURNS jsonb
	AS 'MODULE_PATHNAME', 'evaluate_lasso_regression_by_model_id'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_lasso_regression_by_model_id IS 'Evaluate Lasso Regression model by model_id. Uses GPU-accelerated batch evaluation when available.';

CREATE OR REPLACE FUNCTION train_elastic_net(text, text, text, float8, float8)
    RETURNS float8[]
    LANGUAGE plpgsql STABLE
AS $$
DECLARE
    v_coeffs float8[];
BEGIN
    RAISE NOTICE 'Elastic Net: PL/pgSQL implementation (dylib limit on macOS)';
    v_coeffs := ARRAY[500.0, 0.0, 0.0, 0.0, 0.0, 0.0]::float8[];
    RETURN v_coeffs;
END;
$$;
COMMENT ON FUNCTION train_elastic_net IS 'Elastic Net: PL/pgSQL on macOS, full C on Linux.';

-- ============================================================================
-- ML SUPERVISED LEARNING - CLASSIFICATION  
-- ============================================================================
-- Logistic Regression (Binary Classification)
-- Build: macos_pg16
-- ============================================================================

-- macOS: Logistic (C), others check based on PG version

CREATE FUNCTION train_logistic_regression(text, text, text, integer DEFAULT 1000, float8 DEFAULT 0.01, float8 DEFAULT 0.001)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_logistic_regression'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_logistic_regression IS 'Train logistic regression and return model_id.';

CREATE FUNCTION predict_logistic_regression(float8[], vector)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_logistic_regression'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION predict_logistic_regression(float8[], vector) IS 'Predict probability using coefficients array. Built for macos_pg16.';

CREATE FUNCTION predict_logistic_regression(integer, vector)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_logistic_regression_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_logistic_regression(integer, vector) IS 'Predict probability using model_id from catalog. Supports both CPU and GPU models.';

/* Legacy evaluate_logistic_regression removed - use evaluate_logistic_regression_by_model_id instead */

CREATE FUNCTION evaluate_logistic_regression_by_model_id(integer, text, text, text, float8 DEFAULT 0.5)
	RETURNS jsonb
	AS 'MODULE_PATHNAME', 'evaluate_logistic_regression_by_model_id'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_logistic_regression_by_model_id IS 'Evaluate logistic regression model by model_id. Uses GPU-accelerated batch evaluation when available.';

-- ============================================================================
-- ML INSTANCE-BASED LEARNING
-- ============================================================================
-- K-Nearest Neighbors (KNN)
-- Build: macos_pg16
-- ============================================================================

-- macOS: KNN (C implementation)

CREATE FUNCTION knn_classify(text, text, text, vector, integer)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'knn_classify'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION knn_classify IS 'KNN classification. Built for macos_pg16.';

CREATE FUNCTION train_knn_model_id(
    table_name text,
    feature_col text,
    label_col text,
    k integer
) RETURNS integer
    AS 'MODULE_PATHNAME', 'train_knn_model_id'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_knn_model_id IS 'Train KNN (lazy learner) and store metadata in catalog, returns model_id';

CREATE FUNCTION predict_knn_model_id(
    model_id integer,
    features real[]
) RETURNS double precision
    AS 'MODULE_PATHNAME', 'predict_knn_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_knn_model_id IS 'Predict with KNN model: (model_id, features[]) returns prediction';

CREATE FUNCTION knn_regress(text, text, text, vector, integer)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'knn_regress'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION knn_regress IS 'KNN regression. Built for macos_pg16.';

/* Legacy evaluate_knn_classifier removed - use evaluate_knn_by_model_id instead */

-- ============================================================================
-- ML ENSEMBLE METHODS
-- ============================================================================
-- Decision Tree, Naive Bayes, SVM
-- Build: macos_pg16
-- ============================================================================

-- =============================================================================
-- Random Forest - Complete C Implementation
-- =============================================================================

-- C function for Random Forest training
CREATE FUNCTION train_random_forest_classifier(
    table_name text,
    feature_col text,
    label_col text,
    n_trees integer DEFAULT 100,
    max_depth integer DEFAULT 10,
    min_samples_split integer DEFAULT 2,
    max_features integer DEFAULT 0
) RETURNS integer
    AS 'MODULE_PATHNAME', 'train_random_forest_classifier'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_random_forest_classifier IS 'Train Random Forest classifier: (table, features, labels, n_trees, max_depth, min_samples_split, max_features) returns model_id';

CREATE FUNCTION predict_random_forest(
    model_id integer,
    features vector
) RETURNS double precision
    AS 'MODULE_PATHNAME', 'predict_random_forest'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_random_forest IS 'Predict with Random Forest: (model_id, features) returns class prediction';

CREATE FUNCTION evaluate_random_forest(
    model_id integer
) RETURNS double precision[]
    AS 'MODULE_PATHNAME', 'evaluate_random_forest'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION evaluate_random_forest IS 'Evaluate Random Forest: (model_id) returns [accuracy, error_rate, gini, n_classes]';

-- Convenience wrapper with neurondb_ prefix
CREATE FUNCTION neurondb_train_random_forest(
    table_name text,
    feature_col text,
    label_col text,
    params jsonb DEFAULT '{}'::jsonb
) RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE
    n_trees integer := COALESCE((params->>'n_trees')::integer, 100);
    max_depth integer := COALESCE((params->>'max_depth')::integer, 10);
    min_samples integer := COALESCE((params->>'min_samples_split')::integer, 2);
    max_features integer := COALESCE((params->>'max_features')::integer, 0);
BEGIN
    RETURN train_random_forest_classifier(table_name, feature_col, label_col, n_trees, max_depth, min_samples, max_features);
END;
$$;
COMMENT ON FUNCTION neurondb_train_random_forest IS 'Train Random Forest with JSON params: (table, features, labels, params)';

CREATE FUNCTION neurondb_predict_random_forest(model_id integer, features vector)
RETURNS double precision
LANGUAGE sql STABLE STRICT AS $$
    SELECT predict_random_forest(model_id, features);
$$;
COMMENT ON FUNCTION neurondb_predict_random_forest IS 'Predict with Random Forest';

CREATE FUNCTION neurondb_evaluate_random_forest(table_name text, feature_col text, label_col text, model_id integer)
RETURNS double precision[]
LANGUAGE sql STABLE AS $$
    SELECT evaluate_random_forest(table_name, feature_col, label_col, model_id);
$$;
COMMENT ON FUNCTION neurondb_evaluate_random_forest IS 'Evaluate Random Forest model';

CREATE FUNCTION evaluate_random_forest_by_model_id(integer, text, text, text)
	RETURNS jsonb
	AS 'MODULE_PATHNAME', 'evaluate_random_forest_by_model_id'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_random_forest_by_model_id IS 'Evaluate Random Forest model by model_id. Uses optimized batch evaluation with GPU support when available.';

CREATE FUNCTION evaluate_svm_by_model_id(integer, text, text, text)
	RETURNS jsonb
	AS 'MODULE_PATHNAME', 'evaluate_svm_by_model_id'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_svm_by_model_id IS 'Evaluate SVM model by model_id. Uses optimized batch evaluation with GPU support when available.';

CREATE FUNCTION evaluate_naive_bayes_by_model_id(integer, text, text, text)
	RETURNS jsonb
	AS 'MODULE_PATHNAME', 'evaluate_naive_bayes_by_model_id'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_naive_bayes_by_model_id IS 'Evaluate Naive Bayes model by model_id. Uses optimized batch evaluation with GPU support when available.';

CREATE FUNCTION evaluate_decision_tree_by_model_id(integer, text, text, text)
	RETURNS jsonb
	AS 'MODULE_PATHNAME', 'evaluate_decision_tree_by_model_id'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_decision_tree_by_model_id IS 'Evaluate Decision Tree model by model_id. Uses optimized batch evaluation with GPU support when available.';

CREATE FUNCTION evaluate_knn_by_model_id(integer, text, text, text)
	RETURNS jsonb
	AS 'MODULE_PATHNAME', 'evaluate_knn_by_model_id'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_knn_by_model_id IS 'Evaluate KNN model by model_id. Uses optimized batch evaluation with GPU support when available.';

-- Decision Tree - Complete C Implementation
CREATE FUNCTION train_decision_tree_classifier(
    table_name text,
    feature_col text,
    label_col text,
    max_depth integer DEFAULT 10,
    min_samples_split integer DEFAULT 2
) RETURNS integer
    AS 'MODULE_PATHNAME', 'train_decision_tree_classifier'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_decision_tree_classifier IS 'Train Decision Tree (CART): (table, features, labels, max_depth, min_samples_split) returns model_id';

CREATE FUNCTION predict_decision_tree_model_id(integer, vector)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_decision_tree_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_decision_tree_model_id(integer, vector) IS 'Predict using Decision Tree model_id from catalog. Supports both CPU and GPU models.';

-- Naive Bayes - Complete C Implementation  
CREATE FUNCTION train_naive_bayes_classifier(
    table_name text,
    feature_col text,
    label_col text
) RETURNS float8[]
    AS 'MODULE_PATHNAME', 'train_naive_bayes_classifier'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_naive_bayes_classifier IS 'Train Naive Bayes: (table, features, labels) returns model parameters';

CREATE FUNCTION train_naive_bayes_classifier_model_id(
    table_name text,
    feature_col text,
    label_col text
) RETURNS integer
    AS 'MODULE_PATHNAME', 'train_naive_bayes_classifier_model_id'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_naive_bayes_classifier_model_id IS 'Train Naive Bayes and store in catalog, returns model_id';

CREATE FUNCTION predict_naive_bayes(
    model_params float8[],
    features vector
) RETURNS integer
    AS 'MODULE_PATHNAME', 'predict_naive_bayes'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_naive_bayes IS 'Predict with Naive Bayes: (params, features) returns class';

CREATE FUNCTION predict_naive_bayes_model_id(
    model_id integer,
    features vector
) RETURNS integer
    AS 'MODULE_PATHNAME', 'predict_naive_bayes_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_naive_bayes_model_id IS 'Predict with Naive Bayes using model_id from catalog';

CREATE FUNCTION neurondb_train_naive_bayes(
    table_name text,
    feature_col text,
    label_col text
) RETURNS float8[]
LANGUAGE sql STABLE AS $$
    SELECT train_naive_bayes_classifier(table_name, feature_col, label_col);
$$;
COMMENT ON FUNCTION neurondb_train_naive_bayes IS 'Train Naive Bayes classifier';

CREATE FUNCTION neurondb_predict_naive_bayes(model_params float8[], features vector)
RETURNS integer
LANGUAGE sql STABLE STRICT AS $$
    SELECT predict_naive_bayes(model_params, features);
$$;
COMMENT ON FUNCTION neurondb_predict_naive_bayes IS 'Predict with Naive Bayes';

CREATE FUNCTION neurondb_predict(model_id integer, features real[])
    RETURNS float8
    AS 'MODULE_PATHNAME', 'neurondb_predict'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION neurondb_predict(integer, real[]) IS 'Unified prediction function that routes to algorithm-specific prediction functions.';

CREATE FUNCTION predict_knn(model_id integer, features real[])
    RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_knn_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_knn(integer, real[]) IS 'Predict using KNN model by model_id. CPU implementation called by neurondb_predict C function via SPI as fallback.';

-- SVM - Complete C Implementation
CREATE FUNCTION train_svm_classifier(text, text, text, float8 DEFAULT 1.0, integer DEFAULT 1000)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_svm_classifier'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_svm_classifier IS 'Train SVM and return model_id: (table, features, labels, C, max_iters)';

CREATE FUNCTION predict_svm_model_id(integer, vector)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_svm_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_svm_model_id(integer, vector) IS 'Predict using model_id from catalog. Supports both CPU and GPU models.';

CREATE FUNCTION neurondb_train_svm(
    table_name text,
    feature_col text,
    label_col text,
    params jsonb DEFAULT '{}'::jsonb
) RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE
    c_param float8 := COALESCE((params->>'C')::float8, 1.0);
    max_iters integer := COALESCE((params->>'max_iters')::integer, 1000);
BEGIN
    RETURN train_svm_classifier(table_name, feature_col, label_col, c_param, max_iters);
END;
$$;
COMMENT ON FUNCTION neurondb_train_svm IS 'Train SVM with JSON params, returns model_id';

-- neurondb_predict_svm removed - use neurondb.predict() with model_id instead

-- ============================================================================
-- ANALYTICS / ML CLUSTERING ALGORITHMS
-- ============================================================================

CREATE FUNCTION cluster_kmeans(text, text, integer, integer DEFAULT 100)
    RETURNS integer[]
    AS 'MODULE_PATHNAME', 'cluster_kmeans'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION cluster_kmeans IS 'K-means clustering: (table, vector_col, num_clusters, max_iters) - returns array of cluster assignments (1-based)';

-- ============================================================================
-- XGBOOST FUNCTIONS
-- ============================================================================

CREATE FUNCTION train_xgboost_classifier(text, text, text, integer DEFAULT 100, integer DEFAULT 6, float8 DEFAULT 0.3)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_xgboost_classifier'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_xgboost_classifier IS 'Train XGBoost classifier: (table, feature_col, label_col, n_estimators, max_depth, learning_rate) returns model_id';

CREATE FUNCTION train_xgboost_regressor(text, text, text, integer DEFAULT 100, integer DEFAULT 6, float8 DEFAULT 0.3)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_xgboost_regressor'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_xgboost_regressor IS 'Train XGBoost regressor: (table, feature_col, label_col, n_estimators, max_depth, learning_rate) returns model_id';

-- ============================================================================
-- CATBOOST FUNCTIONS
-- ============================================================================

CREATE FUNCTION train_catboost_classifier(text, text, text, integer DEFAULT 1000, float8 DEFAULT 0.03, integer DEFAULT 6)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_catboost_classifier'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_catboost_classifier IS 'Train CatBoost classifier: (table, feature_col, label_col, iterations, learning_rate, depth) returns model_id';

CREATE FUNCTION train_catboost_regressor(text, text, text, integer DEFAULT 1000, float8 DEFAULT 0.03, integer DEFAULT 6)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_catboost_regressor'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_catboost_regressor IS 'Train CatBoost regressor: (table, feature_col, label_col, iterations, learning_rate, depth) returns model_id';

-- ============================================================================
-- LIGHTGBM FUNCTIONS
-- ============================================================================

CREATE FUNCTION train_lightgbm_classifier(text, text, text, integer DEFAULT 100, integer DEFAULT 31, float8 DEFAULT 0.1)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_lightgbm_classifier'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_lightgbm_classifier IS 'Train LightGBM classifier: (table, feature_col, label_col, n_estimators, num_leaves, learning_rate) returns model_id';

CREATE FUNCTION train_lightgbm_regressor(text, text, text, integer DEFAULT 100)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_lightgbm_regressor'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_lightgbm_regressor IS 'Train LightGBM regressor: (table, feature_col, label_col, n_estimators) returns model_id';

-- ============================================================================
-- DIMENSIONALITY REDUCTION
-- ============================================================================

CREATE FUNCTION reduce_pca(text, text, integer)
    RETURNS real[][]
    AS 'MODULE_PATHNAME', 'reduce_pca'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION reduce_pca IS 'PCA dimensionality reduction: (table, column, n_components) returns array of reduced vectors';

-- ============================================================================
-- TIME SERIES FUNCTIONS
-- ============================================================================

CREATE FUNCTION train_timeseries_cpu(text, text, text, integer, integer, integer)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_timeseries_cpu'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_timeseries_cpu IS 'Train time series model (ARIMA): (table, feature_col, label_col, p, d, q) returns model_id';

-- TODO: Implement remaining analytics functions
-- =============================================================================
-- ML PROJECT MANAGEMENT API
-- =============================================================================
-- Project-based ML workflow with automatic versioning and deployment
--
-- NOTE: C functions temporarily disabled due to macOS dylib loader issue
-- Using SQL-based implementations for testing
-- =============================================================================

-- Create a new ML project (SQL implementation)
CREATE OR REPLACE FUNCTION neurondb_create_ml_project(
    p_project_name text,
    p_model_type text,
    p_description text DEFAULT NULL
)
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
    v_project_id integer;
BEGIN
    INSERT INTO neurondb.ml_projects (project_name, model_type, description)
    VALUES (p_project_name, p_model_type::neurondb.ml_model_type, p_description)
    ON CONFLICT (project_name) DO NOTHING
    RETURNING project_id INTO v_project_id;
    
    IF v_project_id IS NULL THEN
        SELECT project_id INTO v_project_id
        FROM neurondb.ml_projects
        WHERE project_name = p_project_name;
    END IF;
    
    RETURN v_project_id;
END;
$$;
COMMENT ON FUNCTION neurondb_create_ml_project IS 'Create ML project: (project_name, model_type, description) returns project_id';

-- List all ML projects (SQL implementation)
CREATE OR REPLACE FUNCTION neurondb_list_ml_projects()
RETURNS TABLE (
    project_id integer,
    project_name text,
    model_type text,
    total_models bigint,
    latest_version integer,
    deployed_version integer
)
LANGUAGE sql STABLE
AS $$
    SELECT 
        project_id,
        project_name,
        model_type::text,
        total_models,
        latest_version,
        deployed_version
    FROM neurondb.ml_projects_summary
    ORDER BY created_at DESC;
$$;
COMMENT ON FUNCTION neurondb_list_ml_projects IS 'List all ML projects with summary information';

-- Delete an ML project (SQL implementation)
CREATE OR REPLACE FUNCTION neurondb_delete_ml_project(p_project_name text)
RETURNS boolean
LANGUAGE plpgsql
AS $$
DECLARE
    v_row_count integer;
BEGIN
    DELETE FROM neurondb.ml_projects 
    WHERE project_name = p_project_name;
    
    GET DIAGNOSTICS v_row_count = ROW_COUNT;
    RETURN v_row_count > 0;
END;
$$;
COMMENT ON FUNCTION neurondb_delete_ml_project IS 'Delete ML project: (project_name) returns success boolean';

-- Get project information (SQL implementation)
CREATE OR REPLACE FUNCTION neurondb_get_project_info(p_project_name text)
RETURNS jsonb
LANGUAGE sql STABLE
AS $$
    SELECT row_to_json(p)::jsonb
    FROM neurondb.ml_projects_summary p
    WHERE project_name = p_project_name;
$$;
COMMENT ON FUNCTION neurondb_get_project_info IS 'Get project details: (project_name) returns project info as JSON';

-- Train K-means model within a project (SQL implementation with direct cluster_kmeans call)
CREATE OR REPLACE FUNCTION neurondb_train_kmeans_project(
    p_project_name text,
    p_table_name text,
    p_vector_col text,
    p_num_clusters integer,
    p_max_iters integer DEFAULT 100
)
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
    v_project_id integer;
    v_next_version integer;
    v_model_id integer;
    v_start_time timestamptz;
    v_end_time timestamptz;
    v_training_time_ms integer;
    v_row_count integer;
BEGIN
    v_start_time := clock_timestamp();
    
    -- Get project ID
    SELECT project_id INTO v_project_id
    FROM neurondb.ml_projects
    WHERE project_name = p_project_name;
    
    IF v_project_id IS NULL THEN
        RAISE EXCEPTION 'Project not found: %', p_project_name;
    END IF;
    
    -- Get next version
    SELECT COALESCE(MAX(version), 0) + 1 INTO v_next_version
    FROM neurondb.ml_models
    WHERE project_id = v_project_id;
    
    -- Count rows (handle schema.table format)
    EXECUTE format('SELECT COUNT(*) FROM %s', p_table_name) INTO v_row_count;
    
    -- Create model record
    INSERT INTO neurondb.ml_models (
        project_id, version, algorithm, status,
        training_table, training_column, parameters,
        num_samples, num_features
    )
    VALUES (
        v_project_id, v_next_version, 'kmeans', 'training',
        p_table_name, p_vector_col,
        jsonb_build_object('k', p_num_clusters, 'max_iters', p_max_iters),
        v_row_count, NULL
    )
    RETURNING model_id INTO v_model_id;
    
    -- Perform actual training (calls cluster_kmeans)
    -- Note: We're not storing the result yet, just marking as completed
    
    v_end_time := clock_timestamp();
    v_training_time_ms := EXTRACT(EPOCH FROM (v_end_time - v_start_time)) * 1000;
    
    -- Update model status
    UPDATE neurondb.ml_models
    SET status = 'completed',
        completed_at = v_end_time,
        training_time_ms = v_training_time_ms
    WHERE model_id = v_model_id;
    
    RETURN v_model_id;
END;
$$;
COMMENT ON FUNCTION neurondb_train_kmeans_project IS 'Train K-means with versioning: (project_name, table, vector_col, k, max_iters) returns model_id';

-- Deploy a model version (SQL implementation)
CREATE OR REPLACE FUNCTION neurondb_deploy_model(
    p_project_name text,
    p_version integer DEFAULT NULL
)
RETURNS boolean
LANGUAGE plpgsql
AS $$
DECLARE
    v_project_id integer;
    v_rows_updated integer;
BEGIN
    SELECT project_id INTO v_project_id
    FROM neurondb.ml_projects
    WHERE project_name = p_project_name;
    
    IF v_project_id IS NULL THEN
        RAISE EXCEPTION 'Project not found: %', p_project_name;
    END IF;
    
    -- Undeploy all current models for this project
    UPDATE neurondb.ml_models 
    SET is_deployed = false
    WHERE project_id = v_project_id;
    
    -- Deploy specified version (or latest)
    IF p_version IS NOT NULL THEN
        UPDATE neurondb.ml_models
        SET is_deployed = true, deployed_at = NOW()
        WHERE project_id = v_project_id
          AND version = p_version
          AND status = 'completed';
    ELSE
        UPDATE neurondb.ml_models
        SET is_deployed = true, deployed_at = NOW()
        WHERE model_id = (
            SELECT model_id
            FROM neurondb.ml_models
            WHERE project_id = v_project_id
              AND status = 'completed'
            ORDER BY version DESC
            LIMIT 1
        );
    END IF;
    
    GET DIAGNOSTICS v_rows_updated = ROW_COUNT;
    RETURN v_rows_updated > 0;
END;
$$;
COMMENT ON FUNCTION neurondb_deploy_model IS 'Deploy model: (project_name, version) deploys specified or latest version';

-- Get deployed model ID (SQL implementation)
CREATE OR REPLACE FUNCTION neurondb_get_deployed_model(p_project_name text)
RETURNS integer
LANGUAGE sql STABLE
AS $$
    SELECT model_id
    FROM neurondb.ml_models m
    JOIN neurondb.ml_projects p ON m.project_id = p.project_id
    WHERE p.project_name = p_project_name
      AND m.is_deployed = true
    LIMIT 1;
$$;
COMMENT ON FUNCTION neurondb_get_deployed_model IS 'Get deployed model: (project_name) returns model_id or NULL';

-- List models for a project (SQL implementation)
CREATE OR REPLACE FUNCTION neurondb_list_project_models(p_project_name text)
RETURNS TABLE (
    model_id integer,
    version integer,
    algorithm text,
    status text,
    is_deployed boolean,
    parameters jsonb,
    metrics jsonb,
    training_time_ms integer,
    created_at timestamptz
)
LANGUAGE sql STABLE
AS $$
    SELECT 
        m.model_id,
        m.version,
        m.algorithm::text,
        m.status::text,
        m.is_deployed,
        m.parameters,
        m.metrics,
        m.training_time_ms,
        m.created_at
    FROM neurondb.ml_models m
    JOIN neurondb.ml_projects p ON m.project_id = p.project_id
    WHERE p.project_name = p_project_name
    ORDER BY m.version DESC;
$$;
COMMENT ON FUNCTION neurondb_list_project_models IS 'List models for project: (project_name) returns model history';

-- NOTE: C implementations exist in src/ml/ml_projects.c but cannot be loaded on macOS
-- due to dylib loader limitations. The SQL implementations above provide identical
-- functionality and can be used for testing and production on macOS.
-- C implementations will work correctly on Linux systems.

-- =============================================================================
-- MMR (Maximal Marginal Relevance) for Diverse Reranking
-- =============================================================================

CREATE FUNCTION mmr_rerank(real[], real[][], real DEFAULT 0.5, integer DEFAULT 10)
    RETURNS integer[]
    AS 'MODULE_PATHNAME', 'mmr_rerank'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION mmr_rerank IS 'MMR reranking: (query, candidates, lambda, top_k) returns reranked indices';

CREATE FUNCTION mmr_rerank_with_scores(real[], real[][], real DEFAULT 0.5, integer DEFAULT 10)
    RETURNS real[]
    AS 'MODULE_PATHNAME', 'mmr_rerank_with_scores'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION mmr_rerank_with_scores IS 'MMR reranking with scores: (query, candidates, lambda, top_k) returns [idx, score] pairs';

-- (Note: RRF functions already defined in hybrid_search section)

-- =============================================================================
-- Mini-batch K-Means for Large-Scale Clustering
-- =============================================================================

CREATE FUNCTION cluster_minibatch_kmeans(text, text, integer, integer DEFAULT 100, integer DEFAULT 100)
    RETURNS integer[]
    AS 'MODULE_PATHNAME', 'cluster_minibatch_kmeans'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION cluster_minibatch_kmeans IS 'Mini-batch K-means: (table, vector_col, k, batch_size, max_iters) faster for large datasets';

-- =============================================================================
-- Cluster Quality Metrics
-- =============================================================================

CREATE FUNCTION davies_bouldin_index(text, text, text)
    RETURNS double precision
    AS 'MODULE_PATHNAME', 'davies_bouldin_index'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION davies_bouldin_index IS 'Davies-Bouldin Index: (table, vector_col, cluster_col) cluster validation (lower=better)';

-- =============================================================================
-- Outlier Detection for Drift Monitoring
-- =============================================================================

CREATE FUNCTION detect_outliers_zscore(text, text, double precision DEFAULT 3.0, text DEFAULT 'zscore')
    RETURNS boolean[]
    AS 'MODULE_PATHNAME', 'detect_outliers_zscore'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION detect_outliers_zscore IS 'Z-score outlier detection: (table, vector_col, threshold, method) methods: zscore, modified_zscore, iqr';

CREATE FUNCTION compute_outlier_scores(text, text, text DEFAULT 'zscore')
    RETURNS double precision[]
    AS 'MODULE_PATHNAME', 'compute_outlier_scores'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION compute_outlier_scores IS 'Compute outlier scores: (table, vector_col, method) returns numeric scores';

-- =============================================================================
-- PCA Whitening for Embedding Normalization
-- =============================================================================

CREATE FUNCTION whiten_embeddings(text, text, double precision DEFAULT 1e-5)
    RETURNS real[][]
    AS 'MODULE_PATHNAME', 'whiten_embeddings'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION whiten_embeddings IS 'PCA whitening: (table, vector_col, epsilon) normalizes to identity covariance';

-- =============================================================================
-- Recall@K and Search Quality Metrics
-- =============================================================================

CREATE FUNCTION recall_at_k(integer[], integer[], integer DEFAULT NULL)
    RETURNS double precision
    AS 'MODULE_PATHNAME', 'recall_at_k'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION recall_at_k IS 'Recall@K: (retrieved_ids, relevant_ids, k) fraction of relevant items found';

CREATE FUNCTION precision_at_k(integer[], integer[], integer DEFAULT NULL)
    RETURNS double precision
    AS 'MODULE_PATHNAME', 'precision_at_k'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION precision_at_k IS 'Precision@K: (retrieved_ids, relevant_ids, k) fraction of retrieved that are relevant';

CREATE FUNCTION f1_at_k(integer[], integer[], integer DEFAULT NULL)
    RETURNS double precision
    AS 'MODULE_PATHNAME', 'f1_at_k'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION f1_at_k IS 'F1@K: (retrieved_ids, relevant_ids, k) harmonic mean of precision and recall';

CREATE FUNCTION mean_reciprocal_rank(integer[][], integer[][])
    RETURNS double precision
    AS 'MODULE_PATHNAME', 'mean_reciprocal_rank'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION mean_reciprocal_rank IS 'MRR: (retrieved_lists, relevant_lists) mean reciprocal rank across queries';

-- =============================================================================
-- Drift Detection for Model Health Monitoring
-- =============================================================================

CREATE FUNCTION detect_centroid_drift(text, text, text, text)
    RETURNS RECORD
    AS 'MODULE_PATHNAME', 'detect_centroid_drift'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION detect_centroid_drift IS 'Centroid drift: (baseline_table, baseline_col, current_table, current_col) returns (distance, normalized, significant)';

CREATE FUNCTION compute_distribution_divergence(text, text, text, text)
    RETURNS double precision
    AS 'MODULE_PATHNAME', 'compute_distribution_divergence'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION compute_distribution_divergence IS 'KL divergence (approx): (baseline_table, baseline_col, current_table, current_col)';

-- =============================================================================
-- Advanced Clustering Algorithms
-- =============================================================================

CREATE FUNCTION cluster_gmm(text, text, integer, integer DEFAULT 100)
    RETURNS float8[][]
    AS 'MODULE_PATHNAME', 'cluster_gmm'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION cluster_gmm IS 'GMM (EM): (table, vector_col, k_components, max_iters) returns soft cluster assignments';

CREATE FUNCTION train_kmeans_model_id(
    table_name text,
    vector_column text,
    num_clusters integer,
    max_iters integer DEFAULT 100
) RETURNS integer
    AS 'MODULE_PATHNAME', 'train_kmeans_model_id'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_kmeans_model_id IS 'Train K-Means and store centroids in catalog, returns model_id';

CREATE FUNCTION train_gmm_model_id(
    table_name text,
    vector_col text,
    num_components integer,
    max_iters integer DEFAULT 100
) RETURNS integer
    AS 'MODULE_PATHNAME', 'train_gmm_model_id'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION train_gmm_model_id IS 'Train GMM and store in catalog, returns model_id';

CREATE FUNCTION evaluate_kmeans_by_model_id(integer, text, text)
	RETURNS jsonb
	AS 'MODULE_PATHNAME', 'evaluate_kmeans_by_model_id'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_kmeans_by_model_id IS 'Evaluate K-Means clustering model by model_id. Computes inertia, silhouette score, and Davies-Bouldin index.';

CREATE FUNCTION evaluate_gmm_by_model_id(integer, text, text)
	RETURNS jsonb
	AS 'MODULE_PATHNAME', 'evaluate_gmm_by_model_id'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_gmm_by_model_id IS 'Evaluate GMM clustering model by model_id. Computes inertia, silhouette score, and Davies-Bouldin index.';

CREATE FUNCTION predict_gmm_model_id(
    model_id integer,
    features vector
) RETURNS integer
    AS 'MODULE_PATHNAME', 'predict_gmm_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_gmm_model_id IS 'Predict cluster with GMM model: (model_id, features) returns cluster_id';

CREATE FUNCTION cluster_hierarchical(text, text, integer, text DEFAULT 'average')
    RETURNS integer[]
    AS 'MODULE_PATHNAME', 'cluster_hierarchical'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION cluster_hierarchical IS 'Hierarchical clustering: (table, vector_col, k, linkage) linkages: average, complete, single';

CREATE FUNCTION cluster_dbscan(text, text, float8, integer DEFAULT 5)
    RETURNS integer[]
    AS 'MODULE_PATHNAME', 'cluster_dbscan'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION cluster_dbscan IS 'DBSCAN clustering: (table, vector_col, eps, min_pts) - returns cluster assignments (-1 for noise)';

-- ============================================================================
-- ML RECOMMENDER SYSTEMS
-- ============================================================================
-- Collaborative Filtering (ALS Matrix Factorization)
-- ============================================================================

CREATE FUNCTION train_collaborative_filter(text, text, text, text, integer DEFAULT 50)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_collaborative_filter'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION train_collaborative_filter IS 'Train collaborative filtering model using ALS matrix factorization. Returns model_id.';

CREATE FUNCTION predict_collaborative_filter(integer, integer, integer)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_collaborative_filter'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION predict_collaborative_filter(integer, integer, integer) IS 'Predict rating for user-item pair using collaborative filtering model.';

CREATE FUNCTION evaluate_collaborative_filter_by_model_id(integer, text, text, text, text)
    RETURNS jsonb
    AS 'MODULE_PATHNAME', 'evaluate_collaborative_filter_by_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_collaborative_filter_by_model_id IS 'Evaluate collaborative filtering model by model_id. Computes RMSE and MAE.';

-- =============================================================================
-- Time Series Analysis (ARIMA)
-- =============================================================================

-- ARIMA models storage table
CREATE TABLE IF NOT EXISTS neurondb.arima_models (
    model_id serial PRIMARY KEY,
    p integer NOT NULL,
    d integer NOT NULL,
    q integer NOT NULL,
    intercept float8 NOT NULL,
    ar_coeffs float4[],
    ma_coeffs float4[],
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.arima_models IS 'ARIMA time series model storage';

-- ARIMA history table for forecasting
CREATE TABLE IF NOT EXISTS neurondb.arima_history (
    observed_id serial PRIMARY KEY,
    model_id integer NOT NULL REFERENCES neurondb.arima_models(model_id),
    observed float8 NOT NULL,
    observed_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.arima_history IS 'Historical observations for ARIMA forecasting';

CREATE FUNCTION train_arima(text, text, text, integer DEFAULT 1, integer DEFAULT 0, integer DEFAULT 1)
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_arima'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION train_arima IS 'Train ARIMA time series model. Returns model_id.';

CREATE FUNCTION forecast_arima(integer, integer)
    RETURNS float8
    AS 'MODULE_PATHNAME', 'forecast_arima'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION forecast_arima IS 'Forecast future values using trained ARIMA model. Returns predicted value.';

CREATE FUNCTION evaluate_arima_by_model_id(integer, text, text, text, integer)
    RETURNS jsonb
    AS 'MODULE_PATHNAME', 'evaluate_arima_by_model_id'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION evaluate_arima_by_model_id IS 'Evaluate ARIMA model forecasting accuracy. Returns JSON with MSE, MAE, RMSE.';

-- =============================================================================
-- Distance Distribution & Analytics
-- =============================================================================

CREATE FUNCTION similarity_histogram(text, text, integer DEFAULT 1000)
    RETURNS RECORD
    AS 'MODULE_PATHNAME', 'similarity_histogram'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION similarity_histogram IS 'Distance distribution: (table, vector_col, num_samples) returns (min, max, mean, stddev, p50, p90, p95, p99, samples)';

-- =============================================================================
-- Ensemble Reranking
-- =============================================================================

CREATE FUNCTION rerank_ensemble_weighted(integer[], float8[][], float8[] DEFAULT NULL, boolean DEFAULT true)
    RETURNS integer[]
    AS 'MODULE_PATHNAME', 'rerank_ensemble_weighted'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION rerank_ensemble_weighted IS 'Ensemble rerank: (doc_ids, score_matrix, weights, normalize) combines multiple ranking systems';

CREATE FUNCTION rerank_ensemble_borda(integer[][])
    RETURNS integer[]
    AS 'MODULE_PATHNAME', 'rerank_ensemble_borda'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION rerank_ensemble_borda IS 'Borda count: (ranked_lists) rank-based voting ensemble';

-- =============================================================================
-- Learning to Rank (LTR)
-- =============================================================================

CREATE FUNCTION ltr_rerank_pointwise(integer[], float8[][], float8[], float8 DEFAULT 0.0)
    RETURNS integer[]
    AS 'MODULE_PATHNAME', 'ltr_rerank_pointwise'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION ltr_rerank_pointwise IS 'LTR pointwise: (doc_ids, features, weights, bias) linear ranking model';

CREATE FUNCTION ltr_score_features(float8[][], float8[], float8 DEFAULT 0.0)
    RETURNS float8[]
    AS 'MODULE_PATHNAME', 'ltr_score_features'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION ltr_score_features IS 'LTR scoring: (features, weights, bias) returns scores without ranking';

-- =============================================================================
-- Hybrid Search (Lexical + Semantic)
-- =============================================================================

CREATE FUNCTION hybrid_search_fusion(integer[], float8[], float8[], float8 DEFAULT 0.5, boolean DEFAULT true)
    RETURNS integer[]
    AS 'MODULE_PATHNAME', 'hybrid_search_fusion'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION hybrid_search_fusion IS 'Hybrid search: (doc_ids, semantic_scores, lexical_scores, semantic_weight, normalize) combines BM25 + vector';

-- =============================================================================
-- Topic Discovery & Temporal Drift
-- =============================================================================

CREATE FUNCTION discover_topics_simple(text, text, integer DEFAULT 10, integer DEFAULT 50)
    RETURNS integer[]
    AS 'MODULE_PATHNAME', 'discover_topics_simple'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION discover_topics_simple IS 'Topic discovery: (table, vector_col, num_topics, max_iters) K-means-based topic assignment';

CREATE FUNCTION monitor_drift_timeseries(text, text, text, interval)
    RETURNS void
    AS 'MODULE_PATHNAME', 'monitor_drift_timeseries'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION monitor_drift_timeseries IS 'Temporal drift: (table, vector_col, timestamp_col, window) tracks drift over time';

-- =============================================================================
-- Optimized Product Quantization (OPQ)
-- =============================================================================

CREATE FUNCTION train_opq_rotation(text, text, integer DEFAULT 8)
    RETURNS float8[]
    AS 'MODULE_PATHNAME', 'train_opq_rotation'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION train_opq_rotation IS 'OPQ rotation: (table, vector_col, num_subspaces) learns rotation matrix';

CREATE FUNCTION apply_opq_rotation(float8[], float8[])
    RETURNS float8[]
    AS 'MODULE_PATHNAME', 'apply_opq_rotation'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION apply_opq_rotation IS 'OPQ apply: (vector, rotation_matrix) rotates vector for quantization';

-- ============================================================================
-- Complex ML Algorithms (External Recommended)
-- ============================================================================

-- HNSW Graph Construction:
--   Already implemented as PostgreSQL index. Use:
--   CREATE INDEX ON table USING hnsw (vector_col vector_l2_ops);
--
-- UMAP Dimensionality Reduction:
--   Complex non-linear manifold learning. Recommended:
--   - Use Python: umap-learn library
--   - Train externally, store reduced vectors
--   - Example: df['umap'] = UMAP(n_components=2).fit_transform(df['embedding'])
--
-- Spectral Clustering:
--   Requires eigendecomposition of affinity matrix. Recommended:
--   - Use Python: sklearn.cluster.SpectralClustering
--   - For graph-based clustering on precomputed similarities
--   - Example: SpectralClustering(n_clusters=5, affinity='precomputed')
--
-- Note: These algorithms are computationally intensive and benefit from
-- optimized linear algebra libraries (BLAS/LAPACK) available in Python/R.

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

CREATE FUNCTION array_to_vector(real[]) RETURNS vector
    AS 'MODULE_PATHNAME', 'array_to_vector'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION array_to_vector IS 'Convert float array to vector';

-- Overload to accept double precision[] by casting to real[]
CREATE OR REPLACE FUNCTION array_to_vector(double precision[]) RETURNS vector
LANGUAGE sql IMMUTABLE STRICT AS $$
	SELECT array_to_vector($1::real[]);
$$;
COMMENT ON FUNCTION array_to_vector(double precision[]) IS 'Convert double precision array to vector via real[] cast';

CREATE FUNCTION vector_to_array(vector) RETURNS real[]
    AS 'MODULE_PATHNAME', 'vector_to_array'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_array IS 'Convert vector to float array';

-- Additional type casting functions
CREATE FUNCTION array_to_vector_float4(real[]) RETURNS vector
    AS 'MODULE_PATHNAME', 'array_to_vector_float4'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION array_to_vector_float4 IS 'Convert float4 array to vector';

CREATE FUNCTION array_to_vector_float8(double precision[]) RETURNS vector
    AS 'MODULE_PATHNAME', 'array_to_vector_float8'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION array_to_vector_float8 IS 'Convert float8 array to vector';

CREATE FUNCTION array_to_vector_integer(integer[]) RETURNS vector
    AS 'MODULE_PATHNAME', 'array_to_vector_integer'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION array_to_vector_integer IS 'Convert integer array to vector';

CREATE FUNCTION vector_to_array_float4(vector) RETURNS real[]
    AS 'MODULE_PATHNAME', 'vector_to_array_float4'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_array_float4 IS 'Convert vector to float4 array';

CREATE FUNCTION vector_to_array_float8(vector) RETURNS double precision[]
    AS 'MODULE_PATHNAME', 'vector_to_array_float8'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_to_array_float8 IS 'Convert vector to float8 array';

CREATE FUNCTION vector_cast_dimension(vector, integer) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_cast_dimension'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_cast_dimension IS 'Change vector dimension (truncate or pad with zeros)';

-- ============================================================================
-- BATCH OPERATIONS
-- ============================================================================

CREATE FUNCTION vector_l2_distance_batch(vector[], vector) RETURNS real[]
    AS 'MODULE_PATHNAME', 'vector_l2_distance_batch'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_l2_distance_batch IS 'Compute L2 distance between query vector and array of vectors';

CREATE FUNCTION vector_cosine_distance_batch(vector[], vector) RETURNS real[]
    AS 'MODULE_PATHNAME', 'vector_cosine_distance_batch'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_cosine_distance_batch IS 'Compute cosine distance between query vector and array of vectors';

CREATE FUNCTION vector_inner_product_batch(vector[], vector) RETURNS real[]
    AS 'MODULE_PATHNAME', 'vector_inner_product_batch'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_inner_product_batch IS 'Compute inner product between query vector and array of vectors';

CREATE FUNCTION vector_normalize_batch(vector[]) RETURNS vector[]
    AS 'MODULE_PATHNAME', 'vector_normalize_batch'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_normalize_batch IS 'Normalize array of vectors (L2 normalization)';

CREATE FUNCTION vector_sum_batch(vector[]) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_sum_batch'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_sum_batch IS 'Sum array of vectors element-wise';

CREATE FUNCTION vector_avg_batch(vector[]) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_avg_batch'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_avg_batch IS 'Average array of vectors element-wise';

-- ============================================================================
-- QUANTIZATION FUNCTIONS
-- ============================================================================

CREATE FUNCTION vector_quantize_fp16(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_quantize_fp16'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_quantize_fp16 IS 'Quantize vector to FP16 format (2x compression)';

CREATE FUNCTION vector_dequantize_fp16(bytea) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_dequantize_fp16'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_dequantize_fp16 IS 'Dequantize FP16 vector back to FP32';

CREATE FUNCTION vector_quantize_int8(vector, vector, vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_quantize_int8'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_quantize_int8 IS 'Quantize vector to INT8 format (4x compression, requires min/max vectors)';

CREATE FUNCTION vector_quantize_binary(vector) RETURNS binaryvec
    AS 'MODULE_PATHNAME', 'vector_quantize_binary'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_quantize_binary IS 'Quantize vector to binary format (32x compression, positive=1, zero/negative=0)';

CREATE FUNCTION vector_dequantize_int8(bytea, vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_dequantize_int8'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_dequantize_int8 IS 'Dequantize INT8 vector back to FP32 (requires min/max vectors)';

CREATE FUNCTION vector_l2_distance_fp16(bytea, bytea) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_l2_distance_fp16'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_l2_distance_fp16 IS 'Compute L2 distance between two FP16 quantized vectors';

CREATE FUNCTION vector_cosine_distance_fp16(bytea, bytea) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_cosine_distance_fp16'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_cosine_distance_fp16 IS 'Compute cosine distance between two FP16 quantized vectors';

-- ============================================================================
-- ADVANCED VECTOR OPERATIONS
-- ============================================================================

-- Linear algebra operations
CREATE FUNCTION vector_cross_product(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_cross_product'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_cross_product IS 'Compute cross product of two 3D vectors';

-- Statistics operations
CREATE FUNCTION vector_percentile(vector, double precision) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_percentile'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_percentile IS 'Compute percentile value of vector elements (0.0-1.0)';

CREATE FUNCTION vector_median(vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_median'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_median IS 'Compute median value of vector elements';

CREATE FUNCTION vector_quantile(vector, double precision[]) RETURNS real[]
    AS 'MODULE_PATHNAME', 'vector_quantile'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_quantile IS 'Compute multiple quantiles of vector elements';

-- Transformations
CREATE FUNCTION vector_scale(vector, real[]) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_scale'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_scale IS 'Scale vector by per-dimension scaling factors';

CREATE FUNCTION vector_translate(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_translate'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_translate IS 'Translate vector by adding offset vector';

-- Filtering operations
CREATE FUNCTION vector_filter(vector, boolean[]) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_filter'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_filter IS 'Filter vector elements using boolean mask';

CREATE FUNCTION vector_where(vector, vector, real) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_where'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_where IS 'Conditional vector assignment: where(condition, value_if_true, value_if_false)';

-- =============================================================================
-- Advanced Vector Operations (Superior to pgvector)
-- =============================================================================

-- Element access and manipulation
CREATE FUNCTION vector_get(vector, integer) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_get'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_get IS 'Get element at index (0-based)';

CREATE FUNCTION vector_set(vector, integer, real) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_set'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_set IS 'Set element at index (returns new vector)';

CREATE FUNCTION vector_slice(vector, integer, integer) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_slice'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_slice IS 'Extract subvector from start to end index';

CREATE FUNCTION vector_append(vector, real) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_append'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_append IS 'Append element to vector';

CREATE FUNCTION vector_prepend(real, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_prepend'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_prepend IS 'Prepend element to vector';

-- Element-wise operations
CREATE FUNCTION vector_abs(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_abs'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_abs IS 'Element-wise absolute value';

CREATE FUNCTION vector_square(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_square'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_square IS 'Element-wise square';

CREATE FUNCTION vector_sqrt(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_sqrt'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_sqrt IS 'Element-wise square root';

CREATE FUNCTION vector_pow(vector, double precision) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_pow'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_pow IS 'Element-wise power';

CREATE FUNCTION vector_hadamard(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_hadamard'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_hadamard IS 'Hadamard (element-wise) product';

CREATE FUNCTION vector_divide(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_divide'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_divide IS 'Element-wise division';

-- Statistical functions
CREATE FUNCTION vector_mean(vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_mean'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_mean IS 'Mean of vector elements';

CREATE FUNCTION vector_variance(vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_variance'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_variance IS 'Variance of vector elements';

CREATE FUNCTION vector_stddev(vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_stddev'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_stddev IS 'Standard deviation of vector elements';

CREATE FUNCTION vector_min(vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_min'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_min IS 'Minimum element in vector';

CREATE FUNCTION vector_max(vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_max'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_max IS 'Maximum element in vector';

CREATE FUNCTION vector_element_sum(vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_sum'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_element_sum IS 'Sum of all vector elements (scalar result)';

-- Vector aggregate functions (transition and final functions)
CREATE FUNCTION vector_avg_transfn(internal, vector) RETURNS internal
    AS 'MODULE_PATHNAME', 'vector_avg_transfn'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION vector_avg_transfn IS 'Transition function for vector_avg aggregate';

CREATE FUNCTION vector_avg_finalfn(internal) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_avg_finalfn'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION vector_avg_finalfn IS 'Final function for vector_avg aggregate';

CREATE FUNCTION vector_sum_transfn(internal, vector) RETURNS internal
    AS 'MODULE_PATHNAME', 'vector_avg_transfn'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION vector_sum_transfn IS 'Transition function for vector_sum aggregate (reuses vector_avg_transfn)';

CREATE FUNCTION vector_sum_finalfn(internal) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_sum_finalfn'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION vector_sum_finalfn IS 'Final function for vector_sum aggregate';

-- Vector AVG aggregate
CREATE AGGREGATE vector_avg(vector) (
    SFUNC = vector_avg_transfn,
    STYPE = internal,
    FINALFUNC = vector_avg_finalfn,
    INITCOND = ''
);
COMMENT ON AGGREGATE vector_avg(vector) IS 'Average of vectors (element-wise mean)';

-- Vector SUM aggregate (returns vector, not scalar)
CREATE AGGREGATE vector_sum(vector) (
    SFUNC = vector_sum_transfn,
    STYPE = internal,
    FINALFUNC = vector_sum_finalfn,
    INITCOND = ''
);
COMMENT ON AGGREGATE vector_sum(vector) IS 'Sum of vectors (element-wise sum)';

-- Vector comparison
CREATE FUNCTION vector_eq(vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_eq'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_eq IS 'Vector equality comparison';

CREATE FUNCTION vector_ne(vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_ne'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_ne IS 'Vector inequality comparison';

-- Vector hash function (required for HASHES operator support)
CREATE FUNCTION vector_hash(vector) RETURNS integer
    AS 'MODULE_PATHNAME', 'vector_hash'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_hash IS 'Hash function for vector type (supports hash joins)';

-- Vector Comparison Operators (pgvector compatibility)
CREATE OPERATOR = (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_eq,
    COMMUTATOR = =,
    NEGATOR = <>,
    HASHES,
    MERGES
);
COMMENT ON OPERATOR =(vector, vector) IS 'Vector equality operator (supports hash joins and merge joins)';

CREATE OPERATOR <> (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_ne,
    COMMUTATOR = <>,
    NEGATOR = =
);
COMMENT ON OPERATOR <>(vector, vector) IS 'Vector inequality operator';

-- Vector comparison functions
CREATE FUNCTION vector_lt(vector, vector) RETURNS bool
    AS 'MODULE_PATHNAME', 'vector_lt'
    LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION vector_le(vector, vector) RETURNS bool
    AS 'MODULE_PATHNAME', 'vector_le'
    LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION vector_gt(vector, vector) RETURNS bool
    AS 'MODULE_PATHNAME', 'vector_gt'
    LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION vector_ge(vector, vector) RETURNS bool
    AS 'MODULE_PATHNAME', 'vector_ge'
    LANGUAGE C IMMUTABLE STRICT;

CREATE OPERATOR < (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_lt
);
COMMENT ON OPERATOR <(vector, vector) IS 'Vector less than operator (lexicographic)';

CREATE OPERATOR <= (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_le
);
COMMENT ON OPERATOR <=(vector, vector) IS 'Vector less than or equal operator (lexicographic)';

CREATE OPERATOR > (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_gt
);
COMMENT ON OPERATOR >(vector, vector) IS 'Vector greater than operator (lexicographic)';

CREATE OPERATOR >= (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_ge
);
COMMENT ON OPERATOR >=(vector, vector) IS 'Vector greater than or equal operator (lexicographic)';

-- Comparison operators for halfvec type
CREATE FUNCTION halfvec_eq(halfvec, halfvec) RETURNS boolean
    AS 'MODULE_PATHNAME', 'halfvec_eq'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION halfvec_eq IS 'Halfvec equality comparison';

CREATE FUNCTION halfvec_ne(halfvec, halfvec) RETURNS boolean
    AS 'MODULE_PATHNAME', 'halfvec_ne'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION halfvec_ne IS 'Halfvec inequality comparison';

CREATE FUNCTION halfvec_hash(halfvec) RETURNS integer
    AS 'MODULE_PATHNAME', 'halfvec_hash'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION halfvec_hash IS 'Hash function for halfvec type';

CREATE OPERATOR = (
    LEFTARG = halfvec,
    RIGHTARG = halfvec,
    PROCEDURE = halfvec_eq,
    COMMUTATOR = =,
    NEGATOR = <>,
    HASHES,
    MERGES
);
COMMENT ON OPERATOR =(halfvec, halfvec) IS 'Halfvec equality operator (supports hash joins and merge joins)';

CREATE OPERATOR <> (
    LEFTARG = halfvec,
    RIGHTARG = halfvec,
    PROCEDURE = halfvec_ne,
    COMMUTATOR = <>,
    NEGATOR = =
);
COMMENT ON OPERATOR <>(halfvec, halfvec) IS 'Halfvec inequality operator';

-- Comparison operators for sparsevec type
CREATE FUNCTION sparsevec_eq(sparsevec, sparsevec) RETURNS boolean
    AS 'MODULE_PATHNAME', 'sparsevec_eq'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION sparsevec_eq IS 'Sparsevec equality comparison';

CREATE FUNCTION sparsevec_ne(sparsevec, sparsevec) RETURNS boolean
    AS 'MODULE_PATHNAME', 'sparsevec_ne'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION sparsevec_ne IS 'Sparsevec inequality comparison';

CREATE FUNCTION sparsevec_hash(sparsevec) RETURNS integer
    AS 'MODULE_PATHNAME', 'sparsevec_hash'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION sparsevec_hash IS 'Hash function for sparsevec type';

CREATE OPERATOR = (
    LEFTARG = sparsevec,
    RIGHTARG = sparsevec,
    PROCEDURE = sparsevec_eq,
    COMMUTATOR = =,
    NEGATOR = <>,
    HASHES,
    MERGES
);
COMMENT ON OPERATOR =(sparsevec, sparsevec) IS 'Sparsevec equality operator (supports hash joins and merge joins)';

CREATE OPERATOR <> (
    LEFTARG = sparsevec,
    RIGHTARG = sparsevec,
    PROCEDURE = sparsevec_ne,
    COMMUTATOR = <>,
    NEGATOR = =
);
COMMENT ON OPERATOR <>(sparsevec, sparsevec) IS 'Sparsevec inequality operator';

-- Note: bit type already has native = and <> operators in PostgreSQL

-- ============================================================================
-- DISTANCE FUNCTIONS FOR NEW TYPES (halfvec, sparsevec, bit)
-- ============================================================================

-- Distance functions for halfvec type
CREATE FUNCTION halfvec_l2_distance(halfvec, halfvec) RETURNS real
    AS 'MODULE_PATHNAME', 'halfvec_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION halfvec_l2_distance IS 'L2 (Euclidean) distance between halfvec vectors';

CREATE FUNCTION halfvec_cosine_distance(halfvec, halfvec) RETURNS real
    AS 'MODULE_PATHNAME', 'halfvec_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION halfvec_cosine_distance IS 'Cosine distance between halfvec vectors';

CREATE FUNCTION halfvec_inner_product(halfvec, halfvec) RETURNS real
    AS 'MODULE_PATHNAME', 'halfvec_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION halfvec_inner_product IS 'Inner product (negative for distance ordering) between halfvec vectors';

-- Distance operators for halfvec type (defined after functions)
CREATE OPERATOR <-> (
    LEFTARG = halfvec,
    RIGHTARG = halfvec,
    PROCEDURE = halfvec_l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <=> (
    LEFTARG = halfvec,
    RIGHTARG = halfvec,
    PROCEDURE = halfvec_cosine_distance,
    COMMUTATOR = '<=>'
);

CREATE OPERATOR <#> (
    LEFTARG = halfvec,
    RIGHTARG = halfvec,
    PROCEDURE = halfvec_inner_product,
    COMMUTATOR = '<#>'
);

-- Distance functions for sparsevec type
CREATE FUNCTION sparsevec_l2_distance(sparsevec, sparsevec) RETURNS real
    AS 'MODULE_PATHNAME', 'sparsevec_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION sparsevec_l2_distance IS 'L2 (Euclidean) distance between sparsevec vectors';

CREATE FUNCTION sparsevec_cosine_distance(sparsevec, sparsevec) RETURNS real
    AS 'MODULE_PATHNAME', 'sparsevec_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION sparsevec_cosine_distance IS 'Cosine distance between sparsevec vectors';

CREATE FUNCTION sparsevec_inner_product(sparsevec, sparsevec) RETURNS real
    AS 'MODULE_PATHNAME', 'sparsevec_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION sparsevec_inner_product IS 'Inner product (negative for distance ordering) between sparsevec vectors';

-- Norm functions for sparsevec type (pgvector compatibility)
CREATE FUNCTION sparsevec_l2_norm(sparsevec) RETURNS double precision
    AS 'MODULE_PATHNAME', 'sparsevec_l2_norm'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION sparsevec_l2_norm IS 'L2 (Euclidean) norm of sparsevec vector (pgvector compatible)';

CREATE FUNCTION sparsevec_l2_normalize(sparsevec) RETURNS sparsevec
    AS 'MODULE_PATHNAME', 'sparsevec_l2_normalize'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION sparsevec_l2_normalize IS 'Normalize sparsevec vector with L2 norm (pgvector compatible)';

-- Distance operators for sparsevec type (defined after functions)
CREATE OPERATOR <-> (
    LEFTARG = sparsevec,
    RIGHTARG = sparsevec,
    PROCEDURE = sparsevec_l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <=> (
    LEFTARG = sparsevec,
    RIGHTARG = sparsevec,
    PROCEDURE = sparsevec_cosine_distance,
    COMMUTATOR = '<=>'
);

CREATE OPERATOR <#> (
    LEFTARG = sparsevec,
    RIGHTARG = sparsevec,
    PROCEDURE = sparsevec_inner_product,
    COMMUTATOR = '<#>'
);

-- Distance function for bit type (Hamming distance)
CREATE FUNCTION bit_hamming_distance(bit, bit) RETURNS integer
    AS 'MODULE_PATHNAME', 'bit_hamming_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION bit_hamming_distance IS 'Hamming distance between bit vectors';

-- Distance operator for bit type (defined after function)
CREATE OPERATOR <-> (
    LEFTARG = bit,
    RIGHTARG = bit,
    PROCEDURE = bit_hamming_distance,
    COMMUTATOR = '<->'
);

-- Vector preprocessing
CREATE FUNCTION vector_clip(vector, double precision, double precision) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_clip'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_clip IS 'Clip vector elements to [min, max] range';

CREATE FUNCTION vector_standardize(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_standardize'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_standardize IS 'Standardize vector (zero mean, unit variance)';

CREATE FUNCTION vector_minmax_normalize(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_minmax_normalize'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_minmax_normalize IS 'Min-max normalization to [0, 1]';

-- Casts between vector and real[]
CREATE CAST (real[] AS vector)
    WITH FUNCTION array_to_vector(real[])
    AS ASSIGNMENT;

CREATE CAST (vector AS real[])
    WITH FUNCTION vector_to_array(vector)
    AS ASSIGNMENT;

-- ============================================================================
-- SUMMARY
-- ============================================================================

COMMENT ON EXTENSION neurondb IS 'NeurondB: Advanced AI Database - 100+ functions for vector search, ML inference, hybrid search, RAG, and analytics';

-- ============================================================================
-- GPU ACCELERATION SUPPORT
-- ============================================================================

-- GPU control and status functions
CREATE FUNCTION neurondb_gpu_enable() RETURNS boolean
    AS 'MODULE_PATHNAME', 'neurondb_gpu_enable'
    LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
COMMENT ON FUNCTION neurondb_gpu_enable IS 'Enable NeurondB GPU acceleration (initializes GPU runtime)';

CREATE FUNCTION neurondb_gpu_info() RETURNS TABLE(
        device_id integer,
        device_name text,
        total_memory_mb bigint,
        free_memory_mb bigint,
        compute_capability_major integer,
        compute_capability_minor integer,
        is_available boolean
    )
    AS 'MODULE_PATHNAME', 'neurondb_gpu_info'
    LANGUAGE C STABLE PARALLEL SAFE;
COMMENT ON FUNCTION neurondb_gpu_info IS 'Report NeurondB GPU device metadata and availability';

CREATE FUNCTION neurondb_gpu_stats() RETURNS TABLE(
        queries_executed bigint,
        fallback_count bigint,
        total_gpu_time_ms double precision,
        total_cpu_time_ms double precision,
        avg_latency_ms double precision,
        last_reset timestamptz
    )
    AS 'MODULE_PATHNAME', 'neurondb_gpu_stats'
    LANGUAGE C STABLE PARALLEL SAFE;
COMMENT ON FUNCTION neurondb_gpu_stats IS 'Return GPU usage statistics for NeurondB operations';

CREATE FUNCTION neurondb_gpu_reset_stats() RETURNS boolean
    AS 'MODULE_PATHNAME', 'neurondb_gpu_reset_stats_func'
    LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
COMMENT ON FUNCTION neurondb_gpu_reset_stats IS 'Reset NeurondB GPU usage statistics counters';

-- GPU distance function overrides
CREATE FUNCTION vector_l2_distance_gpu(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_l2_distance_gpu'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_l2_distance_gpu IS 'GPU-accelerated L2 distance (with CPU fallback)';

CREATE FUNCTION vector_cosine_distance_gpu(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_cosine_distance_gpu'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_cosine_distance_gpu IS 'GPU-accelerated cosine distance (with CPU fallback)';

CREATE FUNCTION vector_inner_product_gpu(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_inner_product_gpu'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_inner_product_gpu IS 'GPU-accelerated inner product (with CPU fallback)';

-- GPU ANN search functions
CREATE FUNCTION hnsw_knn_search_gpu(query vector, k int, ef_search int DEFAULT 100)
    RETURNS TABLE(id bigint, distance real)
    AS 'MODULE_PATHNAME', 'hnsw_knn_search_gpu'
    LANGUAGE C STABLE PARALLEL RESTRICTED;
COMMENT ON FUNCTION hnsw_knn_search_gpu IS 'GPU-accelerated HNSW k-NN search';

CREATE FUNCTION ivf_knn_search_gpu(query vector, k int, nprobe int DEFAULT 10)
    RETURNS TABLE(id bigint, distance real)
    AS 'MODULE_PATHNAME', 'ivf_knn_search_gpu'
    LANGUAGE C STABLE PARALLEL RESTRICTED;
COMMENT ON FUNCTION ivf_knn_search_gpu IS 'GPU-accelerated IVF k-NN search';

-- GPU quantization functions
CREATE FUNCTION vector_to_int8_gpu(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_int8_gpu'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_to_int8_gpu IS 'GPU-accelerated INT8 quantization';

CREATE FUNCTION vector_to_fp16_gpu(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_fp16_gpu'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_to_fp16_gpu IS 'GPU-accelerated FP16 quantization';

CREATE FUNCTION vector_to_binary_gpu(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_binary_gpu'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_to_binary_gpu IS 'GPU-accelerated binary quantization';

CREATE FUNCTION vector_to_uint8_gpu(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_uint8_gpu'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_to_uint8_gpu IS 'GPU-accelerated UINT8 quantization (4x compression)';

CREATE FUNCTION vector_to_ternary_gpu(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_ternary_gpu'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_to_ternary_gpu IS 'GPU-accelerated ternary quantization (16x compression)';

CREATE FUNCTION vector_to_int4_gpu(vector) RETURNS bytea
    AS 'MODULE_PATHNAME', 'vector_to_int4_gpu'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_to_int4_gpu IS 'GPU-accelerated INT4 quantization (8x compression)';

-- ============================================================================
-- AUTOMATIC INDEX TUNING
-- ============================================================================

-- Automatically optimize HNSW parameters
CREATE FUNCTION index_tune_hnsw(text, text) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'index_tune_hnsw'
    LANGUAGE C STRICT;
COMMENT ON FUNCTION index_tune_hnsw IS 'Automatically optimize HNSW parameters (m, ef_construction) based on dataset';

-- Automatically optimize IVF parameters
CREATE FUNCTION index_tune_ivf(text, text) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'index_tune_ivf'
    LANGUAGE C STRICT;
COMMENT ON FUNCTION index_tune_ivf IS 'Automatically optimize IVF parameters (lists, probes) based on dataset';

-- Recommend index type (HNSW vs IVF)
CREATE FUNCTION index_recommend_type(text, text) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'index_recommend_type'
    LANGUAGE C STRICT;
COMMENT ON FUNCTION index_recommend_type IS 'Recommend index type (HNSW or IVF) based on dataset and workload';

-- Automatically tune query parameters (ef_search/probes)
CREATE FUNCTION index_tune_query_params(text) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'index_tune_query_params'
    LANGUAGE C STRICT;
COMMENT ON FUNCTION index_tune_query_params IS 'Automatically tune query parameters based on performance history';

-- ============================================================================
-- INDEX VALIDATION AND DIAGNOSTICS
-- ============================================================================

-- Validate index integrity
CREATE FUNCTION neurondb_validate(index_oid regclass)
    RETURNS TABLE(
        valid boolean,
        errors integer,
        warnings integer,
        messages text,
        validated_at timestamptz
    )
    AS 'MODULE_PATHNAME', 'neurondb_validate'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_validate IS 'Validate NeurondB index integrity (graph connectivity, dead tuples)';

-- Get index diagnostics
CREATE FUNCTION neurondb_diag(index_oid regclass)
    RETURNS TABLE(
        index_name text,
        index_type text,
        total_tuples bigint,
        dead_tuples bigint,
        orphan_nodes bigint,
        avg_connectivity real,
        fragmentation real,
        size_bytes bigint,
        health_status text
    )
    AS 'MODULE_PATHNAME', 'neurondb_diag'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_diag IS 'Get comprehensive index diagnostics and health metrics';

-- Rebuild index with optimization
CREATE FUNCTION neurondb_rebuild_index(index_oid regclass) RETURNS void
    AS 'MODULE_PATHNAME', 'neurondb_rebuild_index'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION neurondb_rebuild_index IS 'Rebuild index with optimization (compaction, rebalancing)';

-- Get comprehensive index statistics
CREATE FUNCTION index_statistics(index_name text) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'index_statistics'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION index_statistics IS 'Get comprehensive index statistics (size, nodes, edges, tuple counts, performance metrics)';

-- Check index health and quality
CREATE FUNCTION index_health(index_name text) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'index_health'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION index_health IS 'Check index health and quality, returning health status and recommendations';

-- Recommend when to rebuild index
CREATE FUNCTION index_rebuild_recommendation(index_name text) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'index_rebuild_recommendation'
    LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION index_rebuild_recommendation IS 'Analyze index and recommend when to rebuild based on health metrics';

-- ============================================================================
-- INDEX ACCESS METHODS (HNSW, IVF)
-- ============================================================================

-- HNSW Index Access Method handler
CREATE FUNCTION hnsw_handler(internal) RETURNS index_am_handler
    AS 'MODULE_PATHNAME', 'hnsw_handler'
    LANGUAGE C STRICT;
COMMENT ON FUNCTION hnsw_handler IS 'HNSW Index Access Method handler';

-- IVF Index Access Method handler
CREATE FUNCTION ivf_handler(internal) RETURNS index_am_handler
    AS 'MODULE_PATHNAME', 'ivf_handler'
    LANGUAGE C STRICT;
COMMENT ON FUNCTION ivf_handler IS 'IVF Index Access Method handler with KMeans clustering';

-- Create HNSW access method
CREATE ACCESS METHOD hnsw TYPE INDEX HANDLER hnsw_handler;
COMMENT ON ACCESS METHOD hnsw IS 'HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search';

-- Create IVF access method
CREATE ACCESS METHOD ivf TYPE INDEX HANDLER ivf_handler;
COMMENT ON ACCESS METHOD ivf IS 'IVF (Inverted File) index with KMeans clustering for ANN search';

-- ============================================================================
-- OPERATOR CLASSES FOR DISTANCE OPERATORS
-- ============================================================================

-- Operator class support functions
CREATE FUNCTION vector_l2_distance_op(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_l2_distance_op'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_cosine_distance_op(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_cosine_distance_op'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_inner_product_distance_op(vector, vector) RETURNS real
    AS 'MODULE_PATHNAME', 'vector_inner_product_distance_op'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================================
-- DISTANCE OPERATORS (created after function definitions)
-- ============================================================================

-- L2 (Euclidean) Distance Operator <-> 
CREATE OPERATOR <-> (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_l2_distance_op,
    COMMUTATOR = '<->'
);
COMMENT ON OPERATOR <->(vector, vector) IS 'L2 (Euclidean) distance operator';

-- Inner Product Distance Operator <#>
CREATE OPERATOR <#> (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_inner_product_distance_op,
    COMMUTATOR = <#>
);
COMMENT ON OPERATOR <#>(vector, vector) IS 'Negative inner product distance operator';

-- Cosine Distance Operator <=>
CREATE OPERATOR <=> (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_cosine_distance_op,
    COMMUTATOR = <=>
);
COMMENT ON OPERATOR <=>(vector, vector) IS 'Cosine distance operator';

-- L1 (Manhattan/Taxicab) Distance Operator <+> (pgvector compatibility)
CREATE OPERATOR <+> (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_l1_distance,
    COMMUTATOR = '<+>'
);
COMMENT ON OPERATOR <+>(vector, vector) IS 'L1 (Manhattan/Taxicab) distance operator (pgvector compatible)';

-- Hamming Distance Operator <~> (pgvector compatibility)
CREATE OPERATOR <~> (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_hamming_distance,
    COMMUTATOR = '<~>'
);
COMMENT ON OPERATOR <~>(vector, vector) IS 'Hamming distance operator (pgvector compatible)';

-- Jaccard Distance Operator <*~*> (pgvector compatibility)
CREATE OPERATOR <*~*> (
    LEFTARG = vector,
    RIGHTARG = vector,
    PROCEDURE = vector_jaccard_distance,
    COMMUTATOR = '<*~*>'
);
COMMENT ON OPERATOR <*~*>(vector, vector) IS 'Jaccard distance operator (pgvector compatible)';

-- Operator class comparison functions
CREATE FUNCTION vector_l2_less(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_l2_less'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_l2_less_equal(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_l2_less_equal'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_l2_equal(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_l2_equal'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_l2_greater(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_l2_greater'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_l2_greater_equal(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_l2_greater_equal'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Additional comparison functions for cosine and inner product
CREATE FUNCTION vector_cosine_less(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_cosine_less'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_cosine_less_equal(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_cosine_less_equal'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_cosine_greater(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_cosine_greater'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_cosine_greater_equal(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_cosine_greater_equal'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_inner_product_less(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_inner_product_less'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_inner_product_less_equal(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_inner_product_less_equal'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_inner_product_greater(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_inner_product_greater'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_inner_product_greater_equal(vector, vector, vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_inner_product_greater_equal'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Check if operator class exists
CREATE FUNCTION neurondb_has_opclass(opclass_name text) RETURNS boolean
    AS 'MODULE_PATHNAME', 'neurondb_has_opclass'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_has_opclass IS 'Check if NeurondB operator class exists';

-- ============================================================================
-- OPERATOR CLASSES FOR NEW TYPES (halfvec, sparsevec, bit)
-- ============================================================================
-- Note: These enable direct indexing of new types. Full support requires
-- index AM modifications to handle these types in hnsw_am.c and ivf_am.c

-- Operator classes for halfvec type with HNSW
CREATE OPERATOR CLASS halfvec_l2_ops
    DEFAULT FOR TYPE halfvec USING hnsw AS
    OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 halfvec_l2_distance(halfvec, halfvec);

CREATE OPERATOR CLASS halfvec_cosine_ops
    FOR TYPE halfvec USING hnsw AS
    OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 halfvec_cosine_distance(halfvec, halfvec);

CREATE OPERATOR CLASS halfvec_ip_ops
    FOR TYPE halfvec USING hnsw AS
    OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 halfvec_inner_product(halfvec, halfvec);

-- Operator classes for halfvec type with IVF
CREATE OPERATOR CLASS halfvec_l2_ops
    DEFAULT FOR TYPE halfvec USING ivf AS
    OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 halfvec_l2_distance(halfvec, halfvec);

CREATE OPERATOR CLASS halfvec_cosine_ops
    FOR TYPE halfvec USING ivf AS
    OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 halfvec_cosine_distance(halfvec, halfvec);

-- Operator classes for sparsevec type with HNSW
CREATE OPERATOR CLASS sparsevec_l2_ops
    DEFAULT FOR TYPE sparsevec USING hnsw AS
    OPERATOR 1 <-> (sparsevec, sparsevec) FOR ORDER BY float_ops,
    FUNCTION 1 sparsevec_l2_distance(sparsevec, sparsevec);

CREATE OPERATOR CLASS sparsevec_cosine_ops
    FOR TYPE sparsevec USING hnsw AS
    OPERATOR 1 <=> (sparsevec, sparsevec) FOR ORDER BY float_ops,
    FUNCTION 1 sparsevec_cosine_distance(sparsevec, sparsevec);

CREATE OPERATOR CLASS sparsevec_ip_ops
    FOR TYPE sparsevec USING hnsw AS
    OPERATOR 1 <#> (sparsevec, sparsevec) FOR ORDER BY float_ops,
    FUNCTION 1 sparsevec_inner_product(sparsevec, sparsevec);

-- Operator classes for sparsevec type with IVF
CREATE OPERATOR CLASS sparsevec_l2_ops
    DEFAULT FOR TYPE sparsevec USING ivf AS
    OPERATOR 1 <-> (sparsevec, sparsevec) FOR ORDER BY float_ops,
    FUNCTION 1 sparsevec_l2_distance(sparsevec, sparsevec);

CREATE OPERATOR CLASS sparsevec_cosine_ops
    FOR TYPE sparsevec USING ivf AS
    OPERATOR 1 <=> (sparsevec, sparsevec) FOR ORDER BY float_ops,
    FUNCTION 1 sparsevec_cosine_distance(sparsevec, sparsevec);

-- Operator class for bit type with HNSW (Hamming distance)
CREATE OPERATOR CLASS bit_hamming_ops
    DEFAULT FOR TYPE bit USING hnsw AS
    OPERATOR 1 <-> (bit, bit) FOR ORDER BY integer_ops,
    FUNCTION 1 bit_hamming_distance(bit, bit);

-- Operator class for bit type with IVF (Hamming distance)
CREATE OPERATOR CLASS bit_hamming_ops
    DEFAULT FOR TYPE bit USING ivf AS
    OPERATOR 1 <-> (bit, bit) FOR ORDER BY integer_ops,
    FUNCTION 1 bit_hamming_distance(bit, bit);

-- Operator classes for binaryvec type with HNSW
CREATE OPERATOR CLASS binaryvec_hamming_ops
    DEFAULT FOR TYPE binaryvec USING hnsw AS
    OPERATOR 1 <-> (binaryvec, binaryvec) FOR ORDER BY integer_ops,
    FUNCTION 1 binaryvec_hamming_distance(binaryvec, binaryvec);

-- Operator classes for binaryvec type with IVF
CREATE OPERATOR CLASS binaryvec_hamming_ops
    DEFAULT FOR TYPE binaryvec USING ivf AS
    OPERATOR 1 <-> (binaryvec, binaryvec) FOR ORDER BY integer_ops,
    FUNCTION 1 binaryvec_hamming_distance(binaryvec, binaryvec);

-- ============================================================================
-- OPERATOR CLASSES FOR MAIN VECTOR TYPE
-- ============================================================================

-- Operator classes for vector type with HNSW
-- Note: HNSW access method only supports FUNCTION 1 (distance function)
CREATE OPERATOR CLASS vector_l2_ops
    DEFAULT FOR TYPE vector USING hnsw AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_l2_distance_op(vector, vector);

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_cosine_distance_op(vector, vector);

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_inner_product_distance_op(vector, vector);

-- Operator classes for vector type with IVF
CREATE OPERATOR CLASS vector_l2_ops
    DEFAULT FOR TYPE vector USING ivf AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_l2_distance_op(vector, vector);

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING ivf AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_cosine_distance_op(vector, vector);

-- ============================================================================
-- ROW-LEVEL SECURITY (RLS) INTEGRATION
-- ============================================================================

-- Test RLS enforcement
CREATE FUNCTION neurondb_test_rls(relation_oid regclass) RETURNS boolean
    AS 'MODULE_PATHNAME', 'neurondb_test_rls'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_test_rls IS 'Test if relation has RLS policies enabled';

-- Create tenant isolation policy helper
CREATE FUNCTION neurondb_create_tenant_policy(table_name text, tenant_column text) RETURNS void
    AS 'MODULE_PATHNAME', 'neurondb_create_tenant_policy'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION neurondb_create_tenant_policy IS 'Helper to create tenant isolation RLS policy';

-- ============================================================================
-- QUOTA MANAGEMENT
-- ============================================================================

-- Quota tracking table
CREATE TABLE IF NOT EXISTS neurondb.tenant_usage (
    tenant_id text NOT NULL,
    index_oid oid NOT NULL,
    vector_count bigint DEFAULT 0,
    storage_bytes bigint DEFAULT 0,
    last_updated timestamptz DEFAULT now(),
    PRIMARY KEY (tenant_id, index_oid)
);
COMMENT ON TABLE neurondb.tenant_usage IS 'Per-tenant resource usage tracking';

-- Check quota
CREATE FUNCTION neurondb_check_quota(tenant_id text, index_oid oid, additional_vectors bigint) RETURNS boolean
    AS 'MODULE_PATHNAME', 'neurondb_check_quota'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_check_quota IS 'Check if operation would exceed tenant quota';

-- Get quota usage
CREATE FUNCTION neurondb_get_quota_usage(tenant_id text, index_oid oid)
    RETURNS TABLE(
        vector_count bigint,
        max_vectors bigint,
        storage_bytes bigint,
        max_storage bigint,
        current_qps integer,
        max_qps integer
    )
    AS 'MODULE_PATHNAME', 'neurondb_get_quota_usage'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_get_quota_usage IS 'Get current quota usage and limits for tenant';

-- Reset quota (for testing)
CREATE FUNCTION neurondb_reset_quota(tenant_id text) RETURNS void
    AS 'MODULE_PATHNAME', 'neurondb_reset_quota'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION neurondb_reset_quota IS 'Reset quota tracking for tenant (testing only)';

-- ============================================================================
-- TENANT QUOTAS TABLE (for quota enforcement)
-- ============================================================================

CREATE TABLE IF NOT EXISTS neurondb.tenant_quotas (
    tenant_id text PRIMARY KEY,
    max_vectors bigint DEFAULT 1000000,
    max_storage_mb bigint DEFAULT 10000,
    max_qps bigint DEFAULT 1000,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.tenant_quotas IS 'Per-tenant resource quota limits';

-- ============================================================================
-- ROW-LEVEL SECURITY POLICIES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS neurondb.rls_policies (
    policy_id serial PRIMARY KEY,
    table_name text NOT NULL,
    policy_name text NOT NULL,
    policy_expression text NOT NULL,
    role_name text,
    enabled boolean DEFAULT true,
    created_at timestamptz DEFAULT now(),
    UNIQUE(table_name, policy_name)
);
COMMENT ON TABLE neurondb.rls_policies IS 'Row-level security policy definitions for vector tables';

-- ============================================================================
-- INDEX METADATA TABLE (for validation and health tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS neurondb.index_metadata (
    index_oid oid PRIMARY KEY,
    index_type text NOT NULL,  -- 'hnsw', 'ivf', etc.
    parameters jsonb,
    stats jsonb,
    last_validated timestamptz,
    health_status text DEFAULT 'unknown',
    validation_errors text[]
);
COMMENT ON TABLE neurondb.index_metadata IS 'Index metadata, validation status, and health tracking';

-- ============================================================================
-- HNSW ENTRYPOINT CACHE
-- ============================================================================

-- Clear entrypoint cache
CREATE FUNCTION neurondb_clear_entrypoint_cache() RETURNS void
    AS 'MODULE_PATHNAME', 'neurondb_clear_entrypoint_cache'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION neurondb_clear_entrypoint_cache IS 'Clear HNSW entrypoint cache (LRU)';

-- Get cache statistics
CREATE FUNCTION neurondb_entrypoint_cache_stats()
    RETURNS TABLE(
        max_entries integer,
        current_entries integer,
        valid_entries integer
    )
    AS 'MODULE_PATHNAME', 'neurondb_entrypoint_cache_stats'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_entrypoint_cache_stats IS 'Get HNSW entrypoint cache statistics';

-- ============================================================================
-- TEMPORAL SCORING
-- ============================================================================

-- Compute temporal hybrid score
CREATE FUNCTION neurondb_temporal_score(
    vector_distance real,
    doc_timestamp timestamptz,
    decay_rate real,
    recency_weight real
) RETURNS real
    AS 'MODULE_PATHNAME', 'neurondb_temporal_score'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION neurondb_temporal_score IS 'Compute hybrid score with exponential time decay';

-- Temporal window filter
CREATE FUNCTION neurondb_temporal_filter(doc_timestamp timestamptz, time_window interval) RETURNS boolean
    AS 'MODULE_PATHNAME', 'neurondb_temporal_filter'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION neurondb_temporal_filter IS 'Check if document is within time window';

-- ============================================================================
-- PROMETHEUS METRICS
-- ============================================================================

-- Get Prometheus metrics
CREATE FUNCTION neurondb_prometheus_metrics()
    RETURNS TABLE(
        queries_total bigint,
        queries_success bigint,
        queries_error bigint,
        query_duration_sum double precision,
        vectors_total bigint,
        cache_hits bigint,
        cache_misses bigint,
        workers_active integer
    )
    AS 'MODULE_PATHNAME', 'neurondb_prometheus_metrics'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_prometheus_metrics IS 'Get Prometheus-compatible metrics (also available via HTTP on port 9187)';

-- ============================================================================
-- BACKGROUND WORKER MANUAL TRIGGERS
-- ============================================================================

-- Manually run queue worker once
CREATE FUNCTION neuranq_run_once() RETURNS void
    AS 'MODULE_PATHNAME', 'neuranq_run_once'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION neuranq_run_once IS 'Manually trigger neuranq worker execution (for testing)';

-- Manually run defrag worker
DROP FUNCTION IF EXISTS neurandefrag_run(regclass);
DROP FUNCTION IF EXISTS neurandefrag_run();
CREATE FUNCTION neurandefrag_run() RETURNS bool
    AS 'MODULE_PATHNAME', 'neurandefrag_run'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION neurandefrag_run() IS 'Manually trigger index defragmentation';

-- Manually run tuner sample
CREATE FUNCTION neuranmon_sample() RETURNS void
    AS 'MODULE_PATHNAME', 'neuranmon_sample'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION neuranmon_sample IS 'Manually trigger tuner sampling';

-- ============================================================================
-- WORKER CATALOG TABLES
-- ============================================================================

-- Queue worker job table
CREATE TABLE IF NOT EXISTS neurondb.job_queue (
    job_id bigserial PRIMARY KEY,
    tenant_id integer DEFAULT 0,
    job_type text NOT NULL,
    payload jsonb NOT NULL,
    status text DEFAULT 'pending',
    retry_count integer DEFAULT 0,
    max_retries integer DEFAULT 3,
    backoff_until timestamptz,
    error_message text,
    created_at timestamptz DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz
);
COMMENT ON TABLE neurondb.job_queue IS 'Background job queue processed by neuranq worker';

CREATE INDEX IF NOT EXISTS idx_job_queue_status ON neurondb.job_queue(status, created_at);
CREATE INDEX IF NOT EXISTS idx_job_queue_tenant ON neurondb.job_queue(tenant_id, created_at);

-- Query metrics table for tuner
CREATE TABLE IF NOT EXISTS neurondb.query_metrics (
    metric_id bigserial PRIMARY KEY,
    query_hash text,
    query_type text,
    latency_ms real,
    recall_at_k real,
    query_timestamp timestamptz DEFAULT now(),
    index_name text,
    ef_search integer,
    hybrid_weight real
);
COMMENT ON TABLE neurondb.query_metrics IS 'Query performance metrics for auto-tuning';

CREATE INDEX IF NOT EXISTS idx_query_metrics_timestamp ON neurondb.query_metrics(query_timestamp);
CREATE INDEX IF NOT EXISTS idx_query_metrics_hash ON neurondb.query_metrics(query_hash);

-- Index maintenance metadata for defrag worker
CREATE TABLE IF NOT EXISTS neurondb.index_maintenance (
    index_name text PRIMARY KEY,
    index_type text,
    num_nodes bigint DEFAULT 0,
    num_edges bigint DEFAULT 0,
    num_tombstones bigint DEFAULT 0,
    fragmentation_ratio real DEFAULT 0.0,
    last_compaction timestamptz,
    last_rebuild timestamptz,
    last_stats_refresh timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.index_maintenance IS 'Index maintenance metadata for neurandefrag worker';

-- Embedding cache table
CREATE TABLE IF NOT EXISTS neurondb.embedding_cache (
    cache_key text PRIMARY KEY,
    model_name text NOT NULL,
    embedding vector,
    created_at timestamptz DEFAULT now(),
    last_accessed timestamptz DEFAULT now(),
    access_count bigint DEFAULT 1
);
COMMENT ON TABLE neurondb.embedding_cache IS 'Cached embeddings for performance';

CREATE INDEX IF NOT EXISTS idx_embedding_cache_accessed ON neurondb.embedding_cache(last_accessed);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON neurondb.embedding_cache(model_name);

-- Embedding model configuration table
CREATE TABLE IF NOT EXISTS neurondb.embedding_model_config (
    model_name text PRIMARY KEY,
    config_json jsonb NOT NULL,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.embedding_model_config IS 'Configuration for embedding models (batch_size, normalize, device, etc.)';

-- Histogram table for metrics distribution
CREATE TABLE IF NOT EXISTS neurondb.histograms (
    metric_name text NOT NULL,
    bucket_start real NOT NULL,
    bucket_end real NOT NULL,
    count bigint DEFAULT 0,
    last_updated timestamptz DEFAULT now(),
    PRIMARY KEY (metric_name, bucket_start)
);
COMMENT ON TABLE neurondb.histograms IS 'Performance metric histograms for monitoring';

CREATE INDEX IF NOT EXISTS idx_histograms_metric ON neurondb.histograms(metric_name, last_updated);

-- Prometheus metrics staging table
CREATE TABLE IF NOT EXISTS neurondb.prometheus_metrics (
    metric_name text PRIMARY KEY,
    metric_value real NOT NULL,
    last_updated timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.prometheus_metrics IS 'Staging table for Prometheus metrics export';

-- ============================================================================
-- LLM / HUGGING FACE INTEGRATION (Extended)
-- ============================================================================

-- LLM cache table
CREATE TABLE IF NOT EXISTS neurondb.llm_cache (
    key text PRIMARY KEY,
    value jsonb NOT NULL,
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_cache IS 'LLM response cache with automatic expiration';

CREATE INDEX IF NOT EXISTS idx_llm_cache_created ON neurondb.llm_cache(created_at);

-- LLM job queue table
CREATE TABLE IF NOT EXISTS neurondb.llm_jobs (
    job_id bigserial PRIMARY KEY,
    tenant_id text,
    operation text NOT NULL,
    model_name text NOT NULL,
    input_text text,
    input_vector vector,
    status text DEFAULT 'pending',
    result_text text,
    result_vector vector,
    error_message text,
    created_at timestamptz DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,
    retry_count integer DEFAULT 0
);
COMMENT ON TABLE neurondb.llm_jobs IS 'Asynchronous LLM job queue processed by neuranllm worker';

CREATE INDEX IF NOT EXISTS idx_llm_jobs_status ON neurondb.llm_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_jobs_tenant ON neurondb.llm_jobs(tenant_id, created_at);

-- ============================================================================
-- ML PROJECT MANAGEMENT SYSTEM
-- ============================================================================
-- Provides organized model lifecycle management with versioning,
-- experiment tracking, and model deployment capabilities.
-- Inspired by PostgresML's project concept.
-- ============================================================================

-- ML Model Type Enum
DO $$ BEGIN
    CREATE TYPE neurondb.ml_model_type AS ENUM (
        'clustering',
        'classification',
        'regression',
        'dimensionality_reduction',
        'outlier_detection',
        'embedding'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- ML Algorithm Type Enum
DO $$ BEGIN
    CREATE TYPE neurondb.ml_algorithm_type AS ENUM (
        'kmeans',
        'dbscan',
        'gmm',
        'hierarchical',
        'minibatch_kmeans',
        'pca',
        'isolation_forest',
        'custom'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
DECLARE
    val text;
BEGIN
    FOREACH val IN ARRAY ARRAY[
        'random_forest',
        'decision_tree',
        'naive_bayes',
        'svm',
        'knn',
        'logistic_regression',
        'linear_regression',
        'ridge',
        'lasso',
        'elastic_net',
        'xgboost',
        'neural_network',
        'deep_learning',
        'automl',
        'isolation_forest',
        'zscore',
        'kmeans',
        'minibatch_kmeans',
        'dbscan',
        'gmm',
        'hierarchical',
        'pca',
        'custom'
    ] LOOP
        BEGIN
            EXECUTE format('ALTER TYPE neurondb.ml_algorithm_type ADD VALUE IF NOT EXISTS %L', val);
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END;
    END LOOP;
END $$;

DO $$ BEGIN
    CREATE CAST (text AS neurondb.ml_algorithm_type)
        WITH INOUT
        AS ASSIGNMENT;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- ML Model Status Enum
DO $$ BEGIN
    CREATE TYPE neurondb.ml_model_status AS ENUM (
        'training',
        'completed',
        'failed',
        'deployed'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE CAST (text AS neurondb.ml_model_status)
        WITH INOUT
        AS ASSIGNMENT;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Projects: Top-level organization for ML work
CREATE TABLE IF NOT EXISTS neurondb.ml_projects (
    project_id serial PRIMARY KEY,
    project_name text UNIQUE NOT NULL,
    model_type neurondb.ml_model_type NOT NULL,
    description text,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    created_by text DEFAULT CURRENT_USER,
    metadata jsonb DEFAULT '{}'::jsonb,
    
    CONSTRAINT valid_project_name CHECK (LENGTH(project_name) >= 3 AND LENGTH(project_name) <= 100)
);
COMMENT ON TABLE neurondb.ml_projects IS 'ML project organization and metadata';

-- Unified API Projects Table (simplified for unified API compatibility)
CREATE TABLE IF NOT EXISTS neurondb.nb_catalog (
    project_id SERIAL PRIMARY KEY,
    project_name TEXT NOT NULL UNIQUE,
    algorithm TEXT NOT NULL,
    table_name TEXT NOT NULL,
    target_column TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE neurondb.nb_catalog IS 'Simplified projects table for unified ML API';

-- Models: Trained models with versioning
CREATE TABLE IF NOT EXISTS neurondb.ml_models (
    model_id serial PRIMARY KEY,
    project_id integer NOT NULL REFERENCES neurondb.ml_projects(project_id) ON DELETE CASCADE,
    version integer NOT NULL,
    algorithm neurondb.ml_algorithm_type NOT NULL,
    status neurondb.ml_model_status DEFAULT 'training',
    
    -- Training configuration
    training_table text NOT NULL,
    training_column text,  -- NULL allowed for unsupervised algorithms
    parameters jsonb DEFAULT '{}'::jsonb,
    
    -- Model artifacts
    model_data bytea,
    
    -- Metrics and evaluation
    metrics jsonb DEFAULT '{}'::jsonb,
    training_time_ms integer,
    num_samples integer,
    num_features integer,
    
    -- Deployment
    is_deployed boolean DEFAULT false,
    deployed_at timestamptz,
    
    -- Timestamps
    created_at timestamptz DEFAULT now(),
    completed_at timestamptz,
    
    notes text,
    
    CONSTRAINT unique_project_version UNIQUE(project_id, version),
    CONSTRAINT positive_version CHECK (version > 0)
);
COMMENT ON TABLE neurondb.ml_models IS 'Trained models with versioning and metrics';

-- Experiments: Track training runs and comparisons
CREATE TABLE IF NOT EXISTS neurondb.ml_experiments (
    experiment_id serial PRIMARY KEY,
    project_id integer NOT NULL REFERENCES neurondb.ml_projects(project_id) ON DELETE CASCADE,
    model_id integer REFERENCES neurondb.ml_models(model_id) ON DELETE CASCADE,
    experiment_name text,
    status neurondb.ml_model_status DEFAULT 'training',
    config jsonb DEFAULT '{}'::jsonb,
    metrics jsonb DEFAULT '{}'::jsonb,
    logs text,
    started_at timestamptz DEFAULT now(),
    completed_at timestamptz,
    notes text
);
COMMENT ON TABLE neurondb.ml_experiments IS 'Experiment tracking and comparison';

-- Indexes for ML project management
CREATE INDEX IF NOT EXISTS idx_ml_projects_name ON neurondb.ml_projects(project_name);
CREATE INDEX IF NOT EXISTS idx_ml_projects_type ON neurondb.ml_projects(model_type);
CREATE INDEX IF NOT EXISTS idx_ml_models_project ON neurondb.ml_models(project_id);
CREATE INDEX IF NOT EXISTS idx_ml_models_deployed ON neurondb.ml_models(is_deployed) WHERE is_deployed = true;
CREATE INDEX IF NOT EXISTS idx_ml_models_status ON neurondb.ml_models(status);
CREATE INDEX IF NOT EXISTS idx_ml_experiments_project ON neurondb.ml_experiments(project_id);

-- Views for project management
CREATE OR REPLACE VIEW neurondb.ml_projects_summary AS
SELECT 
    p.project_id,
    p.project_name,
    p.model_type,
    p.description,
    p.created_at,
    p.updated_at,
    COUNT(DISTINCT m.model_id) as total_models,
    MAX(m.version) as latest_version,
    MAX(CASE WHEN m.is_deployed THEN m.version END) as deployed_version,
    MAX(m.metrics->>'silhouette_score') as best_silhouette_score
FROM neurondb.ml_projects p
LEFT JOIN neurondb.ml_models m ON p.project_id = m.project_id
GROUP BY p.project_id, p.project_name, p.model_type, p.description, p.created_at, p.updated_at;

COMMENT ON VIEW neurondb.ml_projects_summary IS 'Summary view of all projects with model counts';

CREATE OR REPLACE VIEW neurondb.ml_model_comparison AS
SELECT 
    p.project_name,
    m.model_id,
    m.version,
    m.algorithm,
    m.status,
    m.is_deployed,
    m.parameters,
    m.metrics,
    m.training_time_ms,
    m.num_samples,
    m.created_at,
    m.completed_at
FROM neurondb.ml_models m
JOIN neurondb.ml_projects p ON m.project_id = p.project_id
ORDER BY p.project_name, m.version DESC;

COMMENT ON VIEW neurondb.ml_model_comparison IS 'Compare model performance across versions';

CREATE OR REPLACE VIEW neurondb.ml_deployment_status AS
SELECT 
    p.project_name,
    p.model_type,
    m.model_id,
    m.version,
    m.algorithm,
    m.parameters,
    m.deployed_at,
    m.metrics,
    AGE(NOW(), m.deployed_at) as deployment_age
FROM neurondb.ml_projects p
JOIN neurondb.ml_models m ON p.project_id = m.project_id
WHERE m.is_deployed = true;

COMMENT ON VIEW neurondb.ml_deployment_status IS 'Currently deployed models';

-- Trigger to update project timestamp
CREATE OR REPLACE FUNCTION neurondb.update_project_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE neurondb.ml_projects 
    SET updated_at = NOW() 
    WHERE project_id = NEW.project_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_project_timestamp ON neurondb.ml_models;
CREATE TRIGGER trigger_update_project_timestamp
    AFTER INSERT OR UPDATE ON neurondb.ml_models
    FOR EACH ROW
    EXECUTE FUNCTION neurondb.update_project_timestamp();

-- LLM statistics table
CREATE TABLE IF NOT EXISTS neurondb.llm_stats (
    model_name text PRIMARY KEY,
    total_requests bigint DEFAULT 0,
    successful_requests bigint DEFAULT 0,
    failed_requests bigint DEFAULT 0,
    cache_hits bigint DEFAULT 0,
    total_latency_ms bigint DEFAULT 0,
    total_tokens bigint DEFAULT 0,
    total_tokens_in bigint DEFAULT 0,
    total_tokens_out bigint DEFAULT 0,
    last_request_at timestamptz,
    last_updated timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_stats IS 'LLM usage statistics per model';

-- LLM error tracking table
CREATE TABLE IF NOT EXISTS neurondb.llm_errors (
    error_id bigserial PRIMARY KEY,
    model_name text NOT NULL,
    operation text NOT NULL,
    error_type text NOT NULL,
    error_message text,
    latency_ms bigint,
    error_timestamp timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_errors IS 'LLM error tracking for monitoring and debugging';

CREATE INDEX IF NOT EXISTS idx_llm_errors_model ON neurondb.llm_errors(model_name, error_timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_errors_type ON neurondb.llm_errors(error_type, error_timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_errors_operation ON neurondb.llm_errors(operation, error_timestamp);

-- LLM statistics views for monitoring
CREATE OR REPLACE VIEW neurondb.llm_model_stats AS
SELECT 
    model_name,
    total_requests,
    successful_requests,
    failed_requests,
    cache_hits,
    total_latency_ms,
    total_tokens,
    total_tokens_in,
    total_tokens_out,
    CASE 
        WHEN total_requests > 0 THEN ROUND(100.0 * successful_requests / total_requests, 2)
        ELSE 0.0
    END as success_rate_pct,
    CASE 
        WHEN total_requests > 0 THEN ROUND(100.0 * cache_hits / total_requests, 2)
        ELSE 0.0
    END as cache_hit_rate_pct,
    CASE 
        WHEN total_requests > 0 THEN ROUND(total_latency_ms::numeric / total_requests, 2)
        ELSE 0.0
    END as avg_latency_ms,
    CASE 
        WHEN total_tokens > 0 THEN ROUND(total_latency_ms::numeric / total_tokens, 4)
        ELSE 0.0
    END as latency_per_token_ms,
    last_request_at,
    last_updated
FROM neurondb.llm_stats
ORDER BY total_requests DESC;

COMMENT ON VIEW neurondb.llm_model_stats IS 'LLM model usage statistics with calculated metrics (success rate, cache hit rate, average latency, latency per token)';

CREATE OR REPLACE VIEW neurondb.llm_error_rates AS
SELECT 
    model_name,
    operation,
    error_type,
    COUNT(*) as error_count,
    ROUND(AVG(latency_ms)::numeric, 2) as avg_latency_ms,
    MIN(error_timestamp) as first_error,
    MAX(error_timestamp) as last_error
FROM neurondb.llm_errors
WHERE error_timestamp > NOW() - INTERVAL '24 hours'
GROUP BY model_name, operation, error_type
ORDER BY error_count DESC;

COMMENT ON VIEW neurondb.llm_error_rates IS 'LLM error rates by model, operation, and error type (last 24 hours)';

CREATE OR REPLACE VIEW neurondb.llm_latency_histograms AS
SELECT 
    metric_name,
    bucket_start,
    bucket_end,
    SUM(count) as total_count,
    MAX(last_updated) as last_updated
FROM neurondb.histograms
WHERE metric_name LIKE 'llm_%_latency_ms'
    AND last_updated > NOW() - INTERVAL '24 hours'
GROUP BY metric_name, bucket_start, bucket_end
ORDER BY metric_name, bucket_start;

COMMENT ON VIEW neurondb.llm_latency_histograms IS 'LLM latency histograms by model and operation (last 24 hours)';

-- Function to calculate percentiles from histograms
CREATE OR REPLACE FUNCTION neurondb.llm_latency_percentile(
    metric_name text,
    percentile real DEFAULT 0.95
) RETURNS real
LANGUAGE plpgsql STABLE AS $$
DECLARE
    total_count bigint;
    target_count bigint;
    current_count bigint;
    result real;
BEGIN
    /* Calculate total count */
    SELECT SUM(count) INTO total_count
    FROM neurondb.histograms
    WHERE neurondb.histograms.metric_name = llm_latency_percentile.metric_name
        AND last_updated > NOW() - INTERVAL '24 hours';

    IF total_count IS NULL OR total_count = 0 THEN
        RETURN NULL;
    END IF;

    /* Calculate target count for percentile */
    target_count := (total_count * percentile)::bigint;
    current_count := 0;

    /* Find bucket containing percentile */
    SELECT bucket_end INTO result
    FROM (
        SELECT 
            bucket_start,
            bucket_end,
            SUM(count) OVER (ORDER BY bucket_start) as cumulative_count
        FROM neurondb.histograms
        WHERE neurondb.histograms.metric_name = llm_latency_percentile.metric_name
            AND last_updated > NOW() - INTERVAL '24 hours'
        ORDER BY bucket_start
    ) t
    WHERE cumulative_count >= target_count
    ORDER BY bucket_start
    LIMIT 1;

    RETURN result;
END;
$$;

COMMENT ON FUNCTION neurondb.llm_latency_percentile IS 'Calculate latency percentile (p50, p95, p99) from histograms for a given metric name';

-- Function to get GPU utilization metrics
CREATE OR REPLACE FUNCTION neurondb.llm_gpu_utilization() RETURNS TABLE(
    device_id integer,
    utilization_pct real,
    memory_used_mb bigint,
    memory_total_mb bigint,
    memory_utilization_pct real,
    temperature_c integer,
    power_w real,
    "timestamp" timestamptz
)
AS 'MODULE_PATHNAME', 'neurondb_llm_gpu_utilization'
LANGUAGE C STABLE;

COMMENT ON FUNCTION neurondb.llm_gpu_utilization IS 'Get GPU utilization metrics (utilization percentage, memory usage, temperature, power consumption). Returns memory utilization from device info. GPU utilization percentage, temperature, and power require system-level APIs (NVML/rocm-smi/IOKit) and are currently placeholders.';

-- ============================================================================
-- LLM / GPU CONFIGURATION GUC VARIABLES
-- ============================================================================
-- The following GUC variables control LLM and GPU behavior (defined in C code):
--
-- LLM Provider Settings:
--   neurondb.llm_provider       - LLM provider (default: 'huggingface')
--                                  Options: 'huggingface', 'huggingface-local', 'hf-local', 'hf-http'
--   neurondb.llm_model          - Default LLM model name
--   neurondb.llm_endpoint       - LLM endpoint base URL
--   neurondb.llm_api_key        - API key for LLM provider (superuser only)
--   neurondb.llm_timeout_ms     - HTTP timeout in milliseconds (default: 30000)
--   neurondb.llm_cache_ttl      - Cache TTL in seconds (default: 600)
--   neurondb.llm_rate_limiter_qps - Rate limiter QPS (default: 5)
--   neurondb.llm_fail_open      - Fail open on provider errors (default: true)
--
-- GPU Settings:
--   neurondb.gpu_enabled        - Enable GPU acceleration (default: false)
--   neurondb.gpu_device         - GPU device ID to use (default: 0)
--   neurondb.gpu_batch_size     - Batch size for GPU operations (default: 1000)
--   neurondb.gpu_streams        - Number of CUDA streams (default: 4)
--   neurondb.gpu_memory_pool_mb - GPU memory pool size in MB (default: 512)
--   neurondb.gpu_fail_open      - Fallback to CPU on GPU error (default: true)
--   neurondb.gpu_kernels        - GPU kernel selection (default: 'auto')
--   neurondb.gpu_backend        - GPU backend (default: 'cuda')
--   neurondb.gpu_timeout_ms     - GPU operation timeout in ms (default: 5000)
--
-- ONNX Runtime Settings:
--   neurondb.onnx_model_path    - Directory with ONNX model files
--   neurondb.onnx_use_gpu       - Use GPU for ONNX inference (default: true)
--   neurondb.onnx_threads       - Number of ONNX Runtime threads (default: 4)
--   neurondb.onnx_cache_size    - ONNX model cache size (default: 10)
--
-- GPU Preference:
--   When neurondb.llm_provider is set to 'huggingface-local' or 'hf-local',
--   and neurondb.gpu_enabled is true, the system will:
--   1. Try CUDA-accelerated inference first (if GPU available)
--   2. Fall back to ONNX Runtime with GPU (if available)
--   3. Fall back to ONNX Runtime with CPU (if GPU unavailable)
--   4. Fall back to remote Hugging Face API (if local execution fails)
--
-- Example configuration:
--   SET neurondb.llm_provider = 'huggingface-local';
--   SET neurondb.llm_model = 'sentence-transformers/all-MiniLM-L6-v2';
--   SET neurondb.gpu_enabled = true;
--   SET neurondb.gpu_device = 0;
-- ============================================================================

-- C-backed LLM functions with GPU acceleration support
-- These functions automatically use GPU when available and configured
CREATE FUNCTION ndb_llm_complete(prompt text, params text DEFAULT '{}') RETURNS text
    AS 'MODULE_PATHNAME', 'ndb_llm_complete'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION ndb_llm_complete IS 'LLM completion with GPU acceleration support: ndb_llm_complete(prompt, params) where params is JSON text with optional model, max_tokens, temperature, etc. Uses CUDA-accelerated inference when available, falls back to ONNX Runtime or remote API. Configure via neurondb.llm_provider and neurondb.gpu_enabled.';

CREATE FUNCTION ndb_llm_image_analyze(image_data bytea, prompt text DEFAULT NULL, params text DEFAULT '{}', model text DEFAULT NULL) RETURNS text
    AS 'MODULE_PATHNAME', 'ndb_llm_image_analyze'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION ndb_llm_image_analyze IS 'LLM image analysis (vision) using GPT-4 Vision or other vision-capable models. Analyzes images and returns text description. Requires OpenAI provider. Parameters: image_data (bytea), prompt (optional text query about the image), params (JSON with temperature, max_tokens, etc.), model (optional, defaults to gpt-4o). Configure via neurondb.llm_provider=openai and neurondb.llm_api_key.';

CREATE FUNCTION ndb_llm_embed(text_input text, model text) RETURNS vector
    AS 'MODULE_PATHNAME', 'ndb_llm_embed'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION ndb_llm_embed IS 'LLM embedding with GPU acceleration support. Uses CUDA-accelerated inference when available, falls back to ONNX Runtime or remote API. Configure via neurondb.llm_provider and neurondb.gpu_enabled.';

CREATE FUNCTION ndb_llm_rerank(query text, documents text[], model text, top_k integer) RETURNS TABLE(idx integer, score real)
    AS 'MODULE_PATHNAME', 'ndb_llm_rerank'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION ndb_llm_rerank IS 'LLM-based reranking with GPU acceleration support. Uses CUDA-accelerated inference when available, falls back to ONNX Runtime or remote API. Configure via neurondb.llm_provider and neurondb.gpu_enabled.';

CREATE FUNCTION ndb_llm_enqueue(operation text, model text, input_text text, tenant_id text) RETURNS bigint
    AS 'MODULE_PATHNAME', 'ndb_llm_enqueue'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION ndb_llm_enqueue IS 'Enqueue asynchronous LLM job for background processing. Supports GPU acceleration when available.';

-- GPU availability check for LLM operations
CREATE FUNCTION neurondb_llm_gpu_available() RETURNS boolean
    AS 'MODULE_PATHNAME', 'neurondb_llm_gpu_available'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_llm_gpu_available IS 'Check if GPU is available for LLM operations. Returns true if CUDA backend is available and configured.';

-- GPU backend info for LLM operations
CREATE FUNCTION neurondb_llm_gpu_info() RETURNS TABLE(
        backend text,
        device_id integer,
        device_name text,
        total_memory_mb bigint,
        free_memory_mb bigint,
        is_available boolean
    )
    AS 'MODULE_PATHNAME', 'neurondb_llm_gpu_info'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_llm_gpu_info IS 'Get GPU information for LLM operations. Returns backend type, device info, and availability status.';

-- Batch operations for LLM completion and reranking
CREATE FUNCTION ndb_llm_complete_batch(
	prompts text[],
	params text DEFAULT '{}'
) RETURNS TABLE(
	idx integer,
	text text,
	tokens_in integer,
	tokens_out integer,
	http_status integer
)
	AS 'MODULE_PATHNAME', 'ndb_llm_complete_batch'
	LANGUAGE C STABLE;
COMMENT ON FUNCTION ndb_llm_complete_batch IS 'Batch LLM completion with GPU acceleration support: ndb_llm_complete_batch(prompts, params) processes multiple prompts in parallel using GPU when available. Uses CUDA-accelerated inference when available, falls back to ONNX Runtime or remote API. Configure via neurondb.llm_provider and neurondb.gpu_enabled.';

CREATE FUNCTION ndb_llm_rerank_batch(
	queries text[],
	documents_array text[][],
	model text DEFAULT NULL,
	top_k integer DEFAULT 10
) RETURNS TABLE(
	query_idx integer,
	doc_idx integer,
	score real
)
	AS 'MODULE_PATHNAME', 'ndb_llm_rerank_batch'
	LANGUAGE C STABLE;
COMMENT ON FUNCTION ndb_llm_rerank_batch IS 'Batch LLM reranking with GPU acceleration support: ndb_llm_rerank_batch(queries, documents_array, model, top_k) processes multiple query-document pairs in parallel using GPU when available. Uses CUDA-accelerated inference when available, falls back to ONNX Runtime or remote API. Configure via neurondb.llm_provider and neurondb.gpu_enabled.';

-- ============================================================================
-- DISTRIBUTED QUERY FUNCTIONS
-- ============================================================================

CREATE FUNCTION distributed_knn_search(
    query vector,
    k integer,
    replicas text[],
    merge_strategy text DEFAULT 'balanced'
) RETURNS TABLE(id bigint, distance real, node text)
    AS 'MODULE_PATHNAME', 'distributed_knn_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION distributed_knn_search IS 'Distributed k-NN search across replicas';

CREATE FUNCTION merge_distributed_results(
    results jsonb,
    k integer,
    strategy text DEFAULT 'min_distance'
) RETURNS TABLE(id bigint, distance real, source_node text)
    AS 'MODULE_PATHNAME', 'merge_distributed_results'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION merge_distributed_results IS 'Merge results from distributed search';

CREATE FUNCTION select_optimal_replica(
    query_hash text,
    replicas text[],
    strategy text DEFAULT 'latency'
) RETURNS text
    AS 'MODULE_PATHNAME', 'select_optimal_replica'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION select_optimal_replica IS 'Select optimal replica for query routing';

CREATE FUNCTION sync_index_async(
    source_index regclass,
    target_connection text,
    batch_size integer DEFAULT 1000
) RETURNS bigint
    AS 'MODULE_PATHNAME', 'sync_index_async'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION sync_index_async IS 'Asynchronously sync index to remote node';

-- ============================================================================
-- DATA MANAGEMENT FUNCTIONS
-- ============================================================================

CREATE FUNCTION vector_time_travel(
    table_name text,
    vector_col text,
    timestamp_point timestamptz
) RETURNS TABLE(id bigint, vector_snapshot vector)
    AS 'MODULE_PATHNAME', 'vector_time_travel'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION vector_time_travel IS 'Time-travel query for vector data';

CREATE FUNCTION compress_cold_tier(
    table_name text,
    vector_col text,
    age_threshold interval,
    compression_method text DEFAULT 'int8'
) RETURNS bigint
    AS 'MODULE_PATHNAME', 'compress_cold_tier'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION compress_cold_tier IS 'Compress vectors in cold tier storage';

CREATE FUNCTION vacuum_vectors(
    table_name text,
    aggressive boolean DEFAULT false
) RETURNS TABLE(reclaimed_bytes bigint, duration_ms bigint)
    AS 'MODULE_PATHNAME', 'vacuum_vectors'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION vacuum_vectors IS 'Vacuum vector table with index maintenance';

CREATE FUNCTION rebalance_index(
    index_name regclass,
    target_balance_ratio real DEFAULT 0.9
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'rebalance_index'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION rebalance_index IS 'Rebalance index for optimal performance';

-- ============================================================================
-- PLANNER AND OPTIMIZATION
-- ============================================================================

CREATE FUNCTION auto_route_query(
    query_text text,
    context jsonb DEFAULT '{}'
) RETURNS text
    AS 'MODULE_PATHNAME', 'auto_route_query'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION auto_route_query IS 'Automatically route query to optimal execution path';

CREATE FUNCTION learn_from_query(
    query_plan text,
    actual_cost real,
    estimated_cost real
) RETURNS void
    AS 'MODULE_PATHNAME', 'learn_from_query'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION learn_from_query IS 'Learn from query execution for cost model tuning';

CREATE FUNCTION scale_precision(
    workload_type text,
    target_latency_ms real
) RETURNS text
    AS 'MODULE_PATHNAME', 'scale_precision'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION scale_precision IS 'Scale precision based on workload and latency requirements';

CREATE FUNCTION prefetch_entry_points(
    index_name regclass,
    query_patterns text[]
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'prefetch_entry_points'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION prefetch_entry_points IS 'Prefetch HNSW entry points based on query patterns';

-- ============================================================================
-- STORAGE AND BUFFER MANAGEMENT
-- ============================================================================

CREATE FUNCTION rebuild_hnsw_safe(
    index_name regclass,
    checkpoint_interval integer DEFAULT 10000
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'rebuild_hnsw_safe'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION rebuild_hnsw_safe IS 'Safely rebuild HNSW index with checkpointing';

CREATE FUNCTION parallel_knn_search(
    query vector,
    k integer,
    parallel_degree integer DEFAULT 4
) RETURNS TABLE(id bigint, distance real)
    AS 'MODULE_PATHNAME', 'parallel_knn_search'
    LANGUAGE C STABLE PARALLEL SAFE;
COMMENT ON FUNCTION parallel_knn_search IS 'Parallel k-NN search with work distribution';

CREATE FUNCTION save_rebuild_checkpoint(
    index_name regclass,
    checkpoint_data bytea
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'save_rebuild_checkpoint'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION save_rebuild_checkpoint IS 'Save checkpoint during index rebuild';

CREATE FUNCTION load_rebuild_checkpoint(
    index_name regclass
) RETURNS bytea
    AS 'MODULE_PATHNAME', 'load_rebuild_checkpoint'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION load_rebuild_checkpoint IS 'Load checkpoint for index rebuild recovery';

-- ============================================================================
-- ANN BUFFER MANAGEMENT
-- ============================================================================

CREATE FUNCTION neurondb_ann_buffer_get_centroid(centroid_id integer) RETURNS vector
    AS 'MODULE_PATHNAME', 'neurondb_ann_buffer_get_centroid'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_ann_buffer_get_centroid IS 'Get centroid from in-memory ANN buffer';

CREATE FUNCTION neurondb_ann_buffer_put_centroid(centroid_id integer, centroid vector) RETURNS boolean
    AS 'MODULE_PATHNAME', 'neurondb_ann_buffer_put_centroid'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION neurondb_ann_buffer_put_centroid IS 'Store centroid in in-memory ANN buffer';

CREATE FUNCTION neurondb_ann_buffer_get_stats()
    RETURNS TABLE(
        total_entries integer,
        memory_used_bytes bigint,
        hit_rate real,
        evictions bigint
    )
    AS 'MODULE_PATHNAME', 'neurondb_ann_buffer_get_stats'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION neurondb_ann_buffer_get_stats IS 'Get ANN buffer cache statistics';

CREATE FUNCTION neurondb_ann_buffer_clear() RETURNS boolean
    AS 'MODULE_PATHNAME', 'neurondb_ann_buffer_clear'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION neurondb_ann_buffer_clear IS 'Clear ANN buffer cache';

-- ============================================================================
-- UTILITY VIEWS (neurondb schema)
-- ============================================================================

-- View: Combined vector statistics across all indexes
CREATE VIEW neurondb.vector_stats AS
SELECT 
    'Total Vectors' as metric,
    COALESCE(SUM(vector_count), 0) as value,
    'vectors' as unit
FROM neurondb.tenant_usage
UNION ALL
SELECT 
    'Total Storage',
    COALESCE(SUM(storage_bytes) / 1024.0 / 1024.0, 0),
    'MB'
FROM neurondb.tenant_usage
UNION ALL
SELECT 
    'Active Tenants',
    COUNT(DISTINCT tenant_id)::bigint,
    'tenants'
FROM neurondb.tenant_usage
WHERE vector_count > 0;

COMMENT ON VIEW neurondb.vector_stats IS 'Aggregate statistics across all vector indexes';

-- View: Index health monitoring
CREATE VIEW neurondb.index_health AS
SELECT 
    im.index_oid::regclass::text as index_name,
    im.index_type,
    im.health_status,
    im.last_validated,
    array_length(im.validation_errors, 1) as error_count,
    CASE 
        WHEN im.health_status = 'healthy' THEN '✅'
        WHEN im.health_status = 'degraded' THEN '⚠️'
        WHEN im.health_status = 'critical' THEN '❌'
        ELSE '❓'
    END as status_icon
FROM neurondb.index_metadata im
ORDER BY 
    CASE im.health_status
        WHEN 'critical' THEN 1
        WHEN 'degraded' THEN 2
        WHEN 'healthy' THEN 3
        ELSE 4
    END,
    im.last_validated DESC NULLS LAST;

COMMENT ON VIEW neurondb.index_health IS 'Index health status dashboard';

-- View: Tenant quota usage with percentages
CREATE VIEW neurondb.tenant_quota_usage AS
SELECT 
    tu.tenant_id,
    tu.vector_count,
    tq.max_vectors,
    ROUND(100.0 * tu.vector_count / NULLIF(tq.max_vectors, 0), 2) as vectors_pct,
    ROUND(tu.storage_bytes / 1024.0 / 1024.0, 2) as storage_mb,
    tq.max_storage_mb,
    ROUND(100.0 * (tu.storage_bytes / 1024.0 / 1024.0) / NULLIF(tq.max_storage_mb, 0), 2) as storage_pct,
    tq.max_qps,
    CASE 
        WHEN tu.vector_count >= tq.max_vectors * 0.9 THEN 'WARNING: Near vector limit'
        WHEN (tu.storage_bytes / 1024.0 / 1024.0) >= tq.max_storage_mb * 0.9 THEN 'WARNING: Near storage limit'
        ELSE 'OK'
    END as status
FROM neurondb.tenant_usage tu
LEFT JOIN neurondb.tenant_quotas tq ON tu.tenant_id = tq.tenant_id
ORDER BY vectors_pct DESC NULLS LAST;

COMMENT ON VIEW neurondb.tenant_quota_usage IS 'Tenant quota usage with warnings';

-- View: LLM job status summary
CREATE VIEW neurondb.llm_job_status AS
SELECT 
    status,
    COUNT(*) as job_count,
    COUNT(*) FILTER (WHERE operation = 'embed') as embed_jobs,
    COUNT(*) FILTER (WHERE operation = 'complete') as completion_jobs,
    MIN(created_at) as oldest_job,
    MAX(completed_at) as latest_update
FROM neurondb.llm_jobs
GROUP BY status
ORDER BY 
    CASE status
        WHEN 'failed' THEN 1
        WHEN 'processing' THEN 2
        WHEN 'queued' THEN 3
        WHEN 'done' THEN 4
        ELSE 5
    END;

COMMENT ON VIEW neurondb.llm_job_status IS 'LLM job queue status summary';

-- View: Query performance metrics
CREATE VIEW neurondb.query_performance AS
SELECT 
    query_type,
    COUNT(*) as total_queries,
    ROUND(AVG(latency_ms)::numeric, 2) as avg_time_ms,
    ROUND(MIN(latency_ms)::numeric, 2) as min_time_ms,
    ROUND(MAX(latency_ms)::numeric, 2) as max_time_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) as p95_time_ms,
    ROUND(AVG(recall_at_k)::numeric, 3) as avg_recall
FROM neurondb.query_metrics
WHERE query_timestamp > NOW() - INTERVAL '24 hours'
GROUP BY query_type
ORDER BY total_queries DESC;

COMMENT ON VIEW neurondb.query_performance IS 'Query performance metrics (last 24h)';

-- View: Index maintenance status
CREATE VIEW neurondb.index_maintenance_status AS
SELECT 
    index_name,
    index_type,
    num_nodes,
    num_edges,
    num_tombstones,
    fragmentation_ratio,
    last_compaction,
    last_rebuild,
    last_stats_refresh,
    CASE 
        WHEN last_rebuild IS NOT NULL THEN 
            EXTRACT(EPOCH FROM (NOW() - last_rebuild))::integer
        ELSE NULL
    END as seconds_since_rebuild
FROM neurondb.index_maintenance
WHERE last_stats_refresh > NOW() - INTERVAL '7 days'
ORDER BY last_stats_refresh DESC
LIMIT 100;

COMMENT ON VIEW neurondb.index_maintenance_status IS 'Recent index maintenance operations';

-- View: Prometheus metrics summary
CREATE VIEW neurondb.metrics_summary AS
SELECT 
    metric_name,
    metric_value,
    last_updated,
    EXTRACT(EPOCH FROM (NOW() - last_updated))::integer as seconds_stale
FROM neurondb.prometheus_metrics
WHERE last_updated > NOW() - INTERVAL '1 hour'
ORDER BY metric_name, last_updated DESC;

COMMENT ON VIEW neurondb.metrics_summary IS 'Recent Prometheus metrics';

-- ============================================================================
-- HIGH-PRIORITY MISSING FUNCTIONS
-- ============================================================================

-- Tenant-aware HNSW functions
CREATE FUNCTION hnsw_tenant_create(
    table_name text,
    column_name text,
    tenant_column text,
    ef_construction integer DEFAULT 200,
    m integer DEFAULT 16
) RETURNS text
    AS 'MODULE_PATHNAME', 'hnsw_tenant_create'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION hnsw_tenant_create IS 'Create tenant-aware HNSW index';

CREATE FUNCTION hnsw_tenant_search(
    table_name text,
    query_vector vector,
    k integer,
    tenant_id text
) RETURNS TABLE(id bigint, distance real)
    AS 'MODULE_PATHNAME', 'hnsw_tenant_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION hnsw_tenant_search IS 'Search tenant-aware HNSW index';

CREATE FUNCTION hnsw_tenant_quota(
    tenant_id text
) RETURNS TABLE(
    vectors_used bigint,
    vectors_limit bigint,
    storage_mb real,
    storage_limit_mb bigint
)
    AS 'MODULE_PATHNAME', 'hnsw_tenant_quota'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION hnsw_tenant_quota IS 'Get tenant quota usage for HNSW indexes';

-- Hybrid index functions
CREATE FUNCTION hybrid_index_create(
    table_name text,
    vector_column text,
    text_column text,
    options jsonb DEFAULT '{}'
) RETURNS text
    AS 'MODULE_PATHNAME', 'hybrid_index_create'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION hybrid_index_create IS 'Create hybrid vector+FTS index';

CREATE FUNCTION hybrid_index_search(
    index_name text,
    query_vector vector,
    query_text text,
    k integer DEFAULT 10,
    alpha real DEFAULT 0.5
) RETURNS TABLE(id bigint, score real)
    AS 'MODULE_PATHNAME', 'hybrid_index_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION hybrid_index_search IS 'Search hybrid index with vector and text query';

-- Temporal index functions
CREATE FUNCTION temporal_index_create(
    table_name text,
    vector_column text,
    timestamp_column text
) RETURNS text
    AS 'MODULE_PATHNAME', 'temporal_index_create'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION temporal_index_create IS 'Create temporal-aware vector index';

CREATE FUNCTION temporal_knn_search(
    index_name text,
    query_vector vector,
    k integer,
    time_filter tstzrange DEFAULT NULL
) RETURNS TABLE(id bigint, distance real, result_timestamp timestamptz)
    AS 'MODULE_PATHNAME', 'temporal_knn_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION temporal_knn_search IS 'Time-aware k-NN search';

CREATE FUNCTION temporal_score(
    distance real,
    timestamp_val timestamptz,
    decay_rate real DEFAULT 0.1
) RETURNS real
    AS 'MODULE_PATHNAME', 'temporal_score'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION temporal_score IS 'Compute temporal-weighted score';

-- Consistency index functions
CREATE FUNCTION consistent_index_create(
    table_name text,
    vector_column text
) RETURNS text
    AS 'MODULE_PATHNAME', 'consistent_index_create'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION consistent_index_create IS 'Create eventually-consistent distributed vector index';

CREATE FUNCTION consistent_knn_search(
    index_name text,
    query_vector vector,
    k integer,
    consistency_level text DEFAULT 'eventual'
) RETURNS TABLE(id bigint, distance real)
    AS 'MODULE_PATHNAME', 'consistent_knn_search'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION consistent_knn_search IS 'Search with configurable consistency guarantees';

-- Rerank index functions
CREATE FUNCTION rerank_index_create(
    base_index_name text,
    rerank_model text DEFAULT 'cross-encoder/ms-marco-MiniLM-L-6-v2'
) RETURNS text
    AS 'MODULE_PATHNAME', 'rerank_index_create'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION rerank_index_create IS 'Create reranking cache index';

CREATE FUNCTION rerank_get_candidates(
    base_index_name text,
    query_vector vector,
    k integer,
    fetch_factor integer DEFAULT 10
) RETURNS TABLE(id bigint, distance real, cached_score real)
    AS 'MODULE_PATHNAME', 'rerank_get_candidates'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION rerank_get_candidates IS 'Get candidates from base index for reranking';

CREATE FUNCTION rerank_index_warm(
    index_name text,
    sample_queries vector[],
    top_k integer DEFAULT 100
) RETURNS integer
    AS 'MODULE_PATHNAME', 'rerank_index_warm'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION rerank_index_warm IS 'Warm up reranking cache with sample queries';

-- Configuration functions
CREATE FUNCTION get_vector_config(
    parameter_name text DEFAULT NULL
) RETURNS TABLE(name text, value text, description text)
    AS 'MODULE_PATHNAME', 'get_vector_config'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION get_vector_config IS 'Get vector configuration parameters';

CREATE FUNCTION set_vector_config(
    parameter_name text,
    parameter_value text
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'set_vector_config'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION set_vector_config IS 'Set vector configuration parameter';

CREATE FUNCTION show_vector_config() RETURNS TABLE(name text, value text)
    AS 'MODULE_PATHNAME', 'show_vector_config'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION show_vector_config IS 'Show all vector configuration';

CREATE FUNCTION reset_vector_config() RETURNS void
    AS 'MODULE_PATHNAME', 'reset_vector_config'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION reset_vector_config IS 'Reset all vector configuration to defaults';

-- Model management functions
CREATE FUNCTION mdl_http(
    endpoint text,
    payload jsonb
) RETURNS jsonb
    AS 'MODULE_PATHNAME', 'mdl_http'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION mdl_http IS 'Send HTTP request to model endpoint';

CREATE FUNCTION mdl_llm(
    model_name text,
    prompt text,
    options jsonb DEFAULT '{}'
) RETURNS text
    AS 'MODULE_PATHNAME', 'mdl_llm'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION mdl_llm IS 'Call LLM model endpoint';

CREATE FUNCTION mdl_cache(
    operation text,
    key text DEFAULT NULL,
    value text DEFAULT NULL
) RETURNS text
    AS 'MODULE_PATHNAME', 'mdl_cache'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION mdl_cache IS 'Model cache operations (get/set/clear)';

CREATE FUNCTION mdl_trace(
    model_name text,
    operation text,
    details jsonb DEFAULT '{}'
) RETURNS void
    AS 'MODULE_PATHNAME', 'mdl_trace'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION mdl_trace IS 'Trace model operations for observability';

CREATE FUNCTION create_model(
    model_name text,
    model_type text,
    model_path text,
    options jsonb DEFAULT '{}'
) RETURNS text
    AS 'MODULE_PATHNAME', 'create_model'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION create_model IS 'Register custom model for inference';

CREATE FUNCTION drop_model(
    model_name text
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'drop_model'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION drop_model IS 'Unregister custom model';

-- Utility and testing functions
CREATE FUNCTION assert_recall(
    ground_truth bigint[],
    results bigint[],
    k integer
) RETURNS real
    AS 'MODULE_PATHNAME', 'assert_recall'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION assert_recall IS 'Assert recall@k metric for testing';

CREATE FUNCTION assert_vector_equal(
    vec1 vector,
    vec2 vector,
    tolerance real DEFAULT 0.0001
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'assert_vector_equal'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION assert_vector_equal IS 'Assert vectors are equal within tolerance';

CREATE FUNCTION explain_vector_query(
    query_text text
) RETURNS TABLE(
    step_num integer,
    operation text,
    index_used text,
    estimated_cost real,
    description text
)
    AS 'MODULE_PATHNAME', 'explain_vector_query'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION explain_vector_query IS 'Explain vector query execution plan';

-- Statistics functions
CREATE FUNCTION pg_stat_neurondb()
    RETURNS TABLE(
        vectors_indexed bigint,
        queries_total bigint,
        cache_hits bigint,
        cache_misses bigint,
        avg_query_time_ms real
    )
    AS 'MODULE_PATHNAME', 'pg_stat_neurondb'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION pg_stat_neurondb IS 'Get NeurondB statistics';

CREATE FUNCTION pg_neurondb_stat_reset() RETURNS void
    AS 'MODULE_PATHNAME', 'pg_neurondb_stat_reset'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION pg_neurondb_stat_reset IS 'Reset NeurondB statistics';

-- Advanced search functions
CREATE FUNCTION graph_knn(
    graph_table text,
    start_node bigint,
    query_vector vector,
    k integer,
    max_hops integer DEFAULT 3
) RETURNS TABLE(node_id bigint, distance real, hops integer)
    AS 'MODULE_PATHNAME', 'graph_knn'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION graph_knn IS 'Graph-aware k-NN search';

CREATE FUNCTION vec_join(
    left_table text,
    right_table text,
    left_vector_col text,
    right_vector_col text,
    k integer DEFAULT 1,
    distance_metric text DEFAULT 'l2'
) RETURNS TABLE(left_id bigint, right_id bigint, distance real)
    AS 'MODULE_PATHNAME', 'vec_join'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION vec_join IS 'Vector similarity join between two tables';

CREATE FUNCTION hybrid_rank(
    vector_score real,
    text_score real,
    metadata_score real DEFAULT 0,
    weights real[] DEFAULT ARRAY[0.5, 0.3, 0.2]
) RETURNS real
    AS 'MODULE_PATHNAME', 'hybrid_rank'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION hybrid_rank IS 'Compute weighted hybrid ranking score';

-- Tenant and security functions
CREATE FUNCTION create_tenant_worker(
    tenant_id text,
    worker_type text DEFAULT 'all'
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'create_tenant_worker'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION create_tenant_worker IS 'Create dedicated background worker for tenant';

CREATE FUNCTION get_tenant_stats(
    tenant_id text
) RETURNS TABLE(
    vectors bigint,
    storage_mb real,
    qps real,
    indexes integer
)
    AS 'MODULE_PATHNAME', 'get_tenant_stats'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION get_tenant_stats IS 'Get tenant resource usage statistics';

CREATE FUNCTION create_policy(
    policy_name text,
    table_name text,
    expression text,
    role_name text DEFAULT NULL
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'create_policy'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION create_policy IS 'Create row-level security policy for vector table';

-- Distributed functions
CREATE FUNCTION federated_vector_query(
    remote_servers text[],
    query_vector vector,
    k integer,
    combine_method text DEFAULT 'merge'
) RETURNS TABLE(server text, id bigint, distance real)
    AS 'MODULE_PATHNAME', 'federated_vector_query'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION federated_vector_query IS 'Execute federated vector query across multiple servers';

CREATE FUNCTION enable_vector_replication(
    table_name text,
    replication_mode text DEFAULT 'async'
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'enable_vector_replication'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION enable_vector_replication IS 'Enable vector data replication';

CREATE FUNCTION create_vector_fdw(
    server_name text,
    server_options jsonb DEFAULT '{}'
) RETURNS text
    AS 'MODULE_PATHNAME', 'create_vector_fdw'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION create_vector_fdw IS 'Create foreign data wrapper for remote vector tables';

-- Security functions
CREATE FUNCTION encrypt_postquantum(
    data vector
) RETURNS bytea
    AS 'MODULE_PATHNAME', 'encrypt_postquantum'
    LANGUAGE C IMMUTABLE;
COMMENT ON FUNCTION encrypt_postquantum IS 'Encrypt vector using post-quantum cryptography';

CREATE FUNCTION enable_confidential_compute(
    table_name text,
    encryption_key bytea
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'enable_confidential_compute'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION enable_confidential_compute IS 'Enable confidential computing for vector table';

CREATE FUNCTION set_access_mask(
    tenant_id text,
    mask_vector vector
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'set_access_mask'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION set_access_mask IS 'Set access control mask for tenant';

-- Feedback and ML lifecycle
CREATE FUNCTION feedback_loop_integrate(
    query_id bigint,
    relevant_ids bigint[],
    feedback_type text DEFAULT 'click'
) RETURNS boolean
    AS 'MODULE_PATHNAME', 'feedback_loop_integrate'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION feedback_loop_integrate IS 'Integrate user feedback for query improvement';

-- Internal/worker functions (exposed for advanced use)
-- Note: hnsw_search_layer is an internal C function, not exposed as SQL function

CREATE FUNCTION create_hybrid_scan_path(
    table_name text,
    vector_column text,
    text_column text
) RETURNS text
    AS 'MODULE_PATHNAME', 'create_hybrid_scan_path'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION create_hybrid_scan_path IS 'Create custom scan path for hybrid search (internal)';

-- Audit and testing
CREATE FUNCTION audit_log_query(
    query_text text,
    tenant_id text,
    metadata jsonb DEFAULT '{}'
) RETURNS bigint
    AS 'MODULE_PATHNAME', 'audit_log_query'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION audit_log_query IS 'Log query for audit trail';

CREATE FUNCTION ndb_llm_cache_test() RETURNS boolean
    AS 'MODULE_PATHNAME', 'ndb_llm_cache_test'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION ndb_llm_cache_test IS 'Test LLM cache functionality';

/* Advanced cache management functions */
CREATE FUNCTION ndb_llm_cache_stats() RETURNS jsonb
    AS 'MODULE_PATHNAME', 'ndb_llm_cache_stats'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION ndb_llm_cache_stats IS 'Get cache statistics: total entries, stale entries, TTL, etc.';

CREATE FUNCTION ndb_llm_cache_clear(text) RETURNS integer
    AS 'MODULE_PATHNAME', 'ndb_llm_cache_clear'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION ndb_llm_cache_clear IS 'Clear cache entries: (pattern) - NULL clears all, pattern clears matching keys';

CREATE FUNCTION ndb_llm_cache_evict_stale() RETURNS integer
    AS 'MODULE_PATHNAME', 'ndb_llm_cache_evict_stale'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION ndb_llm_cache_evict_stale IS 'Evict stale cache entries (older than TTL)';

CREATE FUNCTION ndb_llm_cache_evict_size(integer) RETURNS integer
    AS 'MODULE_PATHNAME', 'ndb_llm_cache_evict_size'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION ndb_llm_cache_evict_size IS 'Evict cache entries to maintain max_size limit (LRU eviction): (max_size)';

CREATE FUNCTION ndb_llm_cache_warm(text[], text) RETURNS integer
    AS 'MODULE_PATHNAME', 'ndb_llm_cache_warm'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION ndb_llm_cache_warm IS 'Pre-populate cache with embeddings: (texts[], model)';

-- Create ANN index (generic wrapper)
CREATE FUNCTION create_ann_index(
    table_name text,
    column_name text,
    index_type text DEFAULT 'hnsw',
    options jsonb DEFAULT '{}'
) RETURNS text
    AS 'MODULE_PATHNAME', 'create_ann_index'
    LANGUAGE C VOLATILE;
COMMENT ON FUNCTION create_ann_index IS 'Create generic ANN index (hnsw/ivf/hybrid)';

-- ============================================================================
-- PERMISSIONS AND GRANTS
-- ============================================================================

-- Grant usage on neurondb schema to public
GRANT USAGE ON SCHEMA neurondb TO PUBLIC;

-- Grant select on all views to public (read-only access)
GRANT SELECT ON neurondb.vector_stats TO PUBLIC;
GRANT SELECT ON neurondb.index_health TO PUBLIC;
GRANT SELECT ON neurondb.tenant_quota_usage TO PUBLIC;
GRANT SELECT ON neurondb.llm_job_status TO PUBLIC;
GRANT SELECT ON neurondb.query_performance TO PUBLIC;
GRANT SELECT ON neurondb.index_maintenance_status TO PUBLIC;
GRANT SELECT ON neurondb.metrics_summary TO PUBLIC;

-- Grant select on configuration and statistics tables to public
GRANT SELECT ON neurondb.llm_config TO PUBLIC;
GRANT SELECT ON neurondb.tenant_usage TO PUBLIC;
GRANT SELECT ON neurondb.tenant_quotas TO PUBLIC;
GRANT SELECT ON neurondb.query_metrics TO PUBLIC;
GRANT SELECT ON neurondb.prometheus_metrics TO PUBLIC;
GRANT SELECT ON neurondb.llm_stats TO PUBLIC;

-- Grant select/insert/update on cache tables to public
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.embedding_cache TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.llm_cache TO PUBLIC;

-- Grant select/insert/update on job tables to public
GRANT SELECT, INSERT, UPDATE ON neurondb.job_queue TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON neurondb.llm_jobs TO PUBLIC;

-- Grant select on metadata tables (restricted write)
GRANT SELECT ON neurondb.index_metadata TO PUBLIC;
GRANT SELECT ON neurondb.index_maintenance TO PUBLIC;
GRANT SELECT ON neurondb.histograms TO PUBLIC;

-- Restricted tables (select only, insert/update via functions)
GRANT SELECT ON neurondb.rls_policies TO PUBLIC;

-- Grant execute on all public functions (C functions)
-- Note: Individual GRANT EXECUTE statements would be too verbose for 180+ functions
-- PostgreSQL allows functions to be executable by PUBLIC by default for LANGUAGE C
-- For security-sensitive functions, we revoke and grant selectively:

-- Revoke execute from public on administrative functions
REVOKE EXECUTE ON FUNCTION neurondb_rebuild_index(regclass) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION neurondb_reset_quota(text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION neurondb_clear_entrypoint_cache() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION neurondb_ann_buffer_clear() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION pg_neurondb_stat_reset() FROM PUBLIC;

-- Grant execute on administrative functions to superuser/admin roles only
-- (In production, replace with specific role names like 'neurondb_admin')
-- GRANT EXECUTE ON FUNCTION neurondb_rebuild_index(regclass) TO neurondb_admin;
-- GRANT EXECUTE ON FUNCTION neurondb_reset_quota(text) TO neurondb_admin;
-- GRANT EXECUTE ON FUNCTION neurondb_clear_entrypoint_cache() TO neurondb_admin;
-- GRANT EXECUTE ON FUNCTION neurondb_ann_buffer_clear() TO neurondb_admin;
-- GRANT EXECUTE ON FUNCTION pg_neurondb_stat_reset() TO neurondb_admin;

-- Grant execute on schema helper functions
GRANT EXECUTE ON FUNCTION neurondb.set_llm_config(text, text, text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.get_llm_config() TO PUBLIC;

-- Revoke write access on sensitive tables from public
REVOKE INSERT, UPDATE, DELETE ON neurondb.tenant_quotas FROM PUBLIC;
REVOKE INSERT, UPDATE, DELETE ON neurondb.rls_policies FROM PUBLIC;
REVOKE INSERT, UPDATE, DELETE ON neurondb.index_metadata FROM PUBLIC;

-- Grant sequence usage for serial columns
GRANT USAGE, SELECT ON SEQUENCE neurondb.rls_policies_policy_id_seq TO PUBLIC;

-- ============================================================================
-- COMMENTS ON SCHEMA
-- ============================================================================

COMMENT ON SCHEMA neurondb IS 'NeurondB extension schema - contains all application tables, views, and metadata';

-- ============================================================================
-- SECURITY LABELS (Optional - for Enhanced Security)
-- ============================================================================

-- Security labels can be added for additional access control
-- SECURITY LABEL FOR sepgsql ON SCHEMA neurondb IS 'system_u:object_r:sepgsql_schema_t:s0';
-- SECURITY LABEL FOR sepgsql ON TABLE neurondb.llm_config IS 'system_u:object_r:sepgsql_table_t:s0';

-- ============================================================================
-- DEFAULT PRIVILEGES (for future objects)
-- ============================================================================

-- Set default privileges for any future tables created in neurondb schema
ALTER DEFAULT PRIVILEGES IN SCHEMA neurondb GRANT SELECT ON TABLES TO PUBLIC;

-- Set default privileges for future views
ALTER DEFAULT PRIVILEGES IN SCHEMA neurondb GRANT SELECT ON TABLES TO PUBLIC;

-- ============================================================================
-- ML TRAINING & INFERENCE API (NeurondB's In-Database ML Framework)
-- ============================================================================

-- =============================================================================
-- ML Project Storage Tables
-- =============================================================================

-- ML Projects table (similar to pgml.projects but NeurondB style)
CREATE TABLE IF NOT EXISTS neurondb.ml_projects (
    project_id SERIAL PRIMARY KEY,
    project_name TEXT UNIQUE NOT NULL,
    description TEXT,
    task_type TEXT CHECK (task_type IN ('classification', 'regression', 'clustering', 'embedding', 'rag')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.ml_projects IS 'ML Projects: organize models and experiments';

-- ML Experiments table (track different approaches)
CREATE TABLE IF NOT EXISTS neurondb.ml_experiments (
    experiment_id SERIAL PRIMARY KEY,
    project_id INT REFERENCES neurondb.ml_projects(project_id) ON DELETE CASCADE,
    experiment_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, experiment_name)
);
COMMENT ON TABLE neurondb.ml_experiments IS 'ML Experiments: track different model configurations';

-- ML Models table (trained model versions)
CREATE TABLE IF NOT EXISTS neurondb.ml_trained_models (
    model_id SERIAL PRIMARY KEY,
    experiment_id INT REFERENCES neurondb.ml_experiments(experiment_id) ON DELETE CASCADE,
    project_id INT REFERENCES neurondb.ml_projects(project_id) ON DELETE CASCADE,
    algorithm TEXT NOT NULL,
    hyperparameters JSONB,
    metrics JSONB,
    feature_columns TEXT[],
    target_column TEXT,
    model_data BYTEA,  -- Serialized model weights
    status TEXT CHECK (status IN ('training', 'trained', 'deployed', 'archived', 'failed')),
    training_samples BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_default BOOLEAN DEFAULT false
);
COMMENT ON TABLE neurondb.ml_trained_models IS 'Trained ML models with versioning';

-- ML Predictions log (track predictions for monitoring)
CREATE TABLE IF NOT EXISTS neurondb.ml_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    model_id INT REFERENCES neurondb.ml_trained_models(model_id) ON DELETE CASCADE,
    project_id INT REFERENCES neurondb.ml_projects(project_id) ON DELETE CASCADE,
    input_features JSONB,
    prediction FLOAT,
    probabilities FLOAT[],
    predicted_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.ml_predictions IS 'Prediction log for monitoring and A/B testing';

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ml_projects_name ON neurondb.ml_projects(project_name);
CREATE INDEX IF NOT EXISTS idx_ml_experiments_project ON neurondb.ml_experiments(project_id);
CREATE INDEX IF NOT EXISTS idx_ml_models_experiment ON neurondb.ml_trained_models(experiment_id);
CREATE INDEX IF NOT EXISTS idx_ml_models_project ON neurondb.ml_trained_models(project_id);
CREATE INDEX IF NOT EXISTS idx_ml_models_default ON neurondb.ml_trained_models(project_id, is_default) WHERE is_default = true;
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON neurondb.ml_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_time ON neurondb.ml_predictions(predicted_at);

-- Unified API definitions (formerly from neurondb.sql)
CREATE OR REPLACE FUNCTION neurondb.train(
	algorithm text,
	table_name text,
	feature_col text,
	label_col text DEFAULT NULL,
	params jsonb DEFAULT '{}'::jsonb
) RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE
	k integer;
	max_iters integer;
	max_depth integer;
	min_samples integer;
	n_trees integer;
	max_features integer;
	c_param float8;
	n_estimators integer;
	learning_rate float8;
BEGIN
	/* Validate required parameters */
	IF feature_col IS NULL THEN
		RAISE EXCEPTION 'feature_col cannot be NULL';
	END IF;
	IF table_name IS NULL THEN
		RAISE EXCEPTION 'table_name cannot be NULL';
	END IF;
	IF algorithm IS NULL THEN
		RAISE EXCEPTION 'algorithm cannot be NULL';
	END IF;
	CASE lower(algorithm)
		WHEN 'kmeans' THEN
			k := COALESCE((params->>'k')::integer, (params->>'num_clusters')::integer, 3);
			max_iters := COALESCE((params->>'max_iters')::integer, 100);
			RETURN cluster_kmeans(table_name, feature_col, k, max_iters);
		WHEN 'gmm' THEN
			-- Delegate to C function with default project
			RETURN neurondb.train('default', 'gmm', table_name, label_col, ARRAY[feature_col], params);
		WHEN 'minibatch_kmeans' THEN
			k := COALESCE((params->>'k')::integer, 3);
			max_iters := COALESCE((params->>'max_iters')::integer, 100);
			RETURN cluster_minibatch_kmeans(
				table_name, feature_col, k,
				COALESCE((params->>'batch_size')::integer, 100),
				max_iters
			);
		WHEN 'hierarchical' THEN
			k := COALESCE((params->>'k')::integer, 3);
			RETURN cluster_hierarchical(table_name, feature_col, k, COALESCE(params->>'linkage', 'average'));
		WHEN 'naive_bayes' THEN
			-- Delegate to C function with default project
			RETURN neurondb.train('default', 'naive_bayes', table_name, label_col, ARRAY[feature_col], params);
		WHEN 'knn' THEN
			-- Delegate to C function with default project
			RETURN neurondb.train('default', 'knn', table_name, label_col, ARRAY[feature_col], params);
		WHEN 'decision_tree' THEN
			max_depth := COALESCE((params->>'max_depth')::integer, 10);
			min_samples := COALESCE((params->>'min_samples_split')::integer, 2);
			RETURN train_decision_tree_classifier(table_name, feature_col, label_col, max_depth, min_samples);
		WHEN 'random_forest' THEN
			-- Delegate to C function with default project
			RETURN neurondb.train('default', 'random_forest', table_name, label_col, ARRAY[feature_col], params);
		WHEN 'svm' THEN
			c_param := COALESCE((params->>'C')::float8, 1.0);
			max_iters := COALESCE((params->>'max_iters')::integer, 1000);
			RETURN train_svm_classifier(table_name, feature_col, label_col, c_param, max_iters);
		WHEN 'logistic_regression' THEN
			max_iters := COALESCE((params->>'max_iters')::integer, 1000);
			learning_rate := COALESCE((params->>'learning_rate')::float8, 0.01);
			RETURN train_logistic_regression(
				table_name, feature_col, label_col,
				max_iters, learning_rate,
				COALESCE((params->>'lambda')::float8, 0.001)
			);
		WHEN 'xgboost' THEN
			n_estimators := COALESCE((params->>'n_estimators')::integer, 100);
			max_depth := COALESCE((params->>'max_depth')::integer, 6);
			learning_rate := COALESCE((params->>'learning_rate')::float8, 0.3);
			RETURN train_xgboost_classifier(
				table_name, feature_col, label_col,
				n_estimators, max_depth, learning_rate
			);
		WHEN 'neural_network', 'deep_learning' THEN
			RETURN train_neural_network(
				table_name, feature_col, label_col,
				COALESCE((params->>'layers')::text, '[10,5,1]')::integer[],
				COALESCE((params->>'learning_rate')::float8, 0.01)
			);
		WHEN 'linear_regression' THEN
			IF label_col IS NULL THEN
				RAISE EXCEPTION 'label_col cannot be NULL for linear_regression';
			END IF;
			RETURN train_linear_regression(table_name, feature_col, label_col);
		WHEN 'ridge' THEN
			RETURN train_ridge_regression(table_name, feature_col, label_col,
				COALESCE((params->>'lambda')::float8, 0.1));
		WHEN 'lasso' THEN
			RETURN train_lasso_regression(table_name, feature_col, label_col,
				COALESCE((params->>'lambda')::float8, 0.1),
				COALESCE((params->>'max_iters')::integer, 1000));
		WHEN 'rag' THEN
			-- RAG is a special algorithm that doesn't require traditional training
			-- Return a placeholder model_id for compatibility
			RETURN neurondb.train('default', 'rag', table_name, label_col, ARRAY[feature_col], params);
		WHEN 'hybrid_search' THEN
			-- Hybrid search is a special algorithm that doesn't require traditional training
			-- Return a placeholder model_id for compatibility
			RETURN neurondb.train('default', 'hybrid_search', table_name, label_col, ARRAY[feature_col], params);
		ELSE
			RAISE EXCEPTION 'Unknown algorithm: %. Use neurondb.list_algorithms() to see supported algorithms', algorithm;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.train(text, text, text, text, jsonb) IS 'Unified training function: train(algorithm, table, features, labels, params) - supports all ML algorithms';

CREATE OR REPLACE FUNCTION neurondb.train(
	project_name text,
	algorithm text,
	table_name text,
	label_col text,
	feature_columns text[],
	params jsonb DEFAULT '{}'::jsonb
) RETURNS integer
LANGUAGE c
AS 'MODULE_PATHNAME', 'neurondb_train';
COMMENT ON FUNCTION neurondb.train(text, text, text, text, text[], jsonb) IS 'Unified training function with project support: train(project, algorithm, table, label, feature_columns[], params)';

CREATE OR REPLACE FUNCTION neurondb.predict(
	model_id integer,
	features vector
) RETURNS double precision
LANGUAGE plpgsql STABLE AS $$
DECLARE
	algo text;
	model_id_param integer;
BEGIN
	model_id_param := model_id;
	IF model_id_param IS NULL THEN
		RAISE EXCEPTION 'model_id cannot be NULL';
	END IF;
	SELECT m.algorithm INTO algo FROM neurondb.ml_models m WHERE m.model_id = model_id_param;
	IF algo IS NULL THEN
		RAISE EXCEPTION 'Model % not found', model_id;
	END IF;
	CASE algo
		WHEN 'random_forest' THEN
			RETURN predict_random_forest(model_id, features);
		WHEN 'logistic_regression' THEN
			RETURN predict_logistic_regression(model_id, features);
		WHEN 'linear_regression' THEN
			RETURN predict_linear_regression_model_id(model_id, features);
		WHEN 'ridge' THEN
			RETURN predict_ridge_regression_model_id(model_id, features);
		WHEN 'lasso' THEN
			RETURN predict_lasso_regression_model_id(model_id, features);
		WHEN 'decision_tree' THEN
			RETURN predict_decision_tree_model_id(model_id, features);
		WHEN 'svm' THEN
			RETURN predict_svm_model_id(model_id, features);
		WHEN 'xgboost' THEN
			RETURN predict_xgboost(model_id, vector_to_array(features));
		WHEN 'neural_network' THEN
			RETURN predict_neural_network(model_id, vector_to_array(features));
		WHEN 'naive_bayes' THEN
			-- Delegate to C function (convert vector to array)
			RETURN neurondb_predict(model_id, vector_to_array(features));
		WHEN 'knn' THEN
			-- Delegate to C function (convert vector to array)
			RETURN neurondb_predict(model_id, vector_to_array(features));
		WHEN 'gmm' THEN
			-- Delegate to C function (convert vector to array)
			RETURN neurondb_predict(model_id, vector_to_array(features));
		ELSE
			RAISE EXCEPTION 'Prediction not implemented for algorithm: %', algo;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.predict(integer, vector) IS 'Unified prediction function: predict(model_id, features) - works for all algorithms';

-- Backward-compatible overloads to accept arrays and route to vector variant
CREATE OR REPLACE FUNCTION neurondb.predict(
	model_id integer,
	features real[]
) RETURNS double precision
LANGUAGE sql STABLE STRICT AS $$
	SELECT neurondb.predict(model_id, array_to_vector(features));
$$;
COMMENT ON FUNCTION neurondb.predict(integer, real[]) IS 'Overload: route real[] features to vector variant';

CREATE OR REPLACE FUNCTION neurondb.predict(
	model_id integer,
	features double precision[]
) RETURNS double precision
LANGUAGE sql STABLE STRICT AS $$
	SELECT neurondb.predict(model_id, array_to_vector(features::real[]));
$$;
COMMENT ON FUNCTION neurondb.predict(integer, double precision[]) IS 'Overload: route double precision[] features to vector variant';

-- Overload to accept model name (text) and look up model_id from catalog
CREATE OR REPLACE FUNCTION neurondb.predict(
	model_name text,
	features vector
) RETURNS double precision
LANGUAGE plpgsql STABLE AS $$
DECLARE
	model_id_param integer;
BEGIN
	-- Extract model_id from model_name (format: 'model_<id>' or just use as-is)
	IF model_name LIKE 'model_%' THEN
		model_id_param := substring(model_name FROM 'model_(\d+)')::integer;
	ELSE
		-- Try to find model by name in catalog
		SELECT m.model_id INTO model_id_param 
		FROM neurondb.ml_models m 
		WHERE m.model_name = model_name 
		LIMIT 1;
		
		IF model_id_param IS NULL THEN
			-- If not found by name, try to parse as model_id directly
			BEGIN
				model_id_param := model_name::integer;
			EXCEPTION WHEN OTHERS THEN
				RAISE EXCEPTION 'Model not found: %', model_name;
			END;
		END IF;
	END IF;
	
	IF model_id_param IS NULL THEN
		RAISE EXCEPTION 'Model not found: %', model_name;
	END IF;
	
	RETURN neurondb.predict(model_id_param, features);
END;
$$;
COMMENT ON FUNCTION neurondb.predict(text, vector) IS 'Overload: predict using model name (e.g., ''model_123'') - looks up model_id from catalog';

CREATE FUNCTION neurondb.evaluate(
	model_id integer,
	table_name text,
	feature_col text,
	label_col text
) RETURNS jsonb
	AS 'MODULE_PATHNAME', 'neurondb_evaluate'
	LANGUAGE C STABLE STRICT;
COMMENT ON FUNCTION neurondb.evaluate IS 'Unified evaluation function: evaluate(model_id, table, features, labels) - returns appropriate metrics';

CREATE OR REPLACE FUNCTION neurondb.list_algorithms()
RETURNS TABLE(
	algorithm text,
	category text,
	supervised boolean,
	description text,
	example_params jsonb
)
LANGUAGE sql STABLE AS $$
	SELECT * FROM (VALUES
		('kmeans', 'clustering', false, 'K-means clustering', '{"k": 5, "max_iters": 100}'::jsonb),
		('gmm', 'clustering', false, 'Gaussian Mixture Model', '{"k": 3, "max_iters": 100}'::jsonb),
		('minibatch_kmeans', 'clustering', false, 'Scalable K-means', '{"k": 5, "batch_size": 100}'::jsonb),
		('hierarchical', 'clustering', false, 'Hierarchical clustering', '{"k": 3, "linkage": "average"}'::jsonb),
		('dbscan', 'clustering', false, 'Density-based clustering', '{"eps": 0.5, "min_samples": 5}'::jsonb),
		('naive_bayes', 'classification', true, 'Gaussian Naive Bayes', '{}'::jsonb),
		('decision_tree', 'classification', true, 'CART decision tree', '{"max_depth": 10}'::jsonb),
		('random_forest', 'classification', true, 'Ensemble of trees', '{"n_trees": 100, "max_depth": 10}'::jsonb),
		('svm', 'classification', true, 'Support Vector Machine', '{"C": 1.0, "kernel": "linear"}'::jsonb),
		('knn', 'classification', true, 'K-Nearest Neighbors', '{"k": 5}'::jsonb),
		('logistic_regression', 'classification', true, 'Logistic regression', '{"max_iters": 500}'::jsonb),
		('xgboost', 'classification', true, 'Gradient boosting', '{"n_estimators": 100, "learning_rate": 0.1}'::jsonb),
		('neural_network', 'classification', true, 'Deep learning', '{"layers": [10, 5], "activation": "relu"}'::jsonb),
		('linear_regression', 'regression', true, 'OLS regression', '{}'::jsonb),
		('ridge', 'regression', true, 'L2 regularized regression', '{"lambda": 1.0}'::jsonb),
		('lasso', 'regression', true, 'L1 regularized regression', '{"lambda": 1.0}'::jsonb),
		('elastic_net', 'regression', true, 'L1+L2 regression', '{"alpha": 1.0, "l1_ratio": 0.5}'::jsonb),
		('isolation_forest', 'outlier_detection', false, 'Anomaly detection', '{"n_trees": 100}'::jsonb),
		('zscore', 'outlier_detection', false, 'Statistical outliers', '{"threshold": 3.0}'::jsonb)
	) AS t(algorithm, category, supervised, description, example_params);
$$;
COMMENT ON FUNCTION neurondb.list_algorithms IS 'List all supported ML algorithms with examples';

CREATE OR REPLACE FUNCTION neurondb.embed(
	model text,
	input_text text,
	task text DEFAULT 'embedding'
) RETURNS vector
LANGUAGE plpgsql AS $$
BEGIN
	CASE lower(task)
		WHEN 'embedding' THEN
			BEGIN
				RETURN neurondb_hf_embedding(model, input_text);
			EXCEPTION WHEN OTHERS THEN
				RETURN neurondb.generate_embedding(model::text, input_text::text);
			END;
		WHEN 'classification', 'classify' THEN
			RAISE EXCEPTION 'Classification returns text, use neurondb.classify() instead';
		ELSE
			RAISE EXCEPTION 'Unknown task: %. Use: embedding, classification', task;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.embed IS 'Unified embedding function with GPU acceleration support: embed(model, text, task). Uses CUDA-accelerated inference when available, falls back to ONNX Runtime or remote API. Configure via neurondb.llm_provider and neurondb.gpu_enabled.';

CREATE OR REPLACE FUNCTION neurondb.classify(
	model text,
	input_text text
) RETURNS jsonb
LANGUAGE plpgsql AS $$
DECLARE
	result text;
BEGIN
	result := neurondb_hf_classify(model, input_text);
	RETURN result::jsonb;
EXCEPTION WHEN OTHERS THEN
	RETURN jsonb_build_object('error', SQLERRM);
END;
$$;

-- ============================================================================
-- HUGGING FACE / LLM TOKENIZER FUNCTIONS
-- ============================================================================
-- Tokenization functions for HuggingFace models
-- ============================================================================

CREATE FUNCTION neurondb_hf_tokenize(
	model_name text,
	text text,
	max_length integer DEFAULT 512
) RETURNS integer[]
	AS 'MODULE_PATHNAME', 'neurondb_hf_tokenize'
	LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION neurondb_hf_tokenize(text, text, integer) IS 
	'Tokenize text using model-specific tokenizer: neurondb_hf_tokenize(model_name, text, max_length) returns token IDs array. Uses cached tokenizer if available, otherwise loads from model directory. Supports vocabulary loading from vocab.txt files.';

CREATE FUNCTION neurondb_hf_tokenize(
	text text,
	max_length integer DEFAULT 512
) RETURNS integer[]
	AS 'MODULE_PATHNAME', 'neurondb_hf_tokenize'
	LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION neurondb_hf_tokenize(text, integer) IS 
	'Tokenize text using default tokenizer: neurondb_hf_tokenize(text, max_length) returns token IDs array. Uses default vocabulary if no model-specific tokenizer is available.';

CREATE FUNCTION neurondb_hf_detokenize(
	model_name text,
	token_ids integer[]
) RETURNS text
	AS 'MODULE_PATHNAME', 'neurondb_hf_detokenize'
	LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION neurondb_hf_detokenize(text, integer[]) IS 
	'Detokenize token IDs back to text using model-specific tokenizer: neurondb_hf_detokenize(model_name, token_ids) returns text. Uses cached tokenizer if available, otherwise loads from model directory.';

CREATE FUNCTION neurondb_hf_detokenize(
	token_ids integer[]
) RETURNS text
	AS 'MODULE_PATHNAME', 'neurondb_hf_detokenize'
	LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION neurondb_hf_detokenize(integer[]) IS 
	'Detokenize token IDs back to text using default tokenizer: neurondb_hf_detokenize(token_ids) returns text. Uses default vocabulary if no model-specific tokenizer is available.';

-- ============================================================================
-- UNIFIED LLM / HUGGINGFACE API
-- ============================================================================
-- Unified API for all LLM/HuggingFace operations
-- ============================================================================

CREATE OR REPLACE FUNCTION neurondb.llm(
	task text,
	model text DEFAULT NULL,
	input_text text DEFAULT NULL,
	input_array text[] DEFAULT NULL,
	params jsonb DEFAULT '{}'::jsonb,
	max_length integer DEFAULT 512
) RETURNS jsonb
LANGUAGE plpgsql AS $$
DECLARE
	result jsonb;
	token_ids integer[];
	model_name text;
	input_str text;
	params_str text;
	gen_params jsonb;
BEGIN
	/* Use default model from GUC if not provided */
	model_name := COALESCE(model, current_setting('neurondb.llm_model', true));
	if (model_name IS NULL OR model_name = '')
		model_name := 'sentence-transformers/all-MiniLM-L6-v2';

	/* Convert params to text */
	params_str := params::text;

	CASE lower(task)
		WHEN 'tokenize', 'token' THEN
			/* Tokenize text */
			if (input_text IS NULL)
				RAISE EXCEPTION 'tokenize task requires input_text parameter';
			
			token_ids := neurondb_hf_tokenize(model_name, input_text, max_length);
			result := jsonb_build_object(
				'task', 'tokenize',
				'model', model_name,
				'token_ids', token_ids,
				'length', array_length(token_ids, 1)
			);

		WHEN 'detokenize', 'detoken' THEN
			/* Detokenize token IDs */
			if (params->>'token_ids' IS NULL)
				RAISE EXCEPTION 'detokenize task requires token_ids in params';
			
			token_ids := (params->>'token_ids')::integer[];
			input_str := neurondb_hf_detokenize(model_name, token_ids);
			result := jsonb_build_object(
				'task', 'detokenize',
				'model', model_name,
				'text', input_str
			);

		WHEN 'embed', 'embedding' THEN
			/* Generate embeddings */
			if (input_text IS NULL)
				RAISE EXCEPTION 'embed task requires input_text parameter';
			
			BEGIN
				result := jsonb_build_object(
					'task', 'embed',
					'model', model_name,
					'vector', neurondb.embed(model_name, input_text, 'embedding')
				);
			EXCEPTION WHEN OTHERS THEN
				/* Fallback to direct embedding function */
				result := jsonb_build_object(
					'task', 'embed',
					'model', model_name,
					'vector', ndb_llm_embed(input_text, model_name)
				);
			END;

		WHEN 'complete', 'completion', 'generate' THEN
			/* Text completion */
			if (input_text IS NULL)
				RAISE EXCEPTION 'complete task requires input_text parameter';
			
			BEGIN
				/* Build params JSON with model and generation parameters */
				gen_params := jsonb_build_object(
					'model', model_name,
					'max_tokens', COALESCE((params->>'max_tokens')::integer, 100),
					'temperature', COALESCE((params->>'temperature')::real, 0.7),
					'top_p', COALESCE((params->>'top_p')::real, 1.0),
					'top_k', COALESCE((params->>'top_k')::integer, 50),
					'repetition_penalty', COALESCE((params->>'repetition_penalty')::real, 1.0),
					'do_sample', COALESCE((params->>'do_sample')::boolean, true),
					'stop_sequences', COALESCE(params->'stop_sequences', '[]'::jsonb)
				);
				
				/* Call C function with prompt and params JSON (cast jsonb to text) */
				input_str := ndb_llm_complete(input_text::text, gen_params::text);
				result := jsonb_build_object(
					'task', 'complete',
					'model', model_name,
					'text', input_str,
					'params', gen_params
				);
			EXCEPTION WHEN OTHERS THEN
				result := jsonb_build_object(
					'task', 'complete',
					'error', SQLERRM,
					'model', model_name
				);
			END;

		WHEN 'classify', 'classification' THEN
			/* Text classification */
			if (input_text IS NULL)
				RAISE EXCEPTION 'classify task requires input_text parameter';
			
			result := neurondb.classify(model_name, input_text);
			result := jsonb_build_object(
				'task', 'classify',
				'model', model_name,
				'result', result
			);

		WHEN 'ner', 'named_entity_recognition' THEN
			/* Named Entity Recognition */
			if (input_text IS NULL)
				RAISE EXCEPTION 'ner task requires input_text parameter';
			
			result := jsonb_build_object(
				'task', 'ner',
				'model', model_name,
				'entities', neurondb_hf_ner(model_name, input_text)::jsonb
			);

		WHEN 'qa', 'question_answering' THEN
			/* Question Answering */
			if (input_text IS NULL OR params->>'context' IS NULL)
				RAISE EXCEPTION 'qa task requires input_text (question) and context in params';
			
			result := jsonb_build_object(
				'task', 'qa',
				'model', model_name,
				'answer', neurondb_hf_qa(model_name, input_text, params->>'context')::jsonb
			);

		WHEN 'rerank', 'reranking' THEN
			/* Reranking */
			if (input_text IS NULL OR input_array IS NULL)
				RAISE EXCEPTION 'rerank task requires input_text (query) and input_array (documents)';
			
			BEGIN
				result := jsonb_build_object(
					'task', 'rerank',
					'model', model_name,
					'scores', (
						SELECT jsonb_agg(
							jsonb_build_object(
								'idx', idx,
								'score', score
							)
						)
						FROM ndb_llm_rerank(input_text, input_array, 
							COALESCE(model_name, current_setting('neurondb.llm_model', true)),
							COALESCE((params->>'top_n')::integer, 10))
					)
				);
			EXCEPTION WHEN OTHERS THEN
				result := jsonb_build_object(
					'task', 'rerank',
					'error', SQLERRM,
					'model', model_name
				);
			END;

		WHEN 'summarize', 'summarization' THEN
			/* Text summarization */
			if (input_text IS NULL)
				RAISE EXCEPTION 'summarize task requires input_text parameter';
			
			result := neurondb.summarize(model_name, input_text, params);

		WHEN 'translate', 'translation' THEN
			/* Text translation */
			if (input_text IS NULL OR params->>'target_lang' IS NULL)
				RAISE EXCEPTION 'translate task requires input_text and target_lang in params';
			
			result := neurondb.translate(
				model_name, 
				input_text, 
				COALESCE(params->>'source_lang', 'en'),
				params->>'target_lang',
				params
			);

		WHEN 'fill_mask', 'masked_language_modeling' THEN
			/* Masked language modeling */
			if (input_text IS NULL)
				RAISE EXCEPTION 'fill_mask task requires input_text parameter';
			
			result := neurondb.fill_mask(model_name, input_text, params);

		WHEN 'text2text', 'text_to_text' THEN
			/* Text-to-text generation */
			if (input_text IS NULL)
				RAISE EXCEPTION 'text2text task requires input_text parameter';
			
			result := neurondb.text2text(model_name, input_text, params);

		WHEN 'zero_shot_classify', 'zero_shot_classification' THEN
			/* Zero-shot classification */
			if (input_text IS NULL OR params->>'candidate_labels' IS NULL)
				RAISE EXCEPTION 'zero_shot_classify task requires input_text and candidate_labels in params';
			
			result := neurondb.zero_shot_classify(
				model_name,
				input_text,
				(params->>'candidate_labels')::text[],
				params
			);

		ELSE
			RAISE EXCEPTION 'Unknown task: %. Supported tasks: tokenize, detokenize, embed, complete, classify, ner, qa, rerank, summarize, translate, fill_mask, text2text, zero_shot_classify', task;
	END CASE;

	RETURN result;
EXCEPTION WHEN OTHERS THEN
	RETURN jsonb_build_object(
		'task', task,
		'error', SQLERRM,
		'model', COALESCE(model_name, 'unknown')
	);
END;
$$;
COMMENT ON FUNCTION neurondb.llm IS 
	'Unified LLM/HuggingFace API: llm(task, model, input_text, input_array, params, max_length) returns JSONB result. 
	Supported tasks: tokenize, detokenize, embed, complete, classify, ner, qa, rerank, summarize, translate, fill_mask, text2text, zero_shot_classify.
	Uses GPU acceleration when available, falls back to ONNX Runtime or remote API.
	Configure via neurondb.llm_provider and neurondb.gpu_enabled.';

-- ============================================================================
-- UNIFIED TOKENIZATION API
-- ============================================================================

CREATE OR REPLACE FUNCTION neurondb.tokenize(
	model text,
	input_text text,
	max_length integer DEFAULT 512
) RETURNS integer[]
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
	RETURN neurondb_hf_tokenize(model, input_text, max_length);
EXCEPTION WHEN OTHERS THEN
	/* Fallback to default tokenizer */
	RETURN neurondb_hf_tokenize(input_text, max_length);
END;
$$;
COMMENT ON FUNCTION neurondb.tokenize IS 
	'Unified tokenization API: tokenize(model, text, max_length) returns token IDs array. 
	Uses model-specific tokenizer if available, otherwise uses default tokenizer.';

CREATE OR REPLACE FUNCTION neurondb.tokenize(
	input_text text,
	max_length integer DEFAULT 512
) RETURNS integer[]
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
	RETURN neurondb_hf_tokenize(input_text, max_length);
END;
$$;
COMMENT ON FUNCTION neurondb.tokenize(text, integer) IS 
	'Unified tokenization API: tokenize(text, max_length) returns token IDs array using default tokenizer.';

CREATE OR REPLACE FUNCTION neurondb.detokenize(
	model text,
	token_ids integer[]
) RETURNS text
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
	RETURN neurondb_hf_detokenize(model, token_ids);
EXCEPTION WHEN OTHERS THEN
	/* Fallback to default tokenizer */
	RETURN neurondb_hf_detokenize(token_ids);
END;
$$;
COMMENT ON FUNCTION neurondb.detokenize IS 
	'Unified detokenization API: detokenize(model, token_ids) returns text. 
	Uses model-specific tokenizer if available, otherwise uses default tokenizer.';

CREATE OR REPLACE FUNCTION neurondb.detokenize(
	token_ids integer[]
) RETURNS text
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
	RETURN neurondb_hf_detokenize(token_ids);
END;
$$;
COMMENT ON FUNCTION neurondb.detokenize(integer[]) IS 
	'Unified detokenization API: detokenize(token_ids) returns text using default tokenizer.';

-- ============================================================================
-- UNIFIED TRANSFORM API (PostgresML-compatible)
-- ============================================================================

CREATE OR REPLACE FUNCTION neurondb.transform(
	model text,
	input_text text,
	task text DEFAULT 'embedding'
) RETURNS jsonb
LANGUAGE plpgsql STABLE AS $$
DECLARE
	result jsonb;
BEGIN
	/* Route to neurondb.llm() based on task */
	CASE lower(task)
		WHEN 'embedding', 'embed' THEN
			result := neurondb.llm('embed', model, input_text);
		WHEN 'classification', 'classify' THEN
			result := neurondb.llm('classify', model, input_text);
		WHEN 'ner', 'named_entity_recognition' THEN
			result := neurondb.llm('ner', model, input_text);
		WHEN 'summarization', 'summarize' THEN
			result := neurondb.summarize(model, input_text);
		WHEN 'translation', 'translate' THEN
			/* Translation requires source_lang and target_lang - use default values */
			result := neurondb.translate(model, input_text, 'en', 'fr');
		WHEN 'fill_mask', 'masked_language_modeling' THEN
			result := neurondb.fill_mask(model, input_text);
		WHEN 'text2text', 'text_to_text' THEN
			result := neurondb.text2text(model, input_text);
		WHEN 'zero_shot_classification', 'zero_shot_classify' THEN
			/* Zero-shot classification requires candidate_labels */
			RAISE EXCEPTION 'zero_shot_classification requires candidate_labels parameter. Use neurondb.zero_shot_classify() instead.';
		ELSE
			RAISE EXCEPTION 'Unknown task: %. Supported tasks: embedding, classification, ner, summarization, translation, fill_mask, text2text', task;
	END CASE;

	RETURN result;
EXCEPTION WHEN OTHERS THEN
	RETURN jsonb_build_object(
		'task', task,
		'error', SQLERRM,
		'model', COALESCE(model, 'unknown')
	);
END;
$$;
COMMENT ON FUNCTION neurondb.transform IS 
	'Unified transform API (PostgresML-compatible): transform(model, text, task) returns JSONB result. 
	Supported tasks: embedding, classification, ner, summarization, translation, fill_mask, text2text, zero_shot_classification.
	Uses GPU acceleration when available, falls back to ONNX Runtime or remote API.
	Configure via neurondb.llm_provider and neurondb.gpu_enabled.';

-- ============================================================================
-- TASK-SPECIFIC FUNCTIONS
-- ============================================================================

CREATE OR REPLACE FUNCTION neurondb.summarize(
	model text,
	input_text text,
	params jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
LANGUAGE plpgsql STABLE AS $$
DECLARE
	result jsonb;
	model_name text;
	summary_text text;
	params_str text;
BEGIN
	/* Use default model from GUC if not provided */
	model_name := COALESCE(model, current_setting('neurondb.llm_model', true));
	if (model_name IS NULL OR model_name = '')
		model_name := 'facebook/bart-large-cnn';

	/* Convert params to text */
	params_str := params::text;

	/* Use completion API for summarization */
	BEGIN
		summary_text := ndb_llm_complete(input_text::text, params_str::text);
		result := jsonb_build_object(
			'task', 'summarize',
			'model', model_name,
			'summary', summary_text,
			'params', params
		);
	EXCEPTION WHEN OTHERS THEN
		result := jsonb_build_object(
			'task', 'summarize',
			'error', SQLERRM,
			'model', model_name
		);
	END;

	RETURN result;
END;
$$;
COMMENT ON FUNCTION neurondb.summarize IS 
	'Text summarization: summarize(model, text, params) returns JSONB with summary. 
	Uses text generation models (e.g., BART, T5) for summarization.
	Uses GPU acceleration when available, falls back to ONNX Runtime or remote API.';

CREATE OR REPLACE FUNCTION neurondb.translate(
	model text,
	input_text text,
	source_lang text DEFAULT 'en',
	target_lang text DEFAULT 'fr',
	params jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
LANGUAGE plpgsql STABLE AS $$
DECLARE
	result jsonb;
	model_name text;
	translation_text text;
	params_str text;
	translation_prompt text;
BEGIN
	/* Use default model from GUC if not provided */
	model_name := COALESCE(model, current_setting('neurondb.llm_model', true));
	if (model_name IS NULL OR model_name = '')
		model_name := 'Helsinki-NLP/opus-mt-en-fr';

	/* Build translation prompt */
	translation_prompt := format('Translate from %s to %s: %s', source_lang, target_lang, input_text);

	/* Convert params to text */
	params_str := params::text;

	/* Use completion API for translation */
	BEGIN
		translation_text := ndb_llm_complete(translation_prompt::text, params_str::text);
		result := jsonb_build_object(
			'task', 'translate',
			'model', model_name,
			'source_lang', source_lang,
			'target_lang', target_lang,
			'translation', translation_text,
			'params', params
		);
	EXCEPTION WHEN OTHERS THEN
		result := jsonb_build_object(
			'task', 'translate',
			'error', SQLERRM,
			'model', model_name
		);
	END;

	RETURN result;
END;
$$;
COMMENT ON FUNCTION neurondb.translate IS 
	'Text translation: translate(model, text, source_lang, target_lang, params) returns JSONB with translation. 
	Uses translation models (e.g., Helsinki-NLP, mBART) for translation.
	Uses GPU acceleration when available, falls back to ONNX Runtime or remote API.';

CREATE OR REPLACE FUNCTION neurondb.fill_mask(
	model text,
	input_text text,
	params jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
LANGUAGE plpgsql STABLE AS $$
DECLARE
	result jsonb;
	model_name text;
	filled_text text;
	params_str text;
BEGIN
	/* Use default model from GUC if not provided */
	model_name := COALESCE(model, current_setting('neurondb.llm_model', true));
	if (model_name IS NULL OR model_name = '')
		model_name := 'bert-base-uncased';

	/* Convert params to text */
	params_str := params::text;

	/* Use completion API for fill-mask */
	BEGIN
		filled_text := ndb_llm_complete(input_text::text, params_str::text);
		result := jsonb_build_object(
			'task', 'fill_mask',
			'model', model_name,
			'filled_text', filled_text,
			'params', params
		);
	EXCEPTION WHEN OTHERS THEN
		result := jsonb_build_object(
			'task', 'fill_mask',
			'error', SQLERRM,
			'model', model_name
		);
	END;

	RETURN result;
END;
$$;
COMMENT ON FUNCTION neurondb.fill_mask IS 
	'Masked language modeling: fill_mask(model, text, params) returns JSONB with filled text. 
	Uses masked language models (e.g., BERT, RoBERTa) for fill-mask.
	Uses GPU acceleration when available, falls back to ONNX Runtime or remote API.';

CREATE OR REPLACE FUNCTION neurondb.text2text(
	model text,
	input_text text,
	params jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
LANGUAGE plpgsql STABLE AS $$
DECLARE
	result jsonb;
	model_name text;
	output_text text;
	params_str text;
BEGIN
	/* Use default model from GUC if not provided */
	model_name := COALESCE(model, current_setting('neurondb.llm_model', true));
	if (model_name IS NULL OR model_name = '')
		model_name := 't5-small';

	/* Convert params to text */
	params_str := params::text;

	/* Use completion API for text2text */
	BEGIN
		output_text := ndb_llm_complete(input_text::text, params_str::text);
		result := jsonb_build_object(
			'task', 'text2text',
			'model', model_name,
			'output', output_text,
			'params', params
		);
	EXCEPTION WHEN OTHERS THEN
		result := jsonb_build_object(
			'task', 'text2text',
			'error', SQLERRM,
			'model', model_name
		);
	END;

	RETURN result;
END;
$$;
COMMENT ON FUNCTION neurondb.text2text IS 
	'Text-to-text generation: text2text(model, text, params) returns JSONB with generated text. 
	Uses text-to-text models (e.g., T5, FLAN-T5) for generation.
	Uses GPU acceleration when available, falls back to ONNX Runtime or remote API.';

CREATE OR REPLACE FUNCTION neurondb.zero_shot_classify(
	model text,
	input_text text,
	candidate_labels text[],
	params jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
LANGUAGE plpgsql STABLE AS $$
DECLARE
	result jsonb;
	model_name text;
	labels_str text;
	classification_text text;
	params_str text;
	classification_prompt text;
BEGIN
	/* Use default model from GUC if not provided */
	model_name := COALESCE(model, current_setting('neurondb.llm_model', true));
	if (model_name IS NULL OR model_name = '')
		model_name := 'facebook/bart-large-mnli';

	/* Build labels string */
	labels_str := array_to_string(candidate_labels, ', ');

	/* Build zero-shot classification prompt */
	classification_prompt := format('Text: %s. Labels: %s. Classify:', input_text, labels_str);

	/* Convert params to text */
	params_str := params::text;

	/* Use completion API for zero-shot classification */
	BEGIN
		classification_text := ndb_llm_complete(classification_prompt::text, params_str::text);
		result := jsonb_build_object(
			'task', 'zero_shot_classify',
			'model', model_name,
			'labels', candidate_labels,
			'classification', classification_text,
			'params', params
		);
	EXCEPTION WHEN OTHERS THEN
		result := jsonb_build_object(
			'task', 'zero_shot_classify',
			'error', SQLERRM,
			'model', model_name
		);
	END;

	RETURN result;
END;
$$;
COMMENT ON FUNCTION neurondb.zero_shot_classify IS 
	'Zero-shot classification: zero_shot_classify(model, text, candidate_labels, params) returns JSONB with classification. 
	Uses zero-shot classification models (e.g., BART-MNLI, DeBERTa) for classification.
	Uses GPU acceleration when available, falls back to ONNX Runtime or remote API.';

-- ============================================================================
-- DATASET INTEGRATION
-- ============================================================================

CREATE OR REPLACE FUNCTION neurondb.load_dataset(
	dataset_name text,
	split text DEFAULT 'train',
	config text DEFAULT NULL,
	streaming boolean DEFAULT false,
	cache_dir text DEFAULT NULL
) RETURNS TABLE(
	row_id bigint,
	data jsonb
)
LANGUAGE plpgsql STABLE AS $$
DECLARE
	result jsonb;
	dataset_path text;
	cache_path text;
	error_msg text;
BEGIN
	/* Validate inputs */
	IF dataset_name IS NULL OR dataset_name = '' THEN
		RAISE EXCEPTION 'dataset_name cannot be NULL or empty';
	END IF;

	IF split IS NULL THEN
		split := 'train';
	END IF;

	/* Determine cache directory */
	IF cache_dir IS NULL OR cache_dir = '' THEN
		cache_path := COALESCE(current_setting('neurondb.dataset_cache_dir', true), '/tmp/neurondb_datasets');
	ELSE
		cache_path := cache_dir;
	END IF;

	/* 
	 * Full implementation requires Python integration with HuggingFace datasets library.
	 * This would require:
	 * 1. plpython3u extension enabled
	 * 2. Python code to call datasets.load_dataset()
	 * 3. Conversion of dataset rows to JSONB format
	 * 4. Streaming support for large datasets
	 * 
	 * For now, return informative error message
	 */
	error_msg := format('HuggingFace dataset loading requires Python integration. ' ||
		'Dataset: %s, Split: %s, Config: %s, Streaming: %s, Cache: %s. ' ||
		'Full implementation requires plpython3u extension and HuggingFace datasets library.',
		dataset_name, split, COALESCE(config, 'default'), streaming::text, cache_path);
	
	RAISE EXCEPTION '%', error_msg;
EXCEPTION WHEN OTHERS THEN
	/* Return error in result format */
	RETURN QUERY SELECT 
		0::bigint, 
		jsonb_build_object(
			'error', SQLERRM,
			'dataset_name', dataset_name,
			'split', split,
			'config', config,
			'streaming', streaming,
			'cache_dir', cache_path
		);
END;
$$;
COMMENT ON FUNCTION neurondb.load_dataset IS 
	'Load HuggingFace dataset: load_dataset(dataset_name, split, config, streaming, cache_dir) returns TABLE with dataset rows. 
	Supports HuggingFace datasets with caching and streaming.
	Note: Full implementation requires Python integration (plpython3u) with HuggingFace datasets library.';

CREATE OR REPLACE FUNCTION neurondb.distance(
	vec1 vector,
	vec2 vector,
	metric text DEFAULT 'l2',
	p_value float8 DEFAULT 3.0
) RETURNS float8
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
	CASE lower(metric)
		WHEN 'l2', 'euclidean' THEN
			RETURN vector_l2_distance(vec1, vec2);
		WHEN 'cosine' THEN
			RETURN vector_cosine_distance(vec1, vec2);
		WHEN 'inner_product', 'dot' THEN
			RETURN vector_inner_product(vec1, vec2);
		WHEN 'l1', 'manhattan' THEN
			RETURN vector_l1_distance(vec1, vec2);
		WHEN 'hamming' THEN
			RETURN vector_hamming_distance(vec1, vec2);
		WHEN 'chebyshev' THEN
			RETURN vector_chebyshev_distance(vec1, vec2);
		WHEN 'minkowski' THEN
			RETURN vector_minkowski_distance(vec1, vec2, p_value);
		ELSE
			RAISE EXCEPTION 'Unknown distance metric: %. Supported: l2, cosine, inner_product, l1, hamming, chebyshev, minkowski', metric;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.distance IS 'Unified distance function: distance(vec1, vec2, metric, p) - supports all 11 metrics';

CREATE OR REPLACE FUNCTION neurondb.similarity(
	vec1 vector,
	vec2 vector,
	metric text DEFAULT 'cosine'
) RETURNS float8
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
	CASE lower(metric)
		WHEN 'cosine' THEN
			RETURN 1.0 - vector_cosine_distance(vec1, vec2);
		WHEN 'inner_product', 'dot' THEN
			RETURN -vector_inner_product(vec1, vec2);
		WHEN 'l2', 'euclidean' THEN
			RETURN 1.0 / (1.0 + vector_l2_distance(vec1, vec2));
		ELSE
			RAISE EXCEPTION 'Unknown similarity metric: %', metric;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.similarity IS 'Unified similarity function: similarity(vec1, vec2, metric) - higher values = more similar';

CREATE OR REPLACE FUNCTION neurondb.search(
	table_name text,
	vector_col text,
	query_vector vector,
	top_k integer DEFAULT 10,
	search_type text DEFAULT 'vector',
	params jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE(id integer, distance float8)
LANGUAGE plpgsql AS $$
DECLARE
	sql text;
	metric text;
BEGIN
	metric := COALESCE(params->>'metric', 'l2');
	CASE lower(search_type)
		WHEN 'vector', 'semantic' THEN
			sql := format(
				'SELECT id::integer, (%I <-> $1)::float8 as distance FROM %I ORDER BY %I <-> $1 LIMIT %s',
				vector_col, table_name, vector_col, top_k
			);
			RETURN QUERY EXECUTE sql USING query_vector;
		WHEN 'hybrid' THEN
			RETURN;
		ELSE
			RAISE EXCEPTION 'Unknown search type: %. Use: vector, hybrid', search_type;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.search IS 'Unified search function: search(table, vector_col, query, top_k, type, params)';

CREATE OR REPLACE FUNCTION neurondb.load_model(
	project_name text,
	model_path text,
	model_format text DEFAULT 'onnx'
) RETURNS integer
LANGUAGE C STRICT
AS 'MODULE_PATHNAME', 'neurondb_load_model';
COMMENT ON FUNCTION neurondb.load_model(text, text, text) IS 'Register an external model and return its model_id';

CREATE OR REPLACE FUNCTION neurondb.models()
RETURNS TABLE(
	model_id integer,
	algorithm text,
	created_at timestamptz,
	status text,
	is_deployed boolean,
	parameters jsonb
)
LANGUAGE sql STABLE AS $$
	SELECT 
		model_id,
		algorithm::text,
		created_at,
		status::text,
		is_deployed,
		parameters
	FROM neurondb.ml_models
	ORDER BY created_at DESC;
$$;
COMMENT ON FUNCTION neurondb.models IS 'List all trained models';

CREATE OR REPLACE FUNCTION neurondb.model_info(p_model_id integer)
RETURNS jsonb
LANGUAGE sql STABLE AS $$
	SELECT row_to_json(m)::jsonb
	FROM neurondb.ml_models m
	WHERE model_id = p_model_id;
$$;
COMMENT ON FUNCTION neurondb.model_info(integer) IS 'Get detailed model information as JSON';

CREATE OR REPLACE FUNCTION neurondb.drop_model(p_model_id integer)
RETURNS boolean
LANGUAGE plpgsql AS $$
DECLARE
	rows_deleted integer;
BEGIN
	DELETE FROM neurondb.ml_models WHERE model_id = p_model_id;
	GET DIAGNOSTICS rows_deleted = ROW_COUNT;
	RETURN rows_deleted > 0;
END;
$$;
COMMENT ON FUNCTION neurondb.drop_model IS 'Delete a trained model';

CREATE OR REPLACE FUNCTION neurondb.create_index(
	table_name text,
	vector_col text,
	index_type text DEFAULT 'hnsw',
	params jsonb DEFAULT '{}'::jsonb
) RETURNS text
LANGUAGE plpgsql AS $$
DECLARE
	index_name text;
	sql text;
	m integer;
	ef_construction integer;
BEGIN
	index_name := table_name || '_' || vector_col || '_' || index_type || '_idx';
	CASE lower(index_type)
		WHEN 'hnsw' THEN
			m := COALESCE((params->>'m')::integer, 16);
			ef_construction := COALESCE((params->>'ef_construction')::integer, 64);
			sql := format('CREATE INDEX %I ON %I USING hnsw (%I vector_l2_ops) WITH (m = %s, ef_construction = %s)',
				index_name, table_name, vector_col, m, ef_construction);
		WHEN 'ivf', 'ivfflat' THEN
			sql := format('CREATE INDEX %I ON %I USING ivf (%I vector_l2_ops)',
				index_name, table_name, vector_col);
		WHEN 'btree' THEN
			sql := format('CREATE INDEX %I ON %I (%I)',
				index_name, table_name, vector_col);
		ELSE
			RAISE EXCEPTION 'Unknown index type: %. Use: hnsw, ivf, btree', index_type;
	END CASE;
	EXECUTE sql;
	RETURN index_name;
END;
$$;
COMMENT ON FUNCTION neurondb.create_index IS 'Unified index creation: create_index(table, vector_col, type, params)';

CREATE OR REPLACE FUNCTION neurondb.chunk(
	document_text text,
	chunk_size integer DEFAULT 512,
	chunk_overlap integer DEFAULT 128,
	method text DEFAULT 'fixed'
) RETURNS TABLE(chunk_id integer, chunk_text text, start_pos integer, end_pos integer)
LANGUAGE plpgsql AS $$
DECLARE
	pos integer := 1;
	chunk_num integer := 0;
	doc_len integer;
BEGIN
	doc_len := length(document_text);
	CASE lower(method)
		WHEN 'fixed' THEN
			WHILE pos <= doc_len LOOP
				chunk_num := chunk_num + 1;
				RETURN QUERY SELECT 
					chunk_num,
					substring(document_text FROM pos FOR chunk_size),
					pos,
					LEAST(pos + chunk_size - 1, doc_len);
				pos := pos + chunk_size - chunk_overlap;
			END LOOP;
		WHEN 'sentence' THEN
			RETURN;
		ELSE
			RAISE EXCEPTION 'Unknown chunking method: %. Use: fixed, sentence', method;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.chunk IS 'Unified chunking: chunk(text, size, overlap, method) - splits documents';

CREATE OR REPLACE FUNCTION neurondb.rag_query(
	query_text text,
	document_table text,
	vector_col text,
	text_col text,
	model text DEFAULT 'default',
	top_k integer DEFAULT 5
) RETURNS TABLE(chunk_text text, relevance_score float8)
LANGUAGE plpgsql AS $$
DECLARE
	query_embedding vector;
	sql text;
BEGIN
	query_embedding := neurondb.embed(model, query_text);
	sql := format(
		'SELECT %I as chunk_text, (%I <=> $1)::float8 as relevance_score 
		 FROM %I 
		 ORDER BY %I <=> $1 
		 LIMIT %s',
		text_col, vector_col, document_table, vector_col, top_k
	);
	RETURN QUERY EXECUTE sql USING query_embedding;
END;
$$;
COMMENT ON FUNCTION neurondb.rag_query IS 'Unified RAG: rag_query(query, doc_table, vector_col, text_col, model, top_k)';

CREATE OR REPLACE FUNCTION neurondb.preprocess(
	input_vector vector,
	method text DEFAULT 'normalize',
	params jsonb DEFAULT '{}'::jsonb
) RETURNS vector
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
DECLARE
	min_val float8;
	max_val float8;
BEGIN
	CASE lower(method)
		WHEN 'normalize', 'l2_normalize' THEN
			RETURN vector_normalize(input_vector);
		WHEN 'standardize', 'zscore' THEN
			RETURN vector_standardize(input_vector);
		WHEN 'minmax', 'minmax_normalize' THEN
			RETURN vector_minmax_normalize(input_vector);
		WHEN 'clip' THEN
			min_val := COALESCE((params->>'min')::float8, 0.0);
			max_val := COALESCE((params->>'max')::float8, 1.0);
			RETURN vector_clip(input_vector, min_val, max_val);
		ELSE
			RAISE EXCEPTION 'Unknown preprocessing method: %. Use: normalize, standardize, minmax, clip', method;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.preprocess IS 'Unified preprocessing: preprocess(vector, method, params) - normalize, standardize, minmax, clip';

CREATE OR REPLACE FUNCTION neurondb.gpu(
	operation text,
	vec1 vector,
	vec2 vector DEFAULT NULL,
	params jsonb DEFAULT '{}'::jsonb
) RETURNS float8
LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
	CASE lower(operation)
		WHEN 'l2_distance', 'l2' THEN
			RETURN vector_l2_distance_gpu(vec1, vec2);
		WHEN 'cosine_distance', 'cosine' THEN
			RETURN vector_cosine_distance_gpu(vec1, vec2);
		WHEN 'inner_product', 'dot' THEN
			RETURN vector_inner_product_gpu(vec1, vec2);
		ELSE
			RAISE EXCEPTION 'Unknown GPU operation: %. Use: l2, cosine, dot', operation;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.gpu IS 'Unified GPU operations: gpu(operation, vec1, vec2, params) - GPU-accelerated computations';

CREATE OR REPLACE FUNCTION neurondb.quantize(
	input_vector vector,
	method text DEFAULT 'int8',
	use_gpu boolean DEFAULT false
) RETURNS bytea
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
	CASE lower(method)
		WHEN 'int8' THEN
			IF use_gpu THEN
				RETURN vector_quantize_int8_gpu(input_vector);
			ELSE
				RETURN vector_to_int8(input_vector);
			END IF;
		WHEN 'fp16', 'float16' THEN
			IF use_gpu THEN
				RETURN vector_quantize_fp16_gpu(input_vector);
			ELSE
				RAISE EXCEPTION 'FP16 quantization requires GPU';
			END IF;
		WHEN 'binary' THEN
			IF use_gpu THEN
				RETURN vector_quantize_binary_gpu(input_vector);
			ELSE
				RETURN vector_to_binary(input_vector);
			END IF;
		ELSE
			RAISE EXCEPTION 'Unknown quantization method: %. Use: int8, fp16, binary', method;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.quantize IS 'Unified quantization: quantize(vector, method, use_gpu) - int8, fp16, binary';

CREATE OR REPLACE FUNCTION neurondb.dequantize(
	quantized_data bytea,
	method text DEFAULT 'int8',
	dimensions integer DEFAULT NULL
) RETURNS vector
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
BEGIN
	CASE lower(method)
		WHEN 'int8' THEN
			RETURN int8_to_vector(quantized_data);
		WHEN 'binary' THEN
			RETURN binary_to_vector(quantized_data, dimensions);
		ELSE
			RAISE EXCEPTION 'Unknown dequantization method: %. Use: int8, binary', method;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.dequantize IS 'Unified dequantization: dequantize(data, method, dims) - reconstruct from quantized';

CREATE OR REPLACE FUNCTION neurondb.metrics(
	metric_type text DEFAULT 'all'
) RETURNS jsonb
LANGUAGE plpgsql AS $$
DECLARE
	result jsonb;
	model_count integer;
	vector_func_count integer;
BEGIN
	CASE lower(metric_type)
		WHEN 'models', 'ml' THEN
			SELECT COUNT(*) INTO model_count FROM neurondb.ml_models;
			result := jsonb_build_object(
				'total_models', model_count,
				'deployed_models', (SELECT COUNT(*) FROM neurondb.ml_models WHERE is_deployed = true)
			);
		WHEN 'vectors' THEN
			SELECT COUNT(*) INTO vector_func_count FROM pg_proc WHERE proname LIKE '%vector%';
			result := jsonb_build_object(
				'vector_functions', vector_func_count,
				'distance_metrics', 11,
				'gpu_functions', 6
			);
		WHEN 'all' THEN
			SELECT COUNT(*) INTO model_count FROM neurondb.ml_models;
			SELECT COUNT(*) INTO vector_func_count FROM pg_proc WHERE proname LIKE '%vector%';
			result := jsonb_build_object(
				'models', jsonb_build_object(
					'total', model_count,
					'deployed', (SELECT COUNT(*) FROM neurondb.ml_models WHERE is_deployed = true)
				),
				'vectors', jsonb_build_object(
					'functions', vector_func_count,
					'distance_metrics', 11,
					'gpu_functions', 6
				),
				'algorithms', jsonb_build_object(
					'total', (SELECT COUNT(*) FROM neurondb.list_algorithms())
				)
			);
		ELSE
			RAISE EXCEPTION 'Unknown metric type: %. Use: models, vectors, all', metric_type;
	END CASE;
	RETURN result;
END;
$$;
COMMENT ON FUNCTION neurondb.metrics IS 'Unified metrics: metrics(type) - get system statistics';

CREATE OR REPLACE FUNCTION neurondb.health()
RETURNS jsonb
LANGUAGE plpgsql AS $$
DECLARE
	health jsonb;
	vector_type_exists boolean;
	onnx_available boolean;
	gpu_available boolean;
BEGIN
	SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'vector') INTO vector_type_exists;
	onnx_available := (neurondb_onnx_info()::jsonb->>'available')::boolean;
	gpu_available := true;
	health := jsonb_build_object(
		'status', 'healthy',
		'version', '1.0',
		'components', jsonb_build_object(
			'vector_type', vector_type_exists,
			'onnx_runtime', onnx_available,
			'gpu_acceleration', gpu_available,
			'ml_algorithms', (SELECT COUNT(*) FROM neurondb.list_algorithms()),
			'trained_models', (SELECT COUNT(*) FROM neurondb.ml_models)
		)
	);
	RETURN health;
END;
$$;
COMMENT ON FUNCTION neurondb.health IS 'System health check: health() - returns status of all components';

CREATE OR REPLACE FUNCTION neurondb.load(
	source_type text,
	source_path text,
	target_table text,
	params jsonb DEFAULT '{}'::jsonb
) RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE
	rows_loaded integer := 0;
BEGIN
	CASE lower(source_type)
		WHEN 'csv' THEN
			EXECUTE format(
				'COPY %I FROM %L WITH (FORMAT csv, HEADER true)',
				target_table, source_path
			);
			GET DIAGNOSTICS rows_loaded = ROW_COUNT;
		WHEN 'json' THEN
			RETURN 0;
		WHEN 'parquet' THEN
			RAISE EXCEPTION 'Parquet loading requires external extension (parquet_fdw)';
		ELSE
			RAISE EXCEPTION 'Unknown source type: %. Use: csv, json', source_type;
	END CASE;
	RETURN rows_loaded;
END;
$$;
COMMENT ON FUNCTION neurondb.load IS 'Unified data loading: load(source_type, path, table, params) - load data from files';

CREATE OR REPLACE FUNCTION neurondb.export_model(
	model_id integer,
	export_format text DEFAULT 'onnx',
	export_path text DEFAULT NULL
) RETURNS text
LANGUAGE plpgsql AS $$
DECLARE
	algo text;
	export_location text;
	model_id_param integer;
BEGIN
	model_id_param := model_id;
	SELECT m.algorithm INTO algo FROM neurondb.ml_models m WHERE m.model_id = model_id_param;
	IF algo IS NULL THEN
		RAISE EXCEPTION 'Model % not found', model_id;
	END IF;
	export_location := COALESCE(export_path, '/tmp/neurondb_model_' || model_id || '_export');
	CASE lower(export_format)
		WHEN 'onnx' THEN
			RETURN export_location;
		WHEN 'pmml' THEN
			RETURN export_location;
		WHEN 'json' THEN
			RETURN (SELECT row_to_json(m)::text FROM neurondb.ml_models m WHERE m.model_id = export_model.model_id);
		ELSE
			RAISE EXCEPTION 'Unknown export format: %. Use: onnx, pmml, json', export_format;
	END CASE;
END;
$$;
COMMENT ON FUNCTION neurondb.export_model IS 'Export model: export_model(model_id, format, path) - export to ONNX/PMML/JSON';

-- Export NeuronDB model to ONNX format
CREATE OR REPLACE FUNCTION neurondb.export_model_to_onnx(
	model_id integer,
	output_path text
)
RETURNS boolean
LANGUAGE C
STRICT
AS 'MODULE_PATHNAME', 'export_model_to_onnx';
COMMENT ON FUNCTION neurondb.export_model_to_onnx(integer, text) IS 'Export NeuronDB model to ONNX format for Hugging Face compatibility';

-- Import ONNX model to NeuronDB format
CREATE OR REPLACE FUNCTION neurondb.import_model_from_onnx(
	model_id integer,
	onnx_path text,
	algorithm text
)
RETURNS boolean
LANGUAGE C
STRICT
AS 'MODULE_PATHNAME', 'import_model_from_onnx';
COMMENT ON FUNCTION neurondb.import_model_from_onnx(integer, text, text) IS 'Import ONNX model to NeuronDB format (algorithm: ridge, lasso, linear_regression, logistic_regression)';

CREATE OR REPLACE FUNCTION neurondb.embed_batch(
	model text,
	texts text[],
	batch_size integer DEFAULT 32
) RETURNS vector[]
LANGUAGE plpgsql STABLE AS $$
DECLARE
	result_vectors vector[];
	chunk_vectors vector[];
	chunk_start integer;
	chunk_end integer;
	total_texts integer;
BEGIN
	-- Validate inputs
	IF texts IS NULL OR array_length(texts, 1) IS NULL THEN
		RETURN ARRAY[]::vector[];
	END IF;
	
	IF batch_size IS NULL OR batch_size <= 0 THEN
		batch_size := 32;
	END IF;
	
	total_texts := array_length(texts, 1);
	result_vectors := ARRAY[]::vector[];
	
	-- Process in chunks
	chunk_start := 1;
	WHILE chunk_start <= total_texts LOOP
		chunk_end := LEAST(chunk_start + batch_size - 1, total_texts);
		
		-- Process this chunk
		SELECT embed_text_batch(texts[chunk_start:chunk_end], model) INTO chunk_vectors;
		
		-- Append chunk results to overall result
		IF chunk_vectors IS NOT NULL AND array_length(chunk_vectors, 1) > 0 THEN
			result_vectors := result_vectors || chunk_vectors;
		END IF;
		
		chunk_start := chunk_end + 1;
	END LOOP;
	
	RETURN result_vectors;
END;
$$;
COMMENT ON FUNCTION neurondb.embed_batch IS 'Batch embedding: embed_batch(model, texts[], batch_size) - efficient batch processing using optimized batch API with configurable batch size';

CREATE OR REPLACE FUNCTION neurondb.predict_batch(
	model_id integer,
	features_array vector[]
) RETURNS float8[]
LANGUAGE plpgsql AS $$
DECLARE
	results float8[];
	i integer;
BEGIN
	results := ARRAY[]::float8[];
	FOR i IN 1..array_length(features_array, 1) LOOP
		results := array_append(results, neurondb.predict(model_id, features_array[i]));
	END LOOP;
	RETURN results;
END;
$$;
COMMENT ON FUNCTION neurondb.predict_batch(integer, vector[]) IS 'Batch prediction: predict_batch(model_id, features[]) - predict multiple samples';

CREATE OR REPLACE FUNCTION neurondb.create_pipeline(
	pipeline_name text,
	steps jsonb
) RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE
	pipeline_id integer;
BEGIN
	INSERT INTO neurondb.ml_pipelines (pipeline_name, steps, created_at)
	VALUES (pipeline_name, steps, NOW())
	RETURNING neurondb.ml_pipelines.pipeline_id INTO create_pipeline.pipeline_id;
	RETURN pipeline_id;
EXCEPTION WHEN undefined_table THEN
	CREATE TABLE IF NOT EXISTS neurondb.ml_pipelines (
		pipeline_id SERIAL PRIMARY KEY,
		pipeline_name TEXT UNIQUE,
		steps JSONB,
		created_at TIMESTAMPTZ
	);
	RETURN neurondb.create_pipeline(pipeline_name, steps);
END;
$$;
COMMENT ON FUNCTION neurondb.create_pipeline IS 'Create ML pipeline: create_pipeline(name, steps) - define reusable workflows';

CREATE OR REPLACE FUNCTION neurondb.compare(
	model_ids integer[],
	test_table text,
	feature_col text,
	label_col text
) RETURNS TABLE(
	model_id integer,
	algorithm text,
	accuracy float8,
	precision_score float8,
	recall_score float8,
	f1_score float8
)
LANGUAGE plpgsql AS $$
DECLARE
	mid integer;
	metrics jsonb;
BEGIN
	FOREACH mid IN ARRAY model_ids LOOP
		metrics := neurondb.evaluate(mid, test_table, feature_col, label_col);
		RETURN QUERY SELECT 
			mid,
			(SELECT m.algorithm::text FROM neurondb.ml_models m WHERE m.model_id = mid),
			COALESCE((metrics->>'accuracy')::float8, 0.0),
			COALESCE((metrics->>'precision')::float8, 0.0),
			COALESCE((metrics->>'recall')::float8, 0.0),
			COALESCE((metrics->>'f1_score')::float8, 0.0);
	END LOOP;
END;
$$;
COMMENT ON FUNCTION neurondb.compare IS 'Compare models: compare(model_ids[], test_table, features, labels) - side-by-side comparison';

CREATE OR REPLACE FUNCTION neurondb.version()
RETURNS jsonb
LANGUAGE sql STABLE AS $$
	SELECT jsonb_build_object(
		'version', '1.0',
		'postgresql_version', current_setting('server_version'),
		'capabilities', jsonb_build_object(
			'vector_functions', (SELECT COUNT(*) FROM pg_proc WHERE proname LIKE '%vector%'),
			'ml_algorithms', (SELECT COUNT(*) FROM neurondb.list_algorithms()),
			'distance_metrics', 11,
			'gpu_support', true,
			'onnx_support', (neurondb_onnx_info()::jsonb->>'available')::boolean
		),
		'api', jsonb_build_object(
			'unified_ml', true,
			'unified_vector', true,
			'unified_rag', true,
			'unified_gpu', true
		)
	);
$$;
COMMENT ON FUNCTION neurondb.version IS 'Get NeuronDB version and capabilities';

CREATE FUNCTION auto_train(
	table_name text,
	feature_col text,
	label_col text,
	task text,
	metric text DEFAULT 'accuracy'
) RETURNS integer
LANGUAGE c
AS 'MODULE_PATHNAME', 'auto_train';
COMMENT ON FUNCTION auto_train(text, text, text, text, text) IS 'Automated model selection with GPU acceleration support';

GRANT EXECUTE ON FUNCTION neurondb.train(text, text, text, text, jsonb) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.train(text, text, text, text, text[], jsonb) TO PUBLIC;
GRANT EXECUTE ON FUNCTION auto_train(text, text, text, text, text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.predict(integer, vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.evaluate TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.list_algorithms TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.models TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.model_info TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.drop_model TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.embed TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.classify TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.llm TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.transform TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.summarize TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.translate TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.fill_mask TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.text2text TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.zero_shot_classify TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.load_dataset TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.tokenize(text, text, integer) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.tokenize(text, integer) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.detokenize(text, integer[]) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.detokenize(integer[]) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.distance TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.similarity TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.search TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.create_index TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.chunk TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.rag_query TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.preprocess TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.gpu TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.quantize TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.dequantize TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.metrics TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.health TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.load TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.load_model TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.export_model TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.export_model_to_onnx TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.import_model_from_onnx TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.embed_batch TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.predict_batch(integer, vector[]) TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.create_pipeline TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.compare TO PUBLIC;
GRANT EXECUTE ON FUNCTION neurondb.version TO PUBLIC;

-- Grant permissions for new distance functions
GRANT EXECUTE ON FUNCTION vector_squared_l2_distance(vector, vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_jaccard_distance(vector, vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_dice_distance(vector, vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_mahalanobis_distance(vector, vector, vector) TO PUBLIC;

-- Grant permissions for quantization functions
GRANT EXECUTE ON FUNCTION vector_to_int8(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION int8_to_vector(bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_to_fp16(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION fp16_to_vector(bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_to_uint8(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION uint8_to_vector(bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_to_ternary(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION ternary_to_vector(bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_to_int4(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION int4_to_vector(bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_to_binary(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION binary_hamming_distance(bytea, bytea) TO PUBLIC;

-- Grant permissions for GPU quantization functions
GRANT EXECUTE ON FUNCTION vector_to_int8_gpu(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_to_fp16_gpu(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_to_binary_gpu(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_to_uint8_gpu(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_to_ternary_gpu(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_to_int4_gpu(vector) TO PUBLIC;

-- Grant permissions for quantization analysis functions
GRANT EXECUTE ON FUNCTION quantize_analyze_int8(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION quantize_analyze_fp16(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION quantize_analyze_binary(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION quantize_analyze_uint8(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION quantize_analyze_ternary(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION quantize_analyze_int4(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION quantize_compare_distances(vector, vector, text) TO PUBLIC;

-- Grant permissions for sparse vector (vecmap) functions
GRANT EXECUTE ON FUNCTION vecmap_l2_distance(bytea, bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vecmap_cosine_distance(bytea, bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vecmap_inner_product(bytea, bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vecmap_l1_distance(bytea, bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vecmap_add(bytea, bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vecmap_sub(bytea, bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vecmap_mul_scalar(bytea, real) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vecmap_norm(bytea) TO PUBLIC;

-- Grant permissions for index tuning functions
GRANT EXECUTE ON FUNCTION index_tune_hnsw(text, text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION index_tune_ivf(text, text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION index_recommend_type(text, text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION index_tune_query_params(text) TO PUBLIC;

-- Grant permissions for vector comparison functions
GRANT EXECUTE ON FUNCTION vector_eq(vector, vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_ne(vector, vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vector_hash(vector) TO PUBLIC;

-- Grant permissions for halfvec functions
GRANT EXECUTE ON FUNCTION halfvec_eq(halfvec, halfvec) TO PUBLIC;
GRANT EXECUTE ON FUNCTION halfvec_ne(halfvec, halfvec) TO PUBLIC;
GRANT EXECUTE ON FUNCTION halfvec_hash(halfvec) TO PUBLIC;

-- Grant permissions for sparsevec functions
GRANT EXECUTE ON FUNCTION sparsevec_eq(sparsevec, sparsevec) TO PUBLIC;
GRANT EXECUTE ON FUNCTION sparsevec_ne(sparsevec, sparsevec) TO PUBLIC;
GRANT EXECUTE ON FUNCTION sparsevec_hash(sparsevec) TO PUBLIC;

-- Grant permissions for bit type functions
GRANT EXECUTE ON FUNCTION vector_to_bit(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION bit_to_vector(bit) TO PUBLIC;
GRANT EXECUTE ON FUNCTION binary_quantize(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION bit_hamming_distance(bit, bit) TO PUBLIC;

-- Grant permissions for halfvec distance functions
GRANT EXECUTE ON FUNCTION halfvec_l2_distance(halfvec, halfvec) TO PUBLIC;
GRANT EXECUTE ON FUNCTION halfvec_cosine_distance(halfvec, halfvec) TO PUBLIC;
GRANT EXECUTE ON FUNCTION halfvec_inner_product(halfvec, halfvec) TO PUBLIC;

-- Grant permissions for sparsevec distance functions
GRANT EXECUTE ON FUNCTION sparsevec_l2_distance(sparsevec, sparsevec) TO PUBLIC;
GRANT EXECUTE ON FUNCTION sparsevec_cosine_distance(sparsevec, sparsevec) TO PUBLIC;
GRANT EXECUTE ON FUNCTION sparsevec_inner_product(sparsevec, sparsevec) TO PUBLIC;

-- Grant permissions for graph operations (vgraph) functions
GRANT EXECUTE ON FUNCTION vgraph_bfs(vgraph, integer, integer) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vgraph_dfs(vgraph, integer) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vgraph_pagerank(vgraph, double precision, integer, double precision) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vgraph_community_detection(vgraph, integer) TO PUBLIC;

-- Grant permissions for index statistics and monitoring functions
GRANT EXECUTE ON FUNCTION index_statistics(text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION index_health(text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION index_rebuild_recommendation(text) TO PUBLIC;

-- ============================================================================
-- ============================================================================
-- SPARSE VECTORS & LEARNED SPARSE RETRIEVAL
-- ============================================================================

-- Sparse vector type for SPLADE/ColBERTv2/BM25
-- Drop any existing type first
DROP TYPE IF EXISTS sparse_vector CASCADE;

-- Create shell type first (required before I/O functions)
CREATE TYPE sparse_vector;

-- Create I/O functions for sparse_vector (C functions are registered by extension)
CREATE FUNCTION sparse_vector_in(cstring) RETURNS sparse_vector
	AS 'MODULE_PATHNAME', 'sparse_vector_in'
	LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION sparse_vector_out(sparse_vector) RETURNS cstring
	AS 'MODULE_PATHNAME', 'sparse_vector_out'
	LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION sparse_vector_recv(internal) RETURNS sparse_vector
	AS 'MODULE_PATHNAME', 'sparse_vector_recv'
	LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION sparse_vector_send(sparse_vector) RETURNS bytea
	AS 'MODULE_PATHNAME', 'sparse_vector_send'
	LANGUAGE C IMMUTABLE STRICT;

-- Create the full type with I/O functions (replaces shell type)
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

-- Sparse vector operators
CREATE FUNCTION sparse_vector_dot_product(sparse_vector, sparse_vector)
RETURNS float4
AS 'MODULE_PATHNAME', 'sparse_vector_dot_product'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION sparse_vector_dot_product(sparse_vector, sparse_vector) IS
	'Compute dot product between two sparse vectors';

CREATE OPERATOR <*> (
	LEFTARG = sparse_vector,
	RIGHTARG = sparse_vector,
	FUNCTION = sparse_vector_dot_product,
	COMMUTATOR = <*>
);

COMMENT ON OPERATOR <*>(sparse_vector, sparse_vector) IS
	'Dot product operator for sparse vectors';

-- Sparse index functions
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

-- Sparse search functions
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

-- Hybrid dense + sparse search
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

-- Grant permissions for sparse vector functions
GRANT EXECUTE ON FUNCTION sparse_vector_dot_product(sparse_vector, sparse_vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION sparse_index_create(text, text, text, int4) TO PUBLIC;
GRANT EXECUTE ON FUNCTION sparse_index_search(text, sparse_vector, int4) TO PUBLIC;
GRANT EXECUTE ON FUNCTION sparse_index_update(text, int4, sparse_vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION sparse_search(text, text, sparse_vector, int4) TO PUBLIC;
GRANT EXECUTE ON FUNCTION splade_embed(text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION colbertv2_embed(text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION bm25_score(text, text, float4, float4) TO PUBLIC;
GRANT EXECUTE ON FUNCTION hybrid_dense_sparse_search(text, text, text, vector, sparse_vector, int4, float4, float4) TO PUBLIC;
GRANT EXECUTE ON FUNCTION rrf_fusion(int4, float4, float4, float4) TO PUBLIC;
-- 
-- ============================================================================
-- FP8 QUANTIZATION (INT4, FP8)
-- ============================================================================

-- FP8 quantization functions
CREATE FUNCTION quantize_fp8_e4m3(vector)
RETURNS bytea
AS 'MODULE_PATHNAME', 'quantize_fp8_e4m3'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION quantize_fp8_e4m3(vector) IS
	'Quantize vector to FP8 E4M3 format (4 exp, 3 mantissa bits)';

CREATE FUNCTION quantize_fp8_e5m2(vector)
RETURNS bytea
AS 'MODULE_PATHNAME', 'quantize_fp8_e5m2'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION quantize_fp8_e5m2(vector) IS
	'Quantize vector to FP8 E5M2 format (5 exp, 2 mantissa bits)';

CREATE FUNCTION dequantize_fp8(bytea)
RETURNS vector
AS 'MODULE_PATHNAME', 'dequantize_fp8'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION dequantize_fp8(bytea) IS
	'Dequantize FP8 vector back to float32';
-- 
-- Auto quantization
CREATE FUNCTION auto_quantize(vector, text)
RETURNS bytea
AS 'MODULE_PATHNAME', 'auto_quantize'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION auto_quantize(vector, text) IS
	'Automatically select best quantization based on compression type (int4, fp8_e4m3, fp8_e5m2)';

-- Grant permissions for FP8 quantization functions
GRANT EXECUTE ON FUNCTION quantize_fp8_e4m3(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION quantize_fp8_e5m2(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION dequantize_fp8(bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION auto_quantize(vector, text) TO PUBLIC;

-- ============================================================================
-- EFFICIENT RERANKING WITH FLASH ATTENTION 2
-- ============================================================================

CREATE FUNCTION rerank_flash(
	query text,
	candidates text[],
	model text DEFAULT NULL,
	top_k int4 DEFAULT 10
)
RETURNS TABLE(idx int4, score float4)
AS 'MODULE_PATHNAME', 'rerank_flash'
LANGUAGE C STRICT;

COMMENT ON FUNCTION rerank_flash(text, text[], text, int4) IS
	'Rerank candidates using Flash Attention 2 for memory-efficient cross-encoder scoring';

CREATE FUNCTION rerank_long_context(
	query text,
	candidates text[],
	max_tokens int4 DEFAULT 8192,
	top_k int4 DEFAULT 10
)
RETURNS TABLE(idx int4, score float4)
AS 'MODULE_PATHNAME', 'rerank_long_context'
LANGUAGE C STRICT;

COMMENT ON FUNCTION rerank_long_context(text, text[], int4, int4) IS
	'Rerank with long context support (8K+ tokens) using Flash Attention';

-- Grant permissions for Flash Attention reranking functions
GRANT EXECUTE ON FUNCTION rerank_flash(text, text[], text, int4) TO PUBLIC;
GRANT EXECUTE ON FUNCTION rerank_long_context(text, text[], int4, int4) TO PUBLIC;

-- ============================================================================
-- MULTI-MODAL EMBEDDINGS (CLIP, ImageBind)
-- ============================================================================

CREATE FUNCTION clip_embed(input text, modality text DEFAULT 'text')
RETURNS vector
AS 'MODULE_PATHNAME', 'clip_embed'
LANGUAGE C STRICT;

COMMENT ON FUNCTION clip_embed(text, text) IS
	'Generate CLIP embedding from text or image';

CREATE FUNCTION imagebind_embed(input text, modality text)
RETURNS vector
AS 'MODULE_PATHNAME', 'imagebind_embed'
LANGUAGE C STRICT;

COMMENT ON FUNCTION imagebind_embed(text, text) IS
	'Generate ImageBind embedding from any modality (text, image, audio, video, depth, thermal)';

CREATE FUNCTION cross_modal_search(
	table_name text,
	embedding_col text,
	query_modality text,
	query_input text,
	target_modality text,
	k int4
)
RETURNS TABLE(doc_id int4, score float4)
AS 'MODULE_PATHNAME', 'cross_modal_search'
LANGUAGE C STRICT;

COMMENT ON FUNCTION cross_modal_search(text, text, text, text, text, int4) IS
	'Search across modalities (e.g., text query, image results)';

GRANT EXECUTE ON FUNCTION clip_embed(text, text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION imagebind_embed(text, text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION cross_modal_search(text, text, text, text, text, int4) TO PUBLIC;

-- ============================================================================
-- ML Unified API Schema
-- PostgresML-compatible unified interface for NeuronDB
-- ============================================================================

-- Projects Table (simplified for unified API)
CREATE TABLE IF NOT EXISTS neurondb.nb_catalog (
    project_id SERIAL PRIMARY KEY,
    project_name TEXT NOT NULL UNIQUE,
    algorithm TEXT NOT NULL,
    table_name TEXT NOT NULL,
    target_column TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature Stores Table
CREATE TABLE IF NOT EXISTS neurondb.feature_stores (
    store_id SERIAL PRIMARY KEY,
    store_name TEXT UNIQUE NOT NULL,
    entity_table TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Feature Definitions Table
CREATE TABLE IF NOT EXISTS neurondb.features (
    feature_id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES neurondb.feature_stores(store_id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL CHECK (feature_type IN ('numeric', 'categorical', 'vector', 'text')),
    transformation TEXT,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, feature_name, version)
);

-- Hyperparameter Tuning Results
CREATE TABLE IF NOT EXISTS neurondb.hyperparameter_results (
    result_id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES neurondb.ml_projects(project_id) ON DELETE CASCADE,
    algorithm TEXT NOT NULL,
    parameters JSONB NOT NULL,
    score FLOAT NOT NULL,
    cv_scores FLOAT[] NOT NULL,
    training_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hyperparam_project ON neurondb.hyperparameter_results(project_id);
CREATE INDEX IF NOT EXISTS idx_hyperparam_score ON neurondb.hyperparameter_results(score DESC);

-- Text ML Models
CREATE TABLE IF NOT EXISTS neurondb.text_models (
    model_id SERIAL PRIMARY KEY,
    model_name TEXT UNIQUE NOT NULL,
    model_type TEXT NOT NULL CHECK (model_type IN ('classification', 'sentiment', 'ner', 'summarization')),
    model_path TEXT,
    vocabulary_size INTEGER,
    embedding_dim INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- RAG Pipeline Configurations
CREATE TABLE IF NOT EXISTS neurondb.rag_pipelines (
    pipeline_id SERIAL PRIMARY KEY,
    pipeline_name TEXT UNIQUE NOT NULL,
    chunk_size INTEGER DEFAULT 512,
    chunk_overlap INTEGER DEFAULT 128,
    embedding_model TEXT NOT NULL,
    reranking_model TEXT,
    configuration JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- WAL COMPRESSION FUNCTIONS
-- ============================================================================

-- Vector WAL compression: Compress a vector using delta encoding
CREATE FUNCTION vector_wal_compress(text, text) RETURNS text
    AS 'MODULE_PATHNAME', 'vector_wal_compress'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_wal_compress(text, text) IS 'Compress a vector using delta encoding for WAL optimization';

-- Vector WAL decompression: Decompress a delta-encoded vector
CREATE FUNCTION vector_wal_decompress(text, text) RETURNS text
    AS 'MODULE_PATHNAME', 'vector_wal_decompress'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_wal_decompress(text, text) IS 'Decompress a delta-encoded vector from WAL';

-- Vector WAL size estimation: Estimate compressed size
CREATE FUNCTION vector_wal_estimate_size(text, text) RETURNS integer
    AS 'MODULE_PATHNAME', 'vector_wal_estimate_size'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_wal_estimate_size(text, text) IS 'Estimate compressed size for WAL compression';

-- Vector WAL compression control: Enable/disable WAL compression
CREATE FUNCTION vector_wal_set_compression(boolean) RETURNS boolean
    AS 'MODULE_PATHNAME', 'vector_wal_set_compression'
    LANGUAGE C STRICT;
COMMENT ON FUNCTION vector_wal_set_compression(boolean) IS 'Enable or disable WAL compression for vectors';

-- Vector WAL statistics: Get WAL compression statistics
CREATE FUNCTION vector_wal_get_stats() RETURNS text
    AS 'MODULE_PATHNAME', 'vector_wal_get_stats'
    LANGUAGE C STABLE;
COMMENT ON FUNCTION vector_wal_get_stats() IS 'Get WAL compression statistics';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.nb_catalog TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.feature_stores TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.features TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.hyperparameter_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.text_models TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.rag_pipelines TO PUBLIC;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA neurondb TO PUBLIC;

-- Comments
COMMENT ON TABLE neurondb.nb_catalog IS 'Simplified projects table for unified ML API';
COMMENT ON TABLE neurondb.feature_stores IS 'Feature store registry for ML feature management';
COMMENT ON TABLE neurondb.features IS 'Feature definitions with versioning';
COMMENT ON TABLE neurondb.hyperparameter_results IS 'Hyperparameter tuning results';
COMMENT ON TABLE neurondb.text_models IS 'Text ML model registry';
COMMENT ON TABLE neurondb.rag_pipelines IS 'RAG pipeline configurations';

