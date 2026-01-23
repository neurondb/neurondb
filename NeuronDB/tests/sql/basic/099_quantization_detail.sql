-- Extension created in 01_types_basic
SET neurondb.compute_mode = off;

-- Quantization detail: All supported formats, their storage size, and characteristics
-- Each query below provides the quantized binary size for a vector of four dimensions: [1, -1, 0, 3]

-- INT8 quantization: 8 bits (1 byte) per dimension, signed integers [-128, 127]
SELECT 'INT8 (CPU)' AS quantization_method, octet_length(vector_to_int8('[1,-1,0,3]'::vector)) AS bytes_per_vector;
SELECT 'INT8 (GPU)' AS quantization_method, octet_length(vector_to_int8_gpu('[1,-1,0,3]'::vector)) AS bytes_per_vector;

-- FP16 quantization: 16 bits (2 bytes) per dimension, IEEE 754 half-precision
SELECT 'FP16 (CPU)' AS quantization_method, octet_length(vector_to_fp16('[1,-1,0,3]'::vector)) AS bytes_per_vector;
SELECT 'FP16 (GPU)' AS quantization_method, octet_length(vector_to_fp16_gpu('[1,-1,0,3]'::vector)) AS bytes_per_vector;

-- UINT8 quantization: 8 bits (1 byte) per dimension, unsigned integers [0, 255]
-- Validate that function returns expected results
DO $$
DECLARE
    result_size INT;
BEGIN
    SELECT octet_length(vector_to_uint8('[1,-1,0,3]'::vector)) INTO result_size;
    
    IF result_size IS NULL OR result_size = 0 THEN
        RAISE EXCEPTION 'vector_to_uint8 returned invalid size: %', result_size;
    END IF;
    
    -- 4 dimensions * 1 byte = 4 bytes
    IF result_size != 4 THEN
        RAISE EXCEPTION 'vector_to_uint8 returned % bytes for 4-dim vector, expected 4', result_size;
    END IF;
END$$;

SELECT 'UINT8 (CPU)' AS quantization_method, octet_length(vector_to_uint8('[1,-1,0,3]'::vector)) AS bytes_per_vector;

-- Binary quantization: 1 bit per dimension, packed into bytes; 4 dimensions use 1 byte
SELECT 'BINARY (CPU)' AS quantization_method, octet_length(vector_to_binary('[1,-1,0,3]'::vector)) AS bytes_per_vector;
SELECT 'BINARY (GPU)' AS quantization_method, octet_length(vector_to_binary_gpu('[1,-1,0,3]'::vector)) AS bytes_per_vector;

-- Ternary quantization: 2 bits per dimension, packed; 4 dimensions use 1 byte
-- Validate that function returns expected results
DO $$
DECLARE
    result_size INT;
BEGIN
    SELECT octet_length(vector_to_ternary('[1,-1,0,3]'::vector)) INTO result_size;
    
    IF result_size IS NULL OR result_size = 0 THEN
        RAISE EXCEPTION 'vector_to_ternary returned invalid size: %', result_size;
    END IF;
    
    -- 4 dimensions * 2 bits = 8 bits = 1 byte
    IF result_size != 1 THEN
        RAISE EXCEPTION 'vector_to_ternary returned % bytes for 4-dim vector, expected 1', result_size;
    END IF;
END$$;

SELECT 'TERNARY (CPU)' AS quantization_method, octet_length(vector_to_ternary('[1,-1,0,3]'::vector)) AS bytes_per_vector;

-- (If available) INT4 quantization: 4 bits per dimension, packed; 4 dimensions use 2 bytes
-- Uncomment if the function exists in your extension:

-- Quantization accuracy analysis functions
-- Validate that functions return expected results
DO $$
DECLARE
    int8_result RECORD;
    fp16_result RECORD;
    binary_result RECORD;
    compare_result RECORD;
BEGIN
    -- Test quantize_analyze_int8
    SELECT * INTO int8_result
    FROM quantize_analyze_int8('[1.0, -1.0, 0.5, -0.5, 2.0]'::vector);
    
    IF int8_result IS NULL THEN
        RAISE EXCEPTION 'quantize_analyze_int8 returned NULL';
    END IF;
    
    -- Test quantize_analyze_fp16
    SELECT * INTO fp16_result
    FROM quantize_analyze_fp16('[1.0, -1.0, 0.5, -0.5, 2.0]'::vector);
    
    IF fp16_result IS NULL THEN
        RAISE EXCEPTION 'quantize_analyze_fp16 returned NULL';
    END IF;
    
    -- Test quantize_analyze_binary
    SELECT * INTO binary_result
    FROM quantize_analyze_binary('[1.0, -1.0, 0.5, -0.5, 2.0]'::vector);
    
    IF binary_result IS NULL THEN
        RAISE EXCEPTION 'quantize_analyze_binary returned NULL';
    END IF;
    
    -- Test quantize_compare_distances
    SELECT * INTO compare_result
    FROM quantize_compare_distances('[1.0, 2.0, 3.0]'::vector, '[1.1, 2.1, 3.1]'::vector, 'int8');
    
    IF compare_result IS NULL THEN
        RAISE EXCEPTION 'quantize_compare_distances returned NULL';
    END IF;
END$$;

-- Display analysis results
SELECT quantize_analyze_int8('[1.0, -1.0, 0.5, -0.5, 2.0]'::vector) AS int8_analysis;
SELECT quantize_analyze_fp16('[1.0, -1.0, 0.5, -0.5, 2.0]'::vector) AS fp16_analysis;
SELECT quantize_analyze_binary('[1.0, -1.0, 0.5, -0.5, 2.0]'::vector) AS binary_analysis;
SELECT quantize_compare_distances('[1.0, 2.0, 3.0]'::vector, '[1.1, 2.1, 3.1]'::vector, 'int8') AS distance_comparison;
-- SELECT 'INT4 (CPU)' AS quantization_method, octet_length(vector_to_int4('[1,-1,0,3]'::vector)) AS bytes_per_vector;
-- SELECT 'INT4 (GPU)' AS quantization_method, octet_length(vector_to_int4_gpu('[1,-1,0,3]'::vector)) AS bytes_per_vector;

-- Overview: This script demonstrates every quantization format supported by NeurondB,
-- shows CPU and GPU variants where applicable, and returns the number of bytes required to store a 4-dimensional vector.
