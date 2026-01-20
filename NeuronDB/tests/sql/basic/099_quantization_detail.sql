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

-- UINT8 quantization: 8 bits (1 byte) per dimension, unsigned integers [0, 255] (skip if not available)
DO $$
BEGIN
  BEGIN
    PERFORM vector_to_uint8('[1,-1,0,3]'::vector);
    RAISE NOTICE 'vector_to_uint8 is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'vector_to_uint8 not available, skipping UINT8 tests';
  END;
END$$;

-- Binary quantization: 1 bit per dimension, packed into bytes; 4 dimensions use 1 byte
SELECT 'BINARY (CPU)' AS quantization_method, octet_length(vector_to_binary('[1,-1,0,3]'::vector)) AS bytes_per_vector;
SELECT 'BINARY (GPU)' AS quantization_method, octet_length(vector_to_binary_gpu('[1,-1,0,3]'::vector)) AS bytes_per_vector;

-- Ternary quantization: 2 bits per dimension, packed; 4 dimensions use 1 byte (skip if not available)
DO $$
BEGIN
  BEGIN
    PERFORM vector_to_ternary('[1,-1,0,3]'::vector);
    RAISE NOTICE 'vector_to_ternary is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'vector_to_ternary not available, skipping ternary tests';
  END;
END$$;

-- (If available) INT4 quantization: 4 bits per dimension, packed; 4 dimensions use 2 bytes
-- Uncomment if the function exists in your extension:

-- Quantization accuracy analysis functions (skip if not available)
DO $$
BEGIN
  BEGIN
    PERFORM quantize_analyze_int8('[1.0, -1.0, 0.5, -0.5, 2.0]'::vector);
    RAISE NOTICE 'quantize_analyze_int8 is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'quantize_analyze_int8 not available, skipping analysis tests';
  END;
  
  BEGIN
    PERFORM quantize_analyze_fp16('[1.0, -1.0, 0.5, -0.5, 2.0]'::vector);
    RAISE NOTICE 'quantize_analyze_fp16 is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'quantize_analyze_fp16 not available';
  END;
  
  BEGIN
    PERFORM quantize_analyze_binary('[1.0, -1.0, 0.5, -0.5, 2.0]'::vector);
    RAISE NOTICE 'quantize_analyze_binary is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'quantize_analyze_binary not available';
  END;
  
  BEGIN
    PERFORM quantize_compare_distances('[1.0, 2.0, 3.0]'::vector, '[1.1, 2.1, 3.1]'::vector, 'int8');
    RAISE NOTICE 'quantize_compare_distances is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'quantize_compare_distances not available, skipping comparison tests';
  END;
END$$;
-- SELECT 'INT4 (CPU)' AS quantization_method, octet_length(vector_to_int4('[1,-1,0,3]'::vector)) AS bytes_per_vector;
-- SELECT 'INT4 (GPU)' AS quantization_method, octet_length(vector_to_int4_gpu('[1,-1,0,3]'::vector)) AS bytes_per_vector;

-- Overview: This script demonstrates every quantization format supported by NeurondB,
-- shows CPU and GPU variants where applicable, and returns the number of bytes required to store a 4-dimensional vector.
