# CUDA Support

NVIDIA GPU acceleration for vector operations and ML inference.

## Overview

CUDA support provides GPU acceleration for vector distance calculations and ML operations.

## Configuration

Enable CUDA in `postgresql.conf`:

```conf
shared_preload_libraries = 'neurondb'
neurondb.gpu_enabled = true
neurondb.gpu_backend = 'cuda'
neurondb.gpu_device = 0
```

## GPU Distance Calculations

```sql
-- GPU-accelerated L2 distance
SELECT vector_l2_distance_gpu(
    '[1.0, 2.0, 3.0]'::vector,
    '[4.0, 5.0, 6.0]'::vector
) AS distance;

-- GPU-accelerated cosine distance
SELECT vector_cosine_distance_gpu(
    embedding,
    query_vector
) AS distance
FROM documents;
```

## Check GPU Status

```sql
-- GPU information
SELECT neurondb_gpu_info();

-- GPU statistics
SELECT * FROM pg_stat_neurondb_gpu;
```

## Learn More

For detailed documentation on CUDA setup, GPU optimization, and performance tuning, visit:

**[CUDA Support Documentation](https://neurondb.ai/docs/gpu/)**

## Related Topics

- [GPU Auto-Detection](auto-detection.md) - Automatic GPU setup
- [ROCm Support](rocm-support.md) - AMD GPU support

