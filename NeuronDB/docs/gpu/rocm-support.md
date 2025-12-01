# ROCm Support

AMD GPU acceleration for vector operations.

## Overview

ROCm support enables AMD GPU acceleration for NeuronDB operations.

## Configuration

```conf
shared_preload_libraries = 'neurondb'
neurondb.gpu_enabled = true
neurondb.gpu_backend = 'rocm'
neurondb.gpu_device = 0
```

## GPU Operations

```sql
-- GPU-accelerated operations
SELECT vector_l2_distance_gpu(embedding, query) AS distance
FROM documents;
```

## Learn More

For detailed documentation on ROCm setup and AMD GPU configuration, visit:

**[ROCm Support Documentation](https://neurondb.ai/docs/gpu/)**

## Related Topics

- [CUDA Support](cuda-support.md) - NVIDIA GPU support
- [GPU Auto-Detection](auto-detection.md) - Automatic detection

