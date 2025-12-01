# Metal Support

Apple Silicon GPU acceleration.

## Overview

Metal support enables GPU acceleration on Apple Silicon (M1, M2, M3) Macs.

## Configuration

```conf
shared_preload_libraries = 'neurondb'
neurondb.gpu_enabled = true
neurondb.gpu_backend = 'metal'
```

## GPU Operations

```sql
-- Metal-accelerated vector operations
SELECT vector_l2_distance_gpu(embedding, query) AS distance
FROM documents;
```

## Learn More

For detailed documentation on Metal setup and Apple Silicon optimization, visit:

**[Metal Support Documentation](https://neurondb.ai/docs/gpu/)**

## Related Topics

- [CUDA Support](cuda-support.md) - NVIDIA GPU support
- [GPU Auto-Detection](auto-detection.md) - Automatic detection

