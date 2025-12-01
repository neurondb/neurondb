# GPU Auto-Detection

Automatic GPU detection and fallback to CPU.

## Overview

NeuronDB automatically detects available GPUs and falls back to CPU if GPU is unavailable.

## Automatic Detection

```sql
-- Check GPU status
SELECT neurondb_gpu_info();

-- Enable auto-detection (default)
SET neurondb.gpu_enabled = true;
SET neurondb.gpu_auto_detect = true;
```

## Fallback Behavior

When GPU is unavailable, operations automatically fall back to CPU:

```sql
-- Will use GPU if available, CPU otherwise
SELECT vector_l2_distance_gpu(embedding, query) AS distance
FROM documents;
```

## Learn More

For detailed documentation on GPU auto-detection, fallback behavior, and manual configuration, visit:

**[GPU Auto-Detection Documentation](https://neurondb.ai/docs/gpu/)**

## Related Topics

- [CUDA Support](cuda-support.md) - NVIDIA GPU
- [ROCm Support](rocm-support.md) - AMD GPU
- [Metal Support](metal-support.md) - Apple Silicon

