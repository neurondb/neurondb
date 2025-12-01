# SIMD Optimization

AVX2/AVX512 (x86_64) and NEON (ARM64) optimization with prefetching.

## Overview

NeuronDB uses SIMD (Single Instruction, Multiple Data) instructions for faster vector operations.

## Supported Architectures

- **x86_64**: AVX2 and AVX512 with FMA instructions
- **ARM64**: NEON with dotprod extension

## Automatic Optimization

SIMD optimizations are enabled automatically at compile time:

```sql
-- Operations automatically use SIMD
SELECT embedding <-> query_vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

## Performance

SIMD provides significant speedup:
- **2-4x faster** for distance calculations
- **3-5x faster** for batch operations

## Learn More

For detailed documentation on SIMD optimization, compiler flags, and performance tuning, visit:

**[SIMD Optimization Documentation](https://neurondb.ai/docs/performance/optimization/)**

## Related Topics

- [GPU Acceleration](../gpu/cuda-support.md) - GPU optimization
- [Monitoring](monitoring.md) - Performance monitoring

