# GPU Feature Matrix

Feature parity, known gaps, and current state for CUDA, ROCm, and Metal GPU backends in NeuronDB.

## Feature Parity Matrix

| Feature | CUDA | ROCm | Metal | CPU Fallback |
|---------|------|------|-------|--------------|
| **Vector Distance Calculations** |
| L2 distance | ✅ | ✅ | ✅ | ✅ |
| Cosine distance | ✅ | ✅ | ✅ | ✅ |
| Inner product | ✅ | ✅ | ✅ | ✅ |
| Manhattan distance | ✅ | ⚠️ Partial | ❌ | ✅ |
| Hamming distance | ✅ | ⚠️ Partial | ❌ | ✅ |
| **Index Operations** |
| HNSW search | ✅ | ✅ | ⚠️ Limited | ✅ |
| IVF search | ✅ | ✅ | ❌ | ✅ |
| Index build | ⚠️ CPU only | ⚠️ CPU only | ⚠️ CPU only | ✅ |
| **Batch Operations** |
| Batch distance | ✅ | ✅ | ✅ | ✅ |
| Batch embedding | ✅ | ✅ | ⚠️ Limited | ✅ |
| **Memory Management** |
| GPU memory pool | ✅ | ✅ | ⚠️ Basic | N/A |
| Zero-copy transfers | ✅ | ✅ | ❌ | N/A |
| Memory monitoring | ✅ | ⚠️ Partial | ❌ | N/A |
| **Performance** |
| Query acceleration | ✅ 2-3x | ✅ 2-3x | ✅ 1.5-2x | Baseline |
| Batch acceleration | ✅ 5-10x | ✅ 5-10x | ✅ 3-5x | Baseline |

**Legend:**
- ✅ Fully supported
- ⚠️ Partial support or known limitations
- ❌ Not supported

## CUDA Support

### Supported Features

- **All vector distance metrics:** L2, Cosine, Inner Product, Manhattan, Hamming
- **HNSW and IVF search:** Full acceleration
- **Batch operations:** Optimized for large batches
- **Memory management:** Advanced pooling and monitoring
- **Multi-GPU:** Support for multiple CUDA devices

### Performance Characteristics

- **Query QPS:** 2-3x improvement over CPU
- **Batch operations:** 5-10x improvement for large batches
- **Memory efficiency:** Optimized memory transfers
- **Scalability:** Supports up to 8 GPUs

### Known Limitations

- **Index build:** Currently CPU-only (GPU build planned)
- **Mixed precision:** FP16 support experimental
- **Tensor cores:** Not yet utilized (future optimization)

### Requirements

- **CUDA Version:** 11.0+ (12.1+ recommended)
- **GPU:** NVIDIA GPUs with Compute Capability 7.0+
- **Drivers:** NVIDIA driver 450.80.02+
- **Libraries:** cuBLAS, cuDNN (optional)

## ROCm Support

### Supported Features

- **Core distance metrics:** L2, Cosine, Inner Product
- **HNSW search:** Full acceleration
- **IVF search:** Full acceleration
- **Batch operations:** Optimized for AMD GPUs
- **Memory management:** Basic pooling

### Performance Characteristics

- **Query QPS:** 2-3x improvement over CPU
- **Batch operations:** 5-10x improvement
- **Memory efficiency:** Good for AMD GPUs
- **Scalability:** Supports multiple AMD GPUs

### Known Limitations

- **Manhattan/Hamming:** Partial support (slower than CUDA)
- **Memory monitoring:** Limited metrics compared to CUDA
- **Index build:** Currently CPU-only
- **Mixed precision:** Not yet supported

### Requirements

- **ROCm Version:** 5.0+ (5.7+ recommended)
- **GPU:** AMD GPUs with RDNA2+ or CDNA architecture
- **Drivers:** ROCm driver stack
- **Libraries:** rocBLAS, rocRAND

## Metal Support (macOS)

### Supported Features

- **Core distance metrics:** L2, Cosine, Inner Product
- **HNSW search:** Limited acceleration (some operations CPU-bound)
- **Batch operations:** Basic support
- **Memory management:** Basic pooling

### Performance Characteristics

- **Query QPS:** 1.5-2x improvement over CPU
- **Batch operations:** 3-5x improvement
- **Memory efficiency:** Good for Apple Silicon
- **Power efficiency:** Excellent on Apple Silicon

### Known Limitations

- **IVF search:** Not yet supported
- **Manhattan/Hamming:** Not supported
- **Index build:** Currently CPU-only
- **Memory monitoring:** Limited (no system-level APIs)
- **Multi-GPU:** Limited (Apple Silicon typically single GPU)
- **Known bugs:** Some Metal-specific issues with large batches

### Requirements

- **macOS Version:** 13.0+ (Ventura) or 14.0+ (Sonoma)
- **Hardware:** Apple Silicon (M1, M2, M3, or later)
- **Metal Version:** Metal 3.0+
- **Libraries:** Metal Performance Shaders

### Known Metal Bugs

1. **Large batch size:** Batches > 1000 may cause memory issues
   - **Workaround:** Use smaller batches (500-800)
   - **Status:** Under investigation

2. **Concurrent queries:** Some race conditions with concurrent Metal operations
   - **Workaround:** Limit concurrent GPU queries
   - **Status:** Partial fix in progress

3. **Memory leaks:** Occasional memory leaks in long-running processes
   - **Workaround:** Restart PostgreSQL periodically
   - **Status:** Being addressed

## CPU Fallback

All GPU operations automatically fall back to CPU if:
- GPU is not available
- GPU operation fails
- `neurondb.gpu_fail_open = true` (default)

**Fallback behavior:**
- Automatic and transparent
- No error thrown (unless `gpu_fail_open = false`)
- Performance degrades to CPU baseline

## Performance Comparison

### Query Performance (QPS)

| Dataset | Dimensions | CPU | CUDA | ROCm | Metal |
|---------|------------|-----|------|------|-------|
| SIFT-128 | 128 | 1,200 | 3,500 | 3,200 | 2,000 |
| GIST-960 | 960 | 450 | 1,800 | 1,600 | 800 |
| GloVe-100 | 100 | 1,500 | 4,200 | 3,800 | 2,400 |

*Note: Actual performance varies based on hardware, dataset, and configuration.*

### Batch Performance (Throughput)

| Operation | Batch Size | CPU | CUDA | ROCm | Metal |
|-----------|------------|-----|------|------|-------|
| Distance calc | 1,000 | 500 ops/s | 5,000 ops/s | 4,500 ops/s | 2,000 ops/s |
| Embedding gen | 100 | 10 texts/s | 100 texts/s | 90 texts/s | 40 texts/s |

## Configuration

### Enable GPU

```sql
-- Enable GPU mode
SET neurondb.compute_mode = true;

-- Select GPU device (CUDA/ROCm)
SET neurondb.gpu_device = 0;

-- Configure batch size
SET neurondb.gpu_batch_size = 1000;

-- Enable fail-open (fallback to CPU on error)
SET neurondb.gpu_fail_open = true;
```

### Check GPU Status

```sql
-- Check GPU availability
SELECT * FROM neurondb_gpu_info();

-- Check if GPU is enabled
SELECT neurondb.gpu_enabled();

-- Get GPU device count
SELECT neurondb.gpu_device_count();
```

## Migration Between GPU Backends

### CUDA → ROCm

**Process:**
1. Install ROCm drivers and libraries
2. Rebuild NeuronDB with ROCm support
3. Update configuration:
   ```sql
   SET neurondb.gpu_backend = 'rocm';
   SET neurondb.gpu_device = 0;
   ```
4. Restart PostgreSQL

### CUDA → Metal (macOS)

**Process:**
1. Ensure macOS 13+ and Apple Silicon
2. Rebuild NeuronDB with Metal support
3. Update configuration:
   ```sql
   SET neurondb.gpu_backend = 'metal';
   ```
4. Restart PostgreSQL

**Note:** Some features may not be available on Metal (see limitations above).

## Roadmap

### Planned Features

**Q2 2025:**
- GPU-accelerated index build (CUDA)
- Mixed precision support (FP16)
- Enhanced Metal support (fix known bugs)

**Q3 2025:**
- Tensor core utilization (CUDA)
- ROCm index build acceleration
- Multi-GPU load balancing

**Q4 2025:**
- Metal IVF support
- Unified GPU API across backends
- Advanced memory management

## Troubleshooting

### GPU Not Detected

```sql
-- Check GPU info
SELECT * FROM neurondb_gpu_info();

-- If empty, check:
-- 1. GPU drivers installed
-- 2. PostgreSQL has GPU access
-- 3. Correct backend compiled
```

### Performance Issues

```sql
-- Check GPU utilization
SELECT * FROM neurondb.llm_gpu_utilization();

-- Monitor memory usage
SELECT neurondb.gpu_memory_usage();

-- Adjust batch size if needed
SET neurondb.gpu_batch_size = 500;  -- Reduce if OOM
```

### Metal-Specific Issues

```sql
-- Reduce batch size for Metal
SET neurondb.gpu_batch_size = 500;  -- Lower than CUDA/ROCm

-- Disable Metal if issues persist
SET neurondb.compute_mode = false;  -- Fallback to CPU
```

## Related Documentation

- [CUDA Support](cuda-support.md) - CUDA setup and configuration
- [ROCm Support](rocm-support.md) - ROCm setup and configuration
- [Metal Support](metal-support.md) - Metal setup and configuration
- [Benchmark Results](../benchmarks/benchmark_results.md) - Performance numbers
- [Configuration Guide](../NeuronDB/docs/configuration.md) - All GPU settings

