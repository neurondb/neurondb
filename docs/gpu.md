# GPU acceleration

NeuronDB can use **CUDA**, **ROCm**, or **Metal** for parts of the workload (depending on build and platform). Index build paths are predominantly CPU-oriented; always check the detailed guides for your OS.

## Where to read more

| Topic | Document |
|--------|----------|
| Feature matrix & support | [gpu/gpu-feature-matrix.md](gpu/gpu-feature-matrix.md) |
| NVIDIA CUDA | [gpu/cuda-support.md](gpu/cuda-support.md) |
| AMD ROCm | [gpu/rocm-support.md](gpu/rocm-support.md) |
| Apple Metal | [gpu/metal-support.md](gpu/metal-support.md) |
| Auto-detection | [gpu/auto-detection.md](gpu/auto-detection.md) |

## Docker

Official images on **Docker Hub** and **GHCR** are **CUDA-first**: **`neurondb/neurondb-cuda`** / **`ghcr.io/neurondb/neurondb-cuda`** (built from [`docker/neurondb/Dockerfile.gpu.cuda`](../docker/neurondb/Dockerfile.gpu.cuda)). Use the NVIDIA Container Toolkit and `--gpus all` at runtime. For local development, this repository’s Compose file can still **build** CPU or other GPU variants from source—see [`docker/README.md`](../docker/README.md).
