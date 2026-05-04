# Engineering debt tracker (snapshot)

This file summarizes known gaps aligned with the six-month quality plan. Update as items close.

## GPU / Metal

- [src/gpu/metal/gpu_backend_metal.c](../../src/gpu/metal/gpu_backend_metal.c): memory-corruption and `pfree` TODOs; XGBoost/CatBoost prediction completeness; integrated filtering.
- [src/gpu/cuda/gpu_backend_cuda.c](../../src/gpu/cuda/gpu_backend_cuda.c): CUDA streams / async execution; integrated filtering.
- [src/gpu/rocm/gpu_backend_rocm.c](../../src/gpu/rocm/gpu_backend_rocm.c): integrated filtering parity with CUDA where required.

## Scan / indexes

- [src/scan/filtered_hnsw_scan.c](../../src/scan/filtered_hnsw_scan.c): GPU path with integrated filtering.

## ML

- [src/ml/ml_transformer_llm.c](../../src/ml/ml_transformer_llm.c): ONNX inference path for `neurondb_predict_transformer_llm` when `HAVE_ONNX_RUNTIME` is defined (currently returns `FEATURE_NOT_SUPPORTED` with a clear hint).

## SQL

- Versioned scripts under `sql/` may contain analytics TODOs; track per release.

## CI / tests

- Nightly `run_test.py --category all` skips `*live_required*` files when `NDB_CI_SKIP_LIVE=1` (default in workflows). Run live tests manually or in a dedicated job with secrets.
