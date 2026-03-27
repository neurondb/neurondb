# NeuronDB — testing report follow-ups

This note ties the independent [`neurondb-testing-report`](https://github.com/usamaidrsk/neurondb-testing-report) tree (local path: `/home/pge/pge/neurondb-testing-report`) to this repository: what was fixed in code, what is covered by automated SQL checks, and what still needs environment-specific QA.

## Verification matrix (report → repo)

| Report source | Topic | Status in this repo |
|---------------|--------|---------------------|
| [FINAL-SUMMARY.md](/home/pge/pge/neurondb-testing-report/FINAL-SUMMARY.md) | v3 install: `\ir` / broken 3.1.0 script | **Fixed** — `make` builds a single `sql/neurondb--3.1.0.sql` without psql meta-commands (`neurondb-extension-sql` / dependency on extension SQL). |
| FINAL-SUMMARY | `information_schema` DO block on `CREATE EXTENSION` | **Fixed** — `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` on `neurondb.rag_pipelines` in `neurondb--3.0.0-devel.sql` (+ `.linux` / `.macos`). |
| 3.1.0 script | `GRANT EXECUTE ON FUNCTION neurondb.distance` / `similarity` ambiguous after overloads | **Fixed** — grants use full argument lists (`vector` / `sparsevec` signatures) in devel SQL + regenerated `sql/neurondb--3.1.0.sql`. |
| README | **XGBoost** train / SPI | **Improved** — `load_training_data` uses `quote_identifier`, filters NULL rows, supports **vector** features and **int2/int8** labels, rejects single-class training, fixes **float4[]** row reads; `num_class >= 2` enforced for classifiers (`ml_xgboost.c`). |
| README | **Isolation forest** (`detect_anomalies_isolation_forest` in `ml_anomaly_detection.c`) | **Not linked** — that compilation unit is still **omitted** from `OBJS` in `Makefile.core` (duplicate / broken symbols vs `fmgrprotos.h`); working isolation-style logic also lives in `analytics.c`. Wiring or repair is a follow-up. |
| FINAL-SUMMARY | Hybrid scan / bogus heap OID | **Fixed** — `custom_hybrid_scan.c` uses `rte->relid`; regression SQL `src/tests/sql/basic/112_hybrid_scan_planner_hook.sql`. |
| FINAL-SUMMARY | Unified `decision_tree` crash | **Fixed** — `ml_unified_api.c` CPU path uses `FunctionCall5` into `train_decision_tree_classifier`; `src/tests/sql/basic/113_unified_decision_tree_train.sql`. |
| FINAL-SUMMARY | Sparse distance zero via unified API | **Fixed** — `neurondb.distance` / `neurondb.similarity` **sparsevec** overloads in devel SQL; `src/tests/sql/basic/080_vector_sparsevec.sql` extended. |
| [README.md known issues](/home/pge/pge/neurondb-testing-report/README.md) | `elastic_net` predict | **Fixed** — `predict_elastic_net` uses catalog coefficients + `float8[]` (`ml_ridge_lasso.c`). |
| README | `dbscan` not wired in `neurondb.train(...)` | **Fixed** — PL/pgSQL `train` dispatches `dbscan` to six-arg C `neurondb.train`. |
| README | **KNN multiclass** always class 0 | **Fixed** — Majority vote over arbitrary integer class labels in `knn_classify`, CPU model predict path, batch evaluate paths (binary confusion only when training labels are all 0/1), and CUDA/ROCm `gpu_*_knn_kernels.cu`. Test: `src/tests/sql/basic/114_knn_multiclass_kmeans_train.sql`. |
| README | **kmeans** `train()` return vs model id | **Fixed** — `neurondb.train('kmeans', ...)` returns `train_kmeans_model_id(...)` in `neurondb--3.0.0-devel.sql` (+ platform variants); concatenated `sql/neurondb--3.1.0.sql` picks this up via build. Same SQL test file as above. |
| README / unified `train` | `minibatch_kmeans` / `hierarchical` vs `RETURNS integer` | **Fixed** — those algorithms cannot yield a scalar catalog id; `neurondb.train` now raises a clear exception and points at `cluster_minibatch_kmeans` / `cluster_hierarchical`. |
| README | **random_forest** predict / `n_classes=0` | **Improved** — If `model->n_classes <= 0`, predict allocates a **binary** vote histogram (legacy fallback). Multiclass models with wrong metadata may still need retrain or serializer fixes. |
| README / level4 | Docker build context, missing sibling repos | **Fixed** — `docker/neurondb/docker-compose.yml` + Dockerfiles build from repo root; optional SQL under `docker/neurondb/optional-init/`. |
| README | ridge / xgboost / naive_bayes / neural_network / isolation_forest crashes | **Partial** — elastic_net + decision_tree addressed; remaining algorithms need **ASAN/repro** per [`development/asan-ubsan.md`](development/asan-ubsan.md) (not closed in this pass). |
| README | IVFFlat vs docs | **Open / docs** — IVF access method exists in tree; marketing/docs parity is separate from crash fixes. |
| README | HNSW 100K + WAL in Docker | **Ops** — treat as disk / `max_wal_size` / volume sizing unless a specific engine bug is reproduced. |
| [07-neuronagent](/home/pge/pge/neurondb-testing-report/07-neuronagent/neuronagent-testing-report.md) | REST agent, schema, handlers | **Out of repo** — NeuronAgent Go service; not shipped under `neurondb`. |

## Regression tests added or extended (this repo)

| File | What it guards |
|------|----------------|
| [`src/tests/sql/basic/112_hybrid_scan_planner_hook.sql`](../src/tests/sql/basic/112_hybrid_scan_planner_hook.sql) | Vector index + GIN on one table; planner hook / KNN paths. |
| [`src/tests/sql/basic/113_unified_decision_tree_train.sql`](../src/tests/sql/basic/113_unified_decision_tree_train.sql) | Unified decision tree training returns a positive model id. |
| [`src/tests/sql/basic/080_vector_sparsevec.sql`](../src/tests/sql/basic/080_vector_sparsevec.sql) | Unified distance on `sparsevec` non-zero for a non-trivial pair. |
| [`src/tests/sql/basic/114_knn_multiclass_kmeans_train.sql`](../src/tests/sql/basic/114_knn_multiclass_kmeans_train.sql) | Multiclass `knn_classify`; `neurondb.train('kmeans', ...)` returns catalog model id. |

Run examples (require a cluster with the extension installed):

```bash
psql -v ON_ERROR_STOP=1 -f src/tests/sql/basic/114_knn_multiclass_kmeans_train.sql
```

PgXS `make installcheck` runs the `REGRESS` list in `Makefile.core` (separate from `src/tests/sql/`); both should be green before a release.

**Fresh-cluster checks:** `CREATE EXTENSION neurondb` (default script `neurondb--3.1.0.sql`) requires **`shared_preload_libraries = 'neurondb'`** and a restart before the first `CREATE EXTENSION`. PL/Python LLM helpers are **not** in that script; install them separately: `CREATE EXTENSION plpython3u` then run **`share/extension/neurondb_llm_functions.sql`** (see header in that file).

**PgXS `make installcheck`:** Regression files live under `sql/` and `expected/` (currently `000_smoke`). The test database must use a postmaster that already has **neurondb preloaded** (same as production). Example: `PGHOST=/tmp PGPORT=55433 make installcheck` when the server listens there with `shared_preload_libraries = 'neurondb'`.

## NeuronAgent (separate repository)

Issues in `07-neuronagent/neuronagent-testing-report.md` apply to the **NeuronAgent** Go service. Apply fixes there; optional SQL for images is described under [`docker/neurondb/optional-init/`](../docker/neurondb/optional-init/README.md).

## LLM / RAG / API keys

Capabilities that need **plpython3u**, outbound HTTP, or API keys remain environment-dependent; the report’s “not testable without keys” status is expected.
