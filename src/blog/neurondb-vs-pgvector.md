NeuronDB Vector vs pgvector

You store embeddings in Postgres. You run nearest neighbor search in SQL. Two extensions cover most needs: pgvector and NeuronDB. Both ship a vector type, distance operators, and approximate indexes. NeuronDB adds a wider SQL surface for GPU, quantization, metrics, and background workers.

Share post
- You want a small extension surface. Pick pgvector.
- You want GPU SQL functions and built-in ops tables. Pick NeuronDB.
- You want ivfflat. Pick pgvector.
- You want ivf. Pick NeuronDB.
- You want halfvec above 4000 dimensions. Pick pgvector.
- You want sparsevec above 1000 nonzero entries or above 1,000,000 total dimensions. Pick pgvector.

Scope and sources
pgvector source tree: /Users/pgedge/pge/pgvector
pgvector SQL declares: PG_MODULE_MAGIC version 0.8.1
NeuronDB source tree: NeuronDB in this workspace
NeuronDB SQL declares: NeurondB v3.0.0-devel
Every limit and default below comes from these source trees.

What stays the same across both projects
You still rely on Postgres fundamentals.
You still size work_mem and maintenance_work_mem.
You still run ANALYZE after bulk load.
You still VACUUM and monitor bloat.
You still use ORDER BY distance with LIMIT to drive an index scan.

Public extension names
pgvector: extension name vector
NeuronDB: extension name neurondb

Type system

Dense float32 vector
pgvector type name: vector
NeuronDB type name: vector
Max dimension
pgvector: 16000
NeuronDB: 16000

Dense float16 vector
pgvector type name: halfvec
NeuronDB type name: halfvec
Max dimension
pgvector: 16000
NeuronDB: 4000

Sparse vectors
pgvector type name: sparsevec
NeuronDB type name: sparsevec
Text format
pgvector: {index:value,index:value}/dim with 1-based indices in text input
NeuronDB: {dim:N,idx:value,idx:value}/N format also accepted, stores data in a VectorMap layout
Hard limits
pgvector: max dim 1000000000, max nonzero entries 16000
NeuronDB: max total_dim 1000000, max nonzero entries 1000

Binary vectors and Hamming
pgvector uses Postgres built-in bit type for binary vectors, and defines distance functions for bit
NeuronDB defines a dedicated binaryvec type with a binaryvec_hamming_distance function and a distance operator for binaryvec

NeuronDB-only types in public SQL
vectorp
Storage: varlena with metadata plus float4 data
Metadata fields: CRC32 fingerprint, version, dim, endian guard
vecmap
Storage: varlena with total_dim and nnz, followed by int32 indices and float4 values
vgraph and rtext also exist, both are part of the public SQL surface

Distance operators and functions

pgvector
Dense vector operators
<-> uses l2_distance
negative inner product uses vector_negative_inner_product
<=> uses cosine_distance
<+> uses l1_distance
Bit operators
<~> uses hamming_distance(bit, bit)
<%> uses jaccard_distance(bit, bit)

NeuronDB
Dense vector operators
<-> uses vector_l2_distance_op
negative inner product uses vector_inner_product_distance_op
<=> uses vector_cosine_distance_op
<+> uses vector_l1_distance
Vector hamming and jaccard exist as functions for vector
binaryvec uses <-> for hamming distance through binaryvec_hamming_distance

Index access methods

pgvector index access method names
hnsw
ivfflat

NeuronDB index access method names
hnsw
ivf

Index hard limits that matter for planning

pgvector index limits
hnsw max dense dimensions: 2000
ivfflat max dense dimensions: 2000
hnsw sparsevec max nonzero entries: 1000

NeuronDB index limits
NeuronDB HNSW stores one node per index page.
Node size grows with dim, level, and m.
Large dim or large m leads to PageAddItem failure when a node does not fit a page.
NeuronDB IVF stores centroids as tuples on pages.
Build fails when one centroid tuple exceeds page size with the error text:
ivf: centroid size (N bytes) exceeds page size

Defaults and knobs

pgvector HNSW defaults and knobs
Index options
m default 16
ef_construction default 64
Query GUC
hnsw.ef_search default 40
Filter support
hnsw.iterative_scan values: off, relaxed_order, strict_order
hnsw.max_scan_tuples default 20000
hnsw.scan_mem_multiplier default 1

pgvector IVFFlat defaults and knobs
Index options
lists default 100
Query GUC
ivfflat.probes default 1
Filter support
ivfflat.iterative_scan values: off, relaxed_order
ivfflat.max_probes default 32768

NeuronDB HNSW defaults and knobs
Index options
m default 16
ef_construction default 200
ef_search default 64
Query GUC
neurondb.hnsw_ef_search default 64
Filter support
neurondb.hnsw_iterative_scan values: off, strict_order, relaxed_order
neurondb.hnsw_max_scan_tuples default 20000
neurondb.hnsw_scan_mem_multiplier default 1.0

NeuronDB IVF defaults and knobs
Index options
lists default 100
probes default 10
Query GUC
neurondb.ivf_probes default 10
neurondb.ivf_iterative_scan values: off, strict_order, relaxed_order
neurondb.ivf_max_probes default 100

NeuronDB also defines neurondb.ef_construction default 200 for HNSW index builds.

Practical query patterns

Baseline kNN query pattern
SELECT id, embedding <-> $1 AS distance
FROM items
ORDER BY distance
LIMIT 10

Filtered kNN
Use WHERE for filters.
Expect filters to drop rows after the index produces candidates.
Use iterative scan settings to extend the candidate search when recall drops under filters.

Type selection rules you follow in real systems

Dense embeddings from typical LLMs
Use vector for float32 embeddings.
Use halfvec only when a 4000-dimension ceiling matches your model size in NeuronDB.

Sparse features
Use sparsevec in pgvector for high total dimensions and higher nonzero counts.
Use sparsevec in NeuronDB when total_dim stays under 1,000,000 and nnz stays under 1000.
Use vecmap in NeuronDB when you want an explicit dim and nnz format with indices and values arrays.

Index selection rules you follow in real systems

Choose HNSW for steady low latency queries.
Tune m and ef_search, then measure recall and latency.
Choose IVF for large datasets with controllable probe cost.
Tune lists and probes, then measure recall and latency.

Concrete tuning workflow

Step 1: start with defaults
pgvector
hnsw.ef_search starts at 40
ivfflat.probes starts at 1
NeuronDB
neurondb.hnsw_ef_search starts at 64
neurondb.ivf_probes starts at 10

Step 2: raise the search knob
Raise ef_search for HNSW.
Raise probes for IVF and IVFFlat.
Track p95 latency and recall at k.

Step 3: rebuild only when required
Change m or ef_construction, then rebuild the HNSW index.
Change lists, then rebuild the IVF or IVFFlat index.
Change only search knobs for query-time tradeoffs.

Compute acceleration

pgvector CPU dispatch
pgvector builds alternate code paths through target_clones on supported Linux builds, then selects a path for vector math.

NeuronDB SIMD
NeuronDB compiles AVX2 and AVX-512 variants inside vector_distance_simd.c.
NeuronDB selects a SIMD path through detect_simd_capabilities when compiled with AVX2 or AVX-512 support.

NeuronDB GPU SQL surface
neurondb_gpu_enable
neurondb_gpu_info
neurondb_gpu_stats
neurondb_gpu_reset_stats
vector_l2_distance_gpu
vector_cosine_distance_gpu
vector_inner_product_gpu
hnsw_knn_search_gpu with default ef_search 100
ivf_knn_search_gpu with default nprobe 10
GPU quantization functions exist, including vector_to_int8_gpu and vector_to_fp16_gpu

Operational surface in NeuronDB

Metrics
neurondb_prometheus_metrics returns counters and gauges in a table shape.
NeuronDB SQL comments state a Prometheus HTTP endpoint on port 9187.

Worker tables and manual triggers
Schema neurondb stores tables used by workers.
Tables include job_queue, query_metrics, index_maintenance, embedding_cache.
Manual triggers include neuranq_run_once, neuranmon_sample, neurandefrag_run.

Compatibility and friction points

Index name mismatch
pgvector uses ivfflat
NeuronDB uses ivf
Your migration scripts need an index DDL change.

Halfvec max dimension mismatch
pgvector halfvec max dimension 16000
NeuronDB halfvec max dimension 4000
Large halfvec columns migrate back to vector or stay on pgvector.

Sparsevec limit mismatch
pgvector sparsevec supports larger dim and larger nnz
NeuronDB sparsevec enforces total_dim up to 1000000 and nnz up to 1000

Operator name mismatch for vector jaccard
pgvector defines jaccard_distance for bit
NeuronDB defines vector_jaccard_distance for vector

Copy and test steps you run before committing to a switch

Step 1: verify object names
SELECT extname, extversion FROM pg_extension

Step 2: verify type limits
Try one insert at the model dimension and at the boundary.
Record the exact error text on boundary failures.

Step 3: verify index build
Build the index with defaults.
Increase maintenance_work_mem for build speed.
Run ANALYZE after build.

Step 4: verify recall and latency
Fix a query set.
Measure recall at k using ground truth from an exact scan on a small sample.
Measure p50 and p95 latency.

File map for readers who want source proof
pgvector SQL surface: pgvector/sql/vector.sql
pgvector dense type and distances: pgvector/src/vector.c and pgvector/src/vector.h
pgvector HNSW defaults and limits: pgvector/src/hnsw.c and pgvector/src/hnsw.h
pgvector IVFFlat defaults and limits: pgvector/src/ivfflat.c and pgvector/src/ivfflat.h
pgvector sparsevec limits: pgvector/src/sparsevec.h
NeuronDB vector type and limits: NeuronDB/include/neurondb.h and NeuronDB/src/core/neurondb.c
NeuronDB sparsevec limits: NeuronDB/src/vector/vector_types.c
NeuronDB HNSW defaults: NeuronDB/src/index/hnsw_am.c and NeuronDB/src/util/neurondb_guc.c
NeuronDB IVF defaults: NeuronDB/src/index/ivf_am.c and NeuronDB/src/util/neurondb_guc.c
NeuronDB extension SQL: NeuronDB/sql/neurondb--3.0.0-devel.sql
