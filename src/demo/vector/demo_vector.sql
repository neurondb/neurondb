-- NeuronDB Vector demo SQL
-- Location: Neurondb/demo/vector/demo_vector.sql
-- This demo shows how to use NeuronDB's native `vector` type and APIs.
-- It demonstrates creating a table with `vector`, inserting vectors via
-- `embed_text` and `array_to_vector`, building an HNSW index, and running
-- KNN queries using NeuronDB operators (`<->`, `<=>`, `<#>`).

-- If you cannot load the NeuronDB extension, a float8[] fallback is
-- described near the end (commented) so you can still run similar queries.

-- ====== 1) Create demo table using NeuronDB's native vector type ======
-- Here we create an 8-dimensional vector column. If you prefer dynamic
-- dimensions, omit the typmod (use `vector`).
CREATE TABLE IF NOT EXISTS documents (
  id serial PRIMARY KEY,
  title text,
  embedding vector(8)
);

-- ====== 2) Insert demo vectors ======
-- You can generate embeddings with NeuronDB's `embed_text()` or convert
-- an array of floats via `array_to_vector(real[])`.
INSERT INTO documents (title, embedding) VALUES
  ('apple', array_to_vector(ARRAY[0.21, 0.15, 0.05, 0.6, 0.02, 0.0, 0.0, 0.1]::real[])),
  ('banana', array_to_vector(ARRAY[0.18, 0.12, 0.02, 0.55, 0.05, 0.0, 0.01, 0.12]::real[])),
  ('car', array_to_vector(ARRAY[0.01, 0.9, 0.3, 0.0, 0.0, 0.05, 0.0, 0.0]::real[])),
  ('vehicle', array_to_vector(ARRAY[0.0, 0.88, 0.28, 0.01, 0.0, 0.03, 0.0, 0.0]::real[]));

-- Example: generate embedding from text using the built-in embedding function
-- (uses configured model; default is 'all-MiniLM-L6-v2').
INSERT INTO documents (title, embedding)
  VALUES ('orange', embed_text('orange'));

-- ====== 3) Indexing (HNSW) ======
-- Create an HNSW index for fast ANN search. Use the distance operator
-- class appropriate for your metric (e.g., vector_l2_ops for L2).
-- Example tuning: m (connectivity) and ef_construction (index build ef).
CREATE INDEX IF NOT EXISTS documents_hnsw_idx
  ON documents USING hnsw (embedding vector_l2_ops)
  WITH (m = 16, ef_construction = 200);

-- Note: You can also use `vector_cosine_distance` operator class for cosine.

-- ====== 4) Query examples (use NeuronDB native operators) ======
-- Prepare a query vector (use embed_text in real workloads):
WITH q AS (SELECT embed_text('fruit vehicle query') AS query_vec)

-- Top-3 by L2 distance (closest first) using the `<->` operator:
SELECT d.id, d.title, d.embedding <-> q.query_vec AS l2_dist
FROM documents d, q
ORDER BY l2_dist ASC
LIMIT 3;

-- Top-3 by Cosine distance (use `<=>` operator; lower = closer when using distances):
WITH q AS (SELECT embed_text('fast vehicle') AS query_vec)
SELECT d.id, d.title, d.embedding <=> q.query_vec AS cosine_dist
FROM documents d, q
ORDER BY cosine_dist ASC
LIMIT 3;

-- Top-3 by Inner Product (use `<#>` operator; higher inner product often means higher similarity
-- depending on how your vectors are normalized; note `vector_inner_product` returns a real):
WITH q AS (SELECT array_to_vector(ARRAY[0.2,0.14,0.04,0.59,0.03,0.0,0.0,0.11]::real[]) AS query_vec)
SELECT d.id, d.title, d.embedding <#> q.query_vec AS inner_prod
FROM documents d, q
ORDER BY inner_prod DESC
LIMIT 3;

-- ====== 5) Hybrid search / convenience functions ======
-- Use the extension's `hybrid_search` to combine text + vector ranking:
-- SELECT * FROM hybrid_search('documents', embed_text('query text'), 'title', '{}', 0.7, 5);

-- ====== 6) Upsert helper using native vector type ======
CREATE OR REPLACE FUNCTION upsert_document(_id int, _title text, _emb vector) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  IF _id IS NULL THEN
    INSERT INTO documents (title, embedding) VALUES (_title, _emb);
  ELSE
    INSERT INTO documents (id, title, embedding) VALUES (_id, _title, _emb)
      ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, embedding = EXCLUDED.embedding;
  END IF;
END;
$$;

-- Example: update an existing doc using an embedding produced by the extension
SELECT upsert_document(1, 'apple (updated)', embed_text('apple'));

-- ====== 7) KNN wrapper using NeuronDB operators ======
CREATE OR REPLACE FUNCTION knn_search_native(query vector, k int) RETURNS TABLE(id int, title text, distance real) LANGUAGE sql AS $$
  SELECT d.id, d.title, d.embedding <-> query AS distance
  FROM documents d
  ORDER BY distance ASC
  LIMIT k;
$$;

-- Example call (use embed_text to form the query vector):
SELECT * FROM knn_search_native(embed_text('find similar to car'), 3);

-- ====== 8) Optional fallback: float8[] (if NeuronDB extension not available) ======
-- If you cannot load the NeuronDB extension, uncomment and use the fallback
-- table and helper functions below (they are similar to the earlier demo).
--
-- CREATE TABLE documents_fallback (
--   id serial PRIMARY KEY,
--   title text,
--   embedding float8[]
-- );
--
-- -- Convert float8[] to vector using array_to_vector(ARRAY[...]::real[])
-- -- Or keep using vec_cosine/vec_l2 helpers implemented previously.

-- ====== 9) Cleanup (optional) ======
-- DROP FUNCTION knn_search_native(vector, int);
-- DROP FUNCTION upsert_document(int, text, vector);
-- DROP INDEX IF EXISTS documents_hnsw_idx;
-- DROP TABLE IF EXISTS documents;

-- End of NeuronDB-native demo
