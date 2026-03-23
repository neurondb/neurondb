-- Regression: hybrid planner hook must open relations by OID (rte->relid), not RelOptInfo.relid
-- (range-table index). When both HNSW and GIN exist, queries must not error with "OID 1".

\set ON_ERROR_STOP on

DROP TABLE IF EXISTS hybrid_planner_hook_t CASCADE;
CREATE TABLE hybrid_planner_hook_t (
	id int PRIMARY KEY,
	doc text,
	emb vector(3) NOT NULL
);

INSERT INTO hybrid_planner_hook_t
SELECT i,
	'document ' || i || ' search terms',
	ARRAY[(i % 7)::float / 7.0, ((i + 3) % 5)::float / 5.0, 0.25]::vector(3)
FROM generate_series(1, 30) AS i;

CREATE INDEX hybrid_planner_hook_t_hnsw ON hybrid_planner_hook_t
	USING hnsw (emb vector_l2_ops);

CREATE INDEX hybrid_planner_hook_t_gin ON hybrid_planner_hook_t
	USING gin (to_tsvector('english', doc));

SELECT COUNT(*)::int AS cnt FROM hybrid_planner_hook_t;

SELECT id
FROM hybrid_planner_hook_t
ORDER BY emb <-> '[0.5, 0.5, 0.25]'::vector(3)
LIMIT 5;

SELECT id
FROM hybrid_planner_hook_t
WHERE to_tsvector('english', doc) @@ plainto_tsquery('english', 'terms')
ORDER BY emb <-> '[0.5, 0.5, 0.25]'::vector(3)
LIMIT 3;

DROP TABLE hybrid_planner_hook_t CASCADE;
