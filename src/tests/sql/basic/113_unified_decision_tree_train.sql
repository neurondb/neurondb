-- Regression: neurondb.train(..., 'decision_tree', ...) must not crash (direct C call, no nested SPI).

\set ON_ERROR_STOP on

DROP TABLE IF EXISTS dt_unified_train CASCADE;
CREATE TABLE dt_unified_train (
	id serial PRIMARY KEY,
	features vector(2) NOT NULL,
	label int NOT NULL
);

INSERT INTO dt_unified_train (features, label) VALUES
	('[0,0]'::vector, 0),
	('[1,0]'::vector, 0),
	('[0,1]'::vector, 1),
	('[1,1]'::vector, 1),
	('[0.5,0.5]'::vector, 1);

SELECT (neurondb.train('default', 'decision_tree', 'dt_unified_train', 'label', ARRAY['features'], '{}'::jsonb) > 0) AS got_model_id;

DROP TABLE dt_unified_train CASCADE;
