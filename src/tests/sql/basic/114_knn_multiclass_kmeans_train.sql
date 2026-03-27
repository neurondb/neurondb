-- Regression: KNN majority vote for multiclass; unified neurondb.train('kmeans', ...) returns catalog model id.

\set ON_ERROR_STOP on

DROP TABLE IF EXISTS knn_mc_train CASCADE;
CREATE TABLE knn_mc_train (
	id serial PRIMARY KEY,
	features vector(2) NOT NULL,
	label float8 NOT NULL
);

INSERT INTO knn_mc_train (features, label) VALUES
	('[0,0]'::vector, 0),
	('[0.1,0]'::vector, 0),
	('[10,0]'::vector, 1),
	('[10.1,0]'::vector, 1),
	('[0,10]'::vector, 2),
	('[0,10.1]'::vector, 2);

SELECT (knn_classify('knn_mc_train', 'features', 'label', '[0.05,0.05]'::vector, 3) = 0) AS knn_mc_near_0;
SELECT (knn_classify('knn_mc_train', 'features', 'label', '[10.05,0.05]'::vector, 3) = 1) AS knn_mc_near_1;
SELECT (knn_classify('knn_mc_train', 'features', 'label', '[0.05,10.05]'::vector, 3) = 2) AS knn_mc_near_2;

DROP TABLE IF EXISTS kmeans_train_tbl CASCADE;
CREATE TABLE kmeans_train_tbl (
	id serial PRIMARY KEY,
	feat vector(2) NOT NULL
);

INSERT INTO kmeans_train_tbl (feat) VALUES
	('[0,0]'::vector),
	('[0.1,0.1]'::vector),
	('[0,0.1]'::vector),
	('[5,5]'::vector),
	('[5.1,5]'::vector),
	('[5,5.1]'::vector);

SELECT (neurondb.train('kmeans', 'kmeans_train_tbl', 'feat', NULL, '{"k": 2, "max_iters": 20}'::jsonb) > 0) AS kmeans_returns_model_id;

DROP TABLE knn_mc_train CASCADE;
DROP TABLE kmeans_train_tbl CASCADE;
