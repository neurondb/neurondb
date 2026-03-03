/*-------------------------------------------------------------------------
 *
 * 027_recommender_basic.sql
 *    Collaborative Filtering (ALS) basic test
 *
 *    Basic functionality test with training, prediction, and evaluation
 *
 *-------------------------------------------------------------------------*/

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

/* Step 1: Create test data */

DROP TABLE IF EXISTS cf_ratings;
CREATE TABLE cf_ratings (
    user_id INTEGER,
    item_id INTEGER,
    rating FLOAT4
);

-- Generate synthetic ratings data
INSERT INTO cf_ratings (user_id, item_id, rating)
SELECT
    (random() * 99 + 1)::INTEGER as user_id,
    (random() * 49 + 1)::INTEGER as item_id,
    (random() * 4 + 1)::FLOAT4 as rating
FROM generate_series(1, 1000);

-- Ensure some users have multiple ratings
INSERT INTO cf_ratings (user_id, item_id, rating)
SELECT
    user_id,
    (random() * 49 + 1)::INTEGER,
    (random() * 4 + 1)::FLOAT4
FROM (SELECT DISTINCT user_id FROM cf_ratings LIMIT 50) u
CROSS JOIN generate_series(1, 3);

SELECT
    COUNT(DISTINCT user_id) as users,
    COUNT(DISTINCT item_id) as items,
    COUNT(*) as ratings
FROM cf_ratings;

/* Step 2: Train collaborative filtering model */
\echo 'Step 2: Training collaborative filtering model...'

CREATE TEMP TABLE cf_model AS
SELECT train_collaborative_filter('cf_ratings', 'user_id', 'item_id', 'rating') as model_id;

SELECT * FROM cf_model;

/* Step 3: Test predictions */

-- Predict a few ratings
CREATE TEMP TABLE cf_predictions AS
SELECT
    user_id,
    item_id,
    predict_collaborative_filter((SELECT model_id FROM cf_model), user_id, item_id) as predicted_rating,
    rating as actual_rating
FROM cf_ratings
LIMIT 10;

SELECT * FROM cf_predictions;

/* Step 4: Evaluate model */
\echo 'Step 4: Evaluating collaborative filtering model...'

CREATE TEMP TABLE cf_metrics AS
SELECT evaluate_collaborative_filter_by_model_id(
    (SELECT model_id FROM cf_model),
    'cf_ratings',
    'user_id',
    'item_id',
    'rating'
) as metrics;

SELECT
    'MSE' as metric, ROUND((metrics->>'mse')::numeric, 6)::text as value
FROM cf_metrics
UNION ALL
SELECT 'MAE', ROUND((metrics->>'mae')::numeric, 6)::text
FROM cf_metrics
UNION ALL
SELECT 'RMSE', ROUND((metrics->>'rmse')::numeric, 6)::text
FROM cf_metrics
ORDER BY metric;

/* Step 5: Summary */

SELECT
    (SELECT model_id FROM cf_model) as model_id,
    (SELECT COUNT(DISTINCT user_id) FROM cf_ratings) as users,
    (SELECT COUNT(DISTINCT item_id) FROM cf_ratings) as items,
    (SELECT COUNT(*) FROM cf_ratings) as ratings,
    (SELECT ROUND((metrics->>'mse')::numeric, 6) FROM cf_metrics) as mse;

/* Cleanup */
DROP TABLE IF EXISTS cf_ratings;

\echo 'Test completed successfully'
