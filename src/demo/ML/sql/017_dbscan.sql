\set ON_ERROR_STOP on
\set QUIET on

-- ============================================================================
-- DBSCAN Clustering Demo - Density-Based Spatial Clustering
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS neurondb;

\echo '══════════════════════════════════════════════════════════════════'
\echo '  DBSCAN - Density-Based Spatial Clustering of Applications with Noise'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

-- Step 1: Generate spatial dataset with clusters and noise
\echo 'Step 1: Generating spatial dataset...'
\echo '  • 3,000 data points'
\echo '  • 2D spatial features'
\echo '  • Multiple density regions + noise'
\echo ''

DROP TABLE IF EXISTS dbscan_data CASCADE;
CREATE TABLE dbscan_data AS
SELECT 
    i as id,
    CASE 
        WHEN i % 3 = 0 THEN 
            ARRAY[(random() * 20 + 10)::real, (random() * 20 + 10)::real]
        WHEN i % 3 = 1 THEN 
            ARRAY[(random() * 20 + 50)::real, (random() * 20 + 50)::real]
        ELSE 
            ARRAY[(random() * 100)::real, (random() * 100)::real]
    END::vector(2) as features
FROM generate_series(1, 3000) i;

\echo '  ✓ Dataset ready'
\echo ''

-- Step 2: Run DBSCAN Clustering
\echo 'Step 2: Running DBSCAN clustering...'
\echo '  • Algorithm: Density-Based Clustering'
\echo '  • eps (radius): 5.0'
\echo '  • min_points: 10'
\echo '  • Identifies clusters AND noise points'
\echo ''

\timing on
CREATE TEMP TABLE dbscan_results AS
SELECT 
    id,
    features,
    cluster_dbscan('dbscan_data', 'features', 5.0, 10, id) as cluster_id
FROM dbscan_data
WHERE features IS NOT NULL;
\timing off

\echo ''

-- Step 3: Analyze Cluster Distribution
\echo 'Step 3: Analyzing cluster distribution...'
\echo ''

SELECT 
    CASE 
        WHEN cluster_id = -1 THEN 'NOISE'
        ELSE 'Cluster ' || cluster_id
    END as cluster_label,
    COUNT(*) as point_count,
    ROUND((100.0 * COUNT(*) / (SELECT COUNT(*) FROM dbscan_results))::numeric, 2) as percentage
FROM dbscan_results
GROUP BY cluster_id
ORDER BY 
    CASE WHEN cluster_id = -1 THEN 999999 ELSE cluster_id END;

\echo ''

-- Step 4: Cluster Statistics
\echo 'Step 4: Computing cluster statistics...'
\echo ''

WITH cluster_stats AS (
    SELECT 
        cluster_id,
        COUNT(*) as size,
        AVG((features::text::float8[])[1]) as avg_x,
        AVG((features::text::float8[])[2]) as avg_y,
        STDDEV((features::text::float8[])[1]) as stddev_x,
        STDDEV((features::text::float8[])[2]) as stddev_y
    FROM dbscan_results
    WHERE cluster_id >= 0
    GROUP BY cluster_id
)
SELECT 
    cluster_id,
    size,
    ROUND(avg_x::numeric, 2) as center_x,
    ROUND(avg_y::numeric, 2) as center_y,
    ROUND(stddev_x::numeric, 2) as spread_x,
    ROUND(stddev_y::numeric, 2) as spread_y
FROM cluster_stats
ORDER BY cluster_id;

\echo ''

-- Step 5: Density Analysis
\echo 'Step 5: Density analysis...'
\echo ''

WITH density AS (
    SELECT 
        cluster_id,
        COUNT(*) as points,
        COUNT(*) * 1.0 / NULLIF(
            (MAX((features::text::float8[])[1]) - MIN((features::text::float8[])[1])) *
            (MAX((features::text::float8[])[2]) - MIN((features::text::float8[])[2])), 
            0
        ) as density
    FROM dbscan_results
    WHERE cluster_id >= 0
    GROUP BY cluster_id
)
SELECT 
    cluster_id,
    points,
    ROUND(density::numeric, 6) as density_score,
    CASE 
        WHEN density > 1.0 THEN 'High Density'
        WHEN density > 0.5 THEN 'Medium Density'
        ELSE 'Low Density'
    END as density_class
FROM density
ORDER BY cluster_id;

\echo ''

-- Step 6: Noise Point Analysis
\echo 'Step 6: Analyzing noise points...'
\echo ''

SELECT 
    COUNT(*) as total_noise_points,
    ROUND((100.0 * COUNT(*) / (SELECT COUNT(*) FROM dbscan_results))::numeric, 2) as noise_percentage,
    ROUND(AVG((features::text::float8[])[1])::numeric, 2) as avg_noise_x,
    ROUND(AVG((features::text::float8[])[2])::numeric, 2) as avg_noise_y
FROM dbscan_results
WHERE cluster_id = -1;

\echo ''
\echo '══════════════════════════════════════════════════════════════════'
\echo '  DBSCAN Demo Complete!'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''
\echo 'Key Advantages of DBSCAN:'
\echo '  • Discovers clusters of arbitrary shape'
\echo '  • Automatically identifies noise points'
\echo '  • No need to specify number of clusters'
\echo '  • Robust to outliers'
\echo '  • Handles varying cluster densities'
\echo ''
\echo 'Parameters:'
\echo '  • eps: Maximum distance between two points to be neighbors'
\echo '  • min_points: Minimum points to form a dense region'
\echo ''
\echo '══════════════════════════════════════════════════════════════════'

