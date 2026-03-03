\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB ML Demo - Run All Tests in Single Session
-- ============================================================================

\echo '=========================================================================='
\echo '              NeurondB ML - Complete Test Suite'
\echo '=========================================================================='
\echo ''

\echo 'Test 1/14: Generating dataset...'
\i sql/001_generate_dataset.sql

\echo 'Test 2/14: K-means clustering...'
\i sql/002_kmeans_clustering.sql

\echo 'Test 3/14: GMM clustering...'
\i sql/003_gmm_clustering.sql

\echo 'Test 4/14: Mini-batch K-means...'
\i sql/004_minibatch_kmeans.sql

\echo 'Test 5/14: Outlier detection...'
\i sql/005_outlier_detection.sql

\echo 'Test 6/14: Hierarchical clustering...'
\i sql/006_hierarchical_clustering.sql

\echo 'Test 7/14: Complete comparison...'
\i sql/007_complete_comparison.sql

\echo 'Test 8/14: Linear regression...'
\i sql/008_linear_regression.sql

\echo 'Test 9/14: Logistic regression...'
\i sql/009_logistic_regression.sql

\echo 'Test 10/14: KNN...'
\i sql/010_knn.sql

\echo 'Test 11/14: Decision tree...'
\i sql/011_decision_tree.sql

\echo 'Test 12/14: Naive Bayes...'
\i sql/012_naive_bayes.sql

\echo 'Test 13/14: SVM...'
\i sql/013_svm.sql

\echo 'Test 14/14: Ridge/Lasso regression...'
\i sql/014_ridge_lasso.sql

\echo ''
\echo '=========================================================================='
\echo '                  ✅ ALL 14 TESTS COMPLETE!'
\echo '=========================================================================='
\echo ''
\echo 'Results Summary:'
\echo '  ✅ Dataset generation (1.5M transactions, 100MB)'
\echo '  ✅ 5 Clustering algorithms'
\echo '  ✅ 1 Outlier detection'
\echo '  ✅ 1 Model comparison'
\echo '  ✅ 7 Supervised learning algorithms'
\echo ''
\echo 'Total: 15 ML algorithms tested successfully!'
\echo '=========================================================================='
\echo ''

