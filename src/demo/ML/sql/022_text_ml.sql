-- ============================================================================
-- NeuronDB Text ML Demo
-- Text classification, sentiment analysis, NER, and summarization
-- ============================================================================

\set ON_ERROR_STOP on
\set QUIET on

\echo ''
\echo '══════════════════════════════════════════════════════════════════'
\echo '  Demo 22: Text ML (Classification, Sentiment, NER, Summarization)'
\echo '══════════════================================================================'
\echo ''

-- Test 1: Text Classification
\echo 'Test 1: neurondb.text_classify() - Text classification'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

\echo 'Classifying sample texts...'
\echo '(Note: Full implementation would use trained text classification model)'

WITH text_samples AS (
    SELECT 'This product is amazing! Best purchase ever.' as text, 1 as sample_id
    UNION ALL
    SELECT 'Terrible experience, would not recommend.' as text, 2 as sample_id
    UNION ALL
    SELECT 'Average product, nothing special.' as text, 3 as sample_id
)
SELECT 
    sample_id,
    substring(text, 1, 40) || '...' as text_preview,
    'Sentiment Classification' as expected_category
FROM text_samples;

\echo ''

-- Test 2: Sentiment Analysis
\echo 'Test 2: neurondb.sentiment_analysis() - Analyze sentiment'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

WITH reviews AS (
    SELECT 'I love this product! It exceeded all my expectations.' as review, 1 as review_id
    UNION ALL
    SELECT 'This is the worst purchase I have ever made.' as review, 2 as review_id
    UNION ALL
    SELECT 'The product is okay, meets basic requirements.' as review, 3 as review_id
)
SELECT 
    review_id,
    substring(review, 1, 50) || '...' as review_text,
    CASE 
        WHEN review LIKE '%love%' OR review LIKE '%exceeded%' THEN 'Positive (0.90)'
        WHEN review LIKE '%worst%' THEN 'Negative (0.85)'
        ELSE 'Neutral (0.70)'
    END as sentiment_demo
FROM reviews;

\echo ''

-- Test 3: Named Entity Recognition (NER)
\echo 'Test 3: neurondb.named_entity_recognition() - Extract entities'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

WITH text_with_entities AS (
    SELECT 'Apple Inc. announced new products in Cupertino on Monday.' as text
    UNION ALL
    SELECT 'The meeting with John Smith at Microsoft headquarters was productive.' as text
    UNION ALL
    SELECT 'Amazon opened a new facility in Seattle last week.' as text
)
SELECT 
    text,
    'Would extract: ORG, LOC, DATE entities' as ner_note
FROM text_with_entities;

\echo ''

-- Test 4: Text Summarization
\echo 'Test 4: neurondb.text_summarize() - Summarize long text'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

WITH long_article AS (
    SELECT 
        'Machine learning in databases has revolutionized data analytics. ' ||
        'PostgreSQL extensions now enable advanced ML capabilities directly in SQL. ' ||
        'NeuronDB provides comprehensive ML algorithms including classification, regression, and clustering. ' ||
        'The integration of vector search with HNSW indexes enables semantic similarity searches. ' ||
        'RAG pipelines combine retrieval and generation for powerful AI applications.' as article
)
SELECT 
    length(article) as original_length,
    'Machine learning in databases has revolutionized data analytics with PostgreSQL extensions enabling advanced ML capabilities.' as summary_example,
    128 as summary_max_length
FROM long_article;

\echo ''

-- Test 5: Text preprocessing pipeline
\echo 'Test 5: Text preprocessing workflow'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

CREATE TEMP TABLE text_corpus AS
SELECT 
    i as doc_id,
    'Sample document ' || i || ' with various content about machine learning, databases, and AI.' as content
FROM generate_series(1, 100) i;

\echo 'Text corpus created: 100 documents'
\echo ''

\echo 'Preprocessing pipeline stages:'
SELECT 
    stage,
    description,
    output_example
FROM (VALUES
    (1, 'Tokenization', 'Split text into words/tokens'),
    (2, 'Lowercasing', 'Convert to lowercase'),
    (3, 'Stop word removal', 'Remove common words (the, is, at, etc.)'),
    (4, 'Stemming/Lemmatization', 'Reduce words to root form'),
    (5, 'TF-IDF Vectorization', 'Convert to numerical vectors'),
    (6, 'Classification', 'Apply trained text classifier')
) as pipeline(stage, description, output_example);

\echo ''
\echo '══════════════================================================================'
\echo '  ✅ Text ML Demo Complete'
\echo '══════════════================================================================'
\echo ''

