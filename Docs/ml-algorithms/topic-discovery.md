# Topic Discovery

Discover hidden topics in text collections using topic modeling.

## LDA (Latent Dirichlet Allocation)

Extract topics from documents:

```sql
-- LDA topic modeling
SELECT lda_topic_discovery(
    'documents_table',
    'text_column',
    10,  -- number of topics
    '{}'::jsonb
);
```

## Topic Assignment

Assign topics to documents:

```sql
-- Get topic for each document
SELECT id, text,
       lda_get_topic(text, 'lda_model') AS topic_id,
       lda_get_topic_distribution(text, 'lda_model') AS topic_probs
FROM documents;
```

## Topic Keywords

Get keywords for each topic:

```sql
-- Get top keywords per topic
SELECT topic_id,
       lda_get_topic_keywords('lda_model', topic_id, 10) AS keywords
FROM generate_series(0, 9) AS topic_id;
```

## Learn More

For detailed documentation on topic modeling, choosing number of topics, topic interpretation, and visualization, visit:

**[Topic Discovery Documentation](https://neurondb.ai/docs/analytics/topics/)**

## Related Topics

- [Embedding Generation](../ml-embeddings/embedding-generation.md) - Generate embeddings from text
- [Clustering](clustering.md) - Cluster documents by similarity

