# Recommendation Systems

Build recommendation systems using collaborative filtering and ranking.

## Collaborative Filtering

Train collaborative filtering model:

```sql
CREATE TABLE cf_ratings (
    user_id INTEGER,
    item_id INTEGER,
    rating FLOAT4
);

CREATE TEMP TABLE cf_model AS
SELECT train_collaborative_filter('cf_ratings', 'user_id', 'item_id', 'rating') AS model_id;
```

## Generate Recommendations

```sql
SELECT
    user_id,
    item_id,
    predict_collaborative_filter((SELECT model_id FROM cf_model), user_id, item_id) AS predicted_rating
FROM cf_ratings
LIMIT 10;
```

## Learn More

For detailed documentation on recommendation algorithms, evaluation metrics, cold start problems, and hybrid recommendation systems, visit:

**[Recommendation Systems Documentation](https://neurondb.ai/docs/ml/recommender/)**

## Related Topics

- [Vector Search](../vector-search/indexing.md) - Similarity-based recommendations
- [Quality Metrics](quality-metrics.md) - Evaluate recommendation quality

