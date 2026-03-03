# Random Forest

Random Forest is an ensemble learning method for classification and regression with GPU acceleration support.

## Classification

Train a Random Forest classifier:

```sql
CREATE TEMP TABLE rf_model AS
SELECT neurondb.train(
    'default',
    'random_forest',
    'training_table',
    'label',
    ARRAY['features'],
    '{"n_trees": 3}'::jsonb
)::integer AS model_id;
```

## Regression

Train a Random Forest regressor:

```sql
CREATE TEMP TABLE rf_model AS
SELECT neurondb.train(
    'default',
    'random_forest',
    'training_table',
    'target',
    ARRAY['features'],
    '{"n_trees": 3}'::jsonb
)::integer AS model_id;
```

## Prediction

```sql
SELECT neurondb.predict(
    (SELECT model_id FROM rf_model),
    features
) AS prediction
FROM test_table;
```

## Learn More

For detailed documentation on Random Forest parameters, hyperparameter tuning, feature importance, and GPU optimization, visit:

**[Random Forest Documentation](https://neurondb.ai/docs/ml/random-forest/)**

## Related Topics

- [Classification](classification.md) - Other classification algorithms
- [Model Management](../ml-embeddings/model-management.md) - Managing trained models
