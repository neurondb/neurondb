# Classification

NeuronDB supports multiple classification algorithms.

## Logistic Regression

```sql
CREATE TEMP TABLE logreg_model AS
SELECT neurondb.train(
    'default',
    'logistic_regression',
    'training_table',
    'label',
    ARRAY['features'],
    '{}'::jsonb
)::integer AS model_id;
```

## Prediction

```sql
SELECT neurondb.predict(
    (SELECT model_id FROM logreg_model),
    features
) AS prediction
FROM test_table;
```

## Other Algorithms

NeuronDB also supports:
- SVM (Support Vector Machine)
- Naive Bayes
- Decision Trees
- Neural Networks

## Learn More

For detailed documentation on classification algorithms, evaluation metrics, hyperparameter tuning, and multi-class classification, visit:

**[Classification Documentation](https://neurondb.ai/docs/ml/classification/)**

## Related Topics

- [Random Forest](random-forest.md) - Ensemble classification
- [Quality Metrics](quality-metrics.md) - Classification metrics
