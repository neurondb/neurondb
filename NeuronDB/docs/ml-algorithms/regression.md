# Regression

NeuronDB provides linear and non-linear regression algorithms.

## Linear Regression

```sql
CREATE TEMP TABLE linreg_model AS
SELECT neurondb.train(
    'default',
    'linear_regression',
    'training_table',
    'target',
    ARRAY['features'],
    '{}'::jsonb
)::integer AS model_id;
```

## Ridge Regression

Regularized linear regression with L2 penalty:

```sql
CREATE TEMP TABLE ridge_model AS
SELECT neurondb.train(
    'default',
    'ridge',
    'training_table',
    'target',
    ARRAY['features'],
    '{"alpha": 0.1}'::jsonb
)::integer AS model_id;
```

## Lasso Regression

Regularized linear regression with L1 penalty:

```sql
CREATE TEMP TABLE lasso_model AS
SELECT neurondb.train(
    'default',
    'lasso',
    'training_table',
    'target',
    ARRAY['features'],
    '{"alpha": 0.1}'::jsonb
)::integer AS model_id;
```

## Prediction

```sql
SELECT neurondb.predict(
    (SELECT model_id FROM linreg_model),
    features
) AS prediction
FROM test_table;
```

## Learn More

For detailed documentation on regression algorithms, regularization, feature selection, and evaluation metrics, visit:

**[Regression Documentation](https://neurondb.ai/docs/ml/regression/)**

## Related Topics

- [Random Forest](random-forest.md) - Random Forest regression
- [Gradient Boosting](gradient-boosting.md) - Gradient boosting regression
