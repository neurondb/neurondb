# Gradient Boosting

NeuronDB supports XGBoost, LightGBM, and CatBoost for gradient boosting.

## XGBoost

Train an XGBoost model:

```sql
-- XGBoost classifier
SELECT train_xgboost_classifier(
    'training_table',
    'features',
    'label',
    '{"max_depth": 6, "n_estimators": 100}'::jsonb
);

-- XGBoost regressor
SELECT train_xgboost_regressor(
    'training_table',
    'features',
    'target',
    '{}'::jsonb
);
```

## LightGBM

```sql
-- LightGBM classifier
SELECT train_lightgbm_classifier(
    'training_table',
    'features',
    'label',
    '{"num_leaves": 31}'::jsonb
);
```

## CatBoost

```sql
-- CatBoost classifier
SELECT train_catboost_classifier(
    'training_table',
    'features',
    'label',
    '{}'::jsonb
);
```

## Prediction

```sql
-- Predict with trained model
SELECT id,
       xgboost_predict(features, 'model_name') AS prediction
FROM test_table;
```

## Learn More

For detailed documentation on gradient boosting algorithms, hyperparameter tuning, feature importance, and model comparison, visit:

**[Gradient Boosting Documentation](https://neurondb.ai/docs/ml/gradient-boosting/)**

## Related Topics

- [Random Forest](random-forest.md) - Ensemble methods
- [Classification](classification.md) - Classification algorithms

