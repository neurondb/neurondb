# AutoML

Automated hyperparameter tuning and model selection.

## Hyperparameter Tuning

Automatically tune model hyperparameters:

```sql
-- AutoML hyperparameter tuning
SELECT automl_tune_hyperparameters(
    'training_table',
    'features',
    'label',
    'random_forest',  -- model type
    '{"n_trees": [50, 100, 200], "max_depth": [5, 10, 15]}'::jsonb
);
```

## Model Selection

Automatically select best model:

```sql
-- AutoML model selection
SELECT automl_select_model(
    'training_table',
    'features',
    'label',
    ARRAY['random_forest', 'xgboost', 'svm']  -- candidate models
) AS best_model;
```

## AutoML Training

Train with automatic optimization:

```sql
-- Train with AutoML
SELECT automl_train(
    'training_table',
    'features',
    'label',
    'classification'
);
```

## Learn More

For detailed documentation on AutoML, hyperparameter search strategies, model selection criteria, and optimization techniques, visit:

**[AutoML Documentation](https://neurondb.ai/docs/ml/automl/)**

## Related Topics

- [Classification](../ml-algorithms/classification.md) - Classification algorithms
- [Random Forest](../ml-algorithms/random-forest.md) - Random Forest models

