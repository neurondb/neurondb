# Feature Store

Centralized feature management and versioning.

## Create Feature Store

```sql
-- Create feature store
SELECT create_feature_store(
    'my_feature_store',
    '{"description": "Features"}'::jsonb
);
```

## Register Features

```sql
-- Register feature
SELECT register_feature(
    'feature_store_name',
    'user_age',
    'numeric',
    '{"source": "users_table"}'::jsonb
);
```

## Get Features

```sql
-- Get features for entities
SELECT get_features(
    'feature_store_name',
    ARRAY['user_1', 'user_2'],
    ARRAY['user_age', 'user_rating']
) AS features;
```

## Version Features

```sql
-- Create feature version
SELECT create_feature_version(
    'feature_store_name',
    'user_age',
    'v2',
    'numeric'
);
```

## Learn More

For detailed documentation on feature stores, feature versioning, feature serving, and governance, visit:

**[Feature Store Documentation](https://neurondb.ai/docs/ml/feature-store/)**

## Related Topics

- [Model Management](model-management.md) - Use features in models
- [Embedding Generation](embedding-generation.md) - Feature engineering

