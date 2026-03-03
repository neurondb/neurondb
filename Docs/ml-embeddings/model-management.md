# Model Management

Load, export, version, and monitor models with catalog integration.

## Load Model

```sql
-- Load model from file
SELECT load_model(
    'my_classifier',
    '/path/to/model.onnx',
    'onnx'
);
```

## Export Model

```sql
-- Export model
SELECT export_model(
    'model_name',
    '/path/to/export.onnx'
);
```

## Version Models

Track model versions:

```sql
-- Create model version
SELECT create_model_version(
    'model_name',
    'v1.0',
    '/path/to/model.onnx'
);

-- List versions
SELECT * FROM neurondb.model_versions WHERE model_name = 'model_name';
```

## Monitor Models

```sql
-- Model statistics
SELECT * FROM neurondb.model_stats WHERE model_name = 'model_name';

-- Model performance metrics
SELECT * FROM neurondb.model_metrics WHERE model_name = 'model_name';
```

## Drop Model

```sql
-- Drop model
SELECT drop_model('model_name');
```

## Learn More

For detailed documentation on model management, versioning strategies, monitoring, and catalog integration, visit:

**[Model Management Documentation](https://neurondb.ai/docs/ml/model-management/)**

## Related Topics

- [Model Inference](model-inference.md) - Run inference
- [Embedding Generation](embedding-generation.md) - Generate embeddings

