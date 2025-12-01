# Model Inference

Run inference using ONNX runtime with batch processing support.

## Load Model

Load a model for inference:

```sql
-- Load ONNX model
SELECT load_model(
    'model_name',
    '/path/to/model.onnx',
    'onnx'
);
```

## Batch Inference

Run inference on batches for efficiency:

```sql
-- Batch prediction
SELECT id, features,
       model_predict_batch(features, 'model_name') AS predictions
FROM inference_table;
```

## Single Prediction

```sql
-- Single prediction
SELECT model_predict(
    '[1.0, 2.0, 3.0]'::vector,
    'model_name'
) AS prediction;
```

## Model Management

Check loaded models:

```sql
-- List loaded models
SELECT * FROM neurondb.models;

-- Get model info
SELECT model_info('model_name');
```

## Learn More

For detailed documentation on model inference, ONNX runtime, batch processing, and performance optimization, visit:

**[Model Inference Documentation](https://neurondb.ai/docs/ml/inference/)**

## Related Topics

- [Model Management](model-management.md) - Manage model lifecycle
- [Embedding Generation](embedding-generation.md) - Generate embeddings

