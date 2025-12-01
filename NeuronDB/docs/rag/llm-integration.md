# LLM Integration

Hugging Face and OpenAI integration for LLM-powered features.

## OpenAI Integration

Use OpenAI models:

```sql
-- OpenAI completion
SELECT openai_complete(
    'What is AI?',
    'gpt-4',
    '{"temperature": 0.7}'::jsonb
) AS response;
```

## Hugging Face Integration

Use Hugging Face models:

```sql
-- Hugging Face inference
SELECT huggingface_inference(
    'What is machine learning?',
    'microsoft/DialoGPT-medium'
) AS response;
```

## Configure Providers

```sql
-- Set OpenAI API key
SET neurondb.openai_api_key = 'your-api-key';

-- Set Hugging Face endpoint
SET neurondb.huggingface_endpoint = 'https://api-inference.huggingface.co';
```

## Learn More

For detailed documentation on LLM integration, provider configuration, model selection, and cost optimization, visit:

**[LLM Integration Documentation](https://neurondb.ai/docs/ml/llm/)**

## Related Topics

- [RAG Overview](overview.md) - RAG pipeline
- [Document Processing](document-processing.md) - Text processing

