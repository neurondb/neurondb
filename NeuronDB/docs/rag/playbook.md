# RAG Playbook

This playbook provides practical guidance for building Retrieval-Augmented Generation (RAG) systems with NeuronDB, covering chunking strategies, embedding model selection, context management, and evaluation.

## Chunking Guidance

### Chunking Strategies

#### 1. Fixed-Size Chunking

**Best for:** Uniform documents, simple text

```python
def fixed_size_chunks(text, chunk_size=512, overlap=50):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap for context
    return chunks
```

**Parameters:**
- **chunk_size:** 256-1024 tokens (typical: 512)
- **overlap:** 10-20% of chunk_size (typical: 50-100 tokens)

#### 2. Sentence-Aware Chunking

**Best for:** Natural language, preserving sentence boundaries

```python
import re

def sentence_chunks(text, max_chunk_size=512, overlap_sentences=2):
    """Split by sentences, respecting max size."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Keep last N sentences for overlap
            current_chunk = current_chunk[-overlap_sentences:]
            current_size = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

#### 3. Semantic Chunking

**Best for:** Long documents, preserving semantic coherence

```python
from transformers import AutoTokenizer, AutoModel
import torch

def semantic_chunks(text, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                   max_chunk_size=512, similarity_threshold=0.7):
    """Split by semantic similarity boundaries."""
    # Use embedding model to find natural break points
    # Implementation depends on specific model
    pass
```

#### 4. Document-Structure Chunking

**Best for:** Structured documents (PDFs, markdown, HTML)

```python
def markdown_chunks(markdown_text, max_chunk_size=512):
    """Split markdown by headers, preserving structure."""
    import re
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Split by headers
    sections = re.split(r'(^#+\s+.+$)', markdown_text, flags=re.MULTILINE)
    
    for section in sections:
        if section.startswith('#'):
            # Header - start new chunk if current is large enough
            if current_size > max_chunk_size * 0.8:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [section]
                current_size = len(section.split())
            else:
                current_chunk.append(section)
                current_size += len(section.split())
        else:
            current_chunk.append(section)
            current_size += len(section.split())
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks
```

### Chunk Size Recommendations

| Document Type | Recommended Chunk Size | Overlap | Reasoning |
|--------------|------------------------|---------|-----------|
| Code | 200-400 tokens | 20-40 tokens | Preserve function/class boundaries |
| Technical docs | 400-600 tokens | 50-100 tokens | Balance context and precision |
| Articles/Blogs | 500-800 tokens | 50-100 tokens | Paragraph/section boundaries |
| Books | 600-1000 tokens | 100-200 tokens | Chapter/section coherence |
| Legal docs | 300-500 tokens | 50-100 tokens | Preserve clause boundaries |
| Conversations | 200-400 tokens | 20-40 tokens | Preserve turn boundaries |

### Overlap Guidelines

**Why overlap matters:**
- Preserves context across chunk boundaries
- Improves recall for queries spanning chunks
- Reduces information loss at boundaries

**Overlap recommendations:**
- **Small chunks (< 300 tokens):** 10-20% overlap (20-60 tokens)
- **Medium chunks (300-600 tokens):** 15-25% overlap (50-150 tokens)
- **Large chunks (> 600 tokens):** 20-30% overlap (120-300 tokens)

## Embedding Model Selection

### Model Comparison Table

| Model | Dimensions | Max Tokens | Speed | Quality | Cost | Use Case |
|-------|------------|------------|-------|---------|------|----------|
| `text-embedding-ada-002` | 1536 | 8191 | Fast | High | $$ | General purpose, OpenAI |
| `text-embedding-3-small` | 1536 | 8191 | Fast | High | $$ | OpenAI, improved quality |
| `text-embedding-3-large` | 3072 | 8191 | Medium | Very High | $$$ | Best quality, OpenAI |
| `all-MiniLM-L6-v2` | 384 | 256 | Very Fast | Medium | Free | Fast, local, good quality |
| `all-mpnet-base-v2` | 768 | 384 | Medium | High | Free | Best open-source quality |
| `e5-large-v2` | 1024 | 512 | Medium | High | Free | Multilingual, good quality |
| `BGE-large-en-v1.5` | 1024 | 512 | Medium | Very High | Free | State-of-the-art open-source |

### Dimension Trade-offs

**Lower dimensions (384-512):**
- ✅ Faster queries
- ✅ Less storage
- ✅ Lower cost
- ❌ Slightly lower quality
- **Best for:** Large-scale systems, speed-critical applications

**Medium dimensions (768-1024):**
- ✅ Good quality/speed balance
- ✅ Reasonable storage
- ✅ Moderate cost
- **Best for:** Most production applications

**Higher dimensions (1536-3072):**
- ✅ Highest quality
- ✅ Better semantic understanding
- ❌ Slower queries
- ❌ More storage
- ❌ Higher cost
- **Best for:** Quality-critical applications, complex queries

### Cost Considerations

**OpenAI models:**
- `text-embedding-ada-002`: $0.0001 per 1K tokens
- `text-embedding-3-small`: $0.00002 per 1K tokens
- `text-embedding-3-large`: $0.00013 per 1K tokens

**Open-source models:**
- Free to run locally
- GPU required for reasonable speed
- Consider hosting costs if using cloud

**Cost calculation example:**
```
1M documents × 500 tokens/doc = 500M tokens
Cost (ada-002) = 500M / 1000 × $0.0001 = $50
Cost (3-small) = 500M / 1000 × $0.00002 = $10
```

### Model Selection Decision Tree

```
Start
  │
  ├─ Need highest quality?
  │   ├─ Yes → text-embedding-3-large or BGE-large
  │   └─ No → Continue
  │
  ├─ Budget constrained?
  │   ├─ Yes → all-MiniLM-L6-v2 or all-mpnet-base-v2
  │   └─ No → Continue
  │
  ├─ Need multilingual?
  │   ├─ Yes → e5-large-v2 or multilingual models
  │   └─ No → Continue
  │
  └─ General purpose → text-embedding-3-small or all-mpnet-base-v2
```

## Context Window Management

### Token Limits

**Model context windows:**
- GPT-3.5-turbo: 16K tokens
- GPT-4: 128K tokens
- Claude 3: 200K tokens
- Llama 2: 4K-32K tokens

### Truncation Strategies

#### 1. Simple Truncation

```python
def truncate_context(contexts, max_tokens=4000):
    """Truncate to fit within token limit."""
    total_tokens = 0
    selected = []
    
    for context in contexts:
        tokens = len(context.split())  # Approximate
        if total_tokens + tokens > max_tokens:
            break
        selected.append(context)
        total_tokens += tokens
    
    return selected
```

#### 2. Priority-Based Truncation

```python
def priority_truncate(contexts_with_scores, max_tokens=4000):
    """Keep highest-scoring contexts first."""
    # Sort by relevance score
    sorted_contexts = sorted(contexts_with_scores, 
                            key=lambda x: x['score'], 
                            reverse=True)
    
    total_tokens = 0
    selected = []
    
    for item in sorted_contexts:
        tokens = len(item['text'].split())
        if total_tokens + tokens > max_tokens:
            break
        selected.append(item['text'])
        total_tokens += tokens
    
    return selected
```

#### 3. Sliding Window Truncation

```python
def sliding_window(contexts, max_tokens=4000, window_size=500):
    """Keep most relevant parts of each context."""
    selected = []
    total_tokens = 0
    
    for context in contexts:
        if total_tokens >= max_tokens:
            break
        
        # If context fits, use it all
        context_tokens = len(context.split())
        if total_tokens + context_tokens <= max_tokens:
            selected.append(context)
            total_tokens += context_tokens
        else:
            # Truncate to fit remaining space
            remaining = max_tokens - total_tokens
            truncated = ' '.join(context.split()[:remaining])
            selected.append(truncated)
            break
    
    return selected
```

### Multi-Pass Retrieval

**Strategy:** Retrieve in multiple passes, refining query each time

```python
def multi_pass_retrieval(query, initial_results, db_connection):
    """Refine retrieval with follow-up queries."""
    # Pass 1: Initial retrieval
    results_1 = vector_search(query, k=20)
    
    # Pass 2: Expand query with top results
    expanded_query = expand_query_with_context(query, results_1[:5])
    results_2 = vector_search(expanded_query, k=10)
    
    # Pass 3: Hybrid search combining original and expanded
    final_results = hybrid_search(query, expanded_query, k=10)
    
    return final_results
```

## Evaluation Harness

### Metrics

#### 1. Precision@K

**Definition:** Fraction of retrieved items that are relevant

```python
def precision_at_k(retrieved, relevant, k):
    """Calculate precision at k."""
    retrieved_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_k) & set(relevant))
    return relevant_retrieved / k if k > 0 else 0
```

#### 2. Recall@K

**Definition:** Fraction of relevant items that are retrieved

```python
def recall_at_k(retrieved, relevant, k):
    """Calculate recall at k."""
    retrieved_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_k) & set(relevant))
    return relevant_retrieved / len(relevant) if len(relevant) > 0 else 0
```

#### 3. Mean Reciprocal Rank (MRR)

**Definition:** Average of reciprocal ranks of first relevant result

```python
def mean_reciprocal_rank(queries_results, relevant_sets):
    """Calculate MRR."""
    reciprocal_ranks = []
    
    for query_results, relevant in zip(queries_results, relevant_sets):
        for rank, result in enumerate(query_results, 1):
            if result in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
```

#### 4. Normalized Discounted Cumulative Gain (NDCG)

**Definition:** Measures ranking quality with position discounts

```python
import math

def ndcg_at_k(retrieved, relevant_scores, k):
    """Calculate NDCG at k."""
    dcg = 0.0
    for i, score in enumerate(retrieved[:k], 1):
        dcg += score / math.log2(i + 1)
    
    # Ideal DCG
    ideal_scores = sorted(relevant_scores, reverse=True)[:k]
    idcg = sum(score / math.log2(i + 1) 
               for i, score in enumerate(ideal_scores, 1))
    
    return dcg / idcg if idcg > 0 else 0.0
```

### Evaluation Queries

**Create test set:**
```sql
-- Store evaluation queries and expected results
CREATE TABLE eval_queries (
  id SERIAL PRIMARY KEY,
  query_text TEXT,
  expected_doc_ids INTEGER[],
  category TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Example queries
INSERT INTO eval_queries (query_text, expected_doc_ids, category) VALUES
  ('What is machine learning?', ARRAY[1, 5, 12], 'definition'),
  ('How to train a neural network?', ARRAY[3, 7, 15], 'how-to'),
  ('Vector database performance', ARRAY[8, 11, 20], 'technical');
```

**Run evaluation:**
```python
def evaluate_retrieval(eval_queries, retrieval_function):
    """Evaluate retrieval system."""
    results = {
        'precision@5': [],
        'precision@10': [],
        'recall@5': [],
        'recall@10': [],
        'mrr': []
    }
    
    for query_row in eval_queries:
        query = query_row['query_text']
        expected = set(query_row['expected_doc_ids'])
        
        # Retrieve results
        retrieved = retrieval_function(query, k=10)
        retrieved_ids = [r['id'] for r in retrieved]
        
        # Calculate metrics
        results['precision@5'].append(
            precision_at_k(retrieved_ids, expected, 5))
        results['precision@10'].append(
            precision_at_k(retrieved_ids, expected, 10))
        results['recall@5'].append(
            recall_at_k(retrieved_ids, expected, 5))
        results['recall@10'].append(
            recall_at_k(retrieved_ids, expected, 10))
        results['mrr'].append(
            1.0 / (retrieved_ids.index(list(expected)[0]) + 1) 
            if expected & set(retrieved_ids) else 0.0)
    
    # Calculate averages
    return {metric: sum(values) / len(values) 
            for metric, values in results.items()}
```

### Target Metrics

**Production targets:**
- **Precision@10:** > 0.7 (70% of top 10 are relevant)
- **Recall@10:** > 0.8 (80% of relevant items in top 10)
- **MRR:** > 0.8 (first relevant result in top 2 on average)
- **NDCG@10:** > 0.75 (good ranking quality)

## RAG Implementation Checklist

- [ ] **Chunking strategy:** Selected and implemented
- [ ] **Chunk size:** Tuned for document type
- [ ] **Overlap:** Configured appropriately
- [ ] **Embedding model:** Selected based on requirements
- [ ] **Embedding generation:** Batch processing implemented
- [ ] **Vector storage:** Indexes created and tuned
- [ ] **Retrieval:** kNN search implemented
- [ ] **Context management:** Truncation strategy defined
- [ ] **LLM integration:** Context formatting implemented
- [ ] **Evaluation:** Test set created and metrics calculated
- [ ] **Monitoring:** Retrieval quality tracked

## Related Documentation

- [RAG Overview](overview.md) - RAG concepts and architecture
- [Document Processing](document-processing.md) - Document ingestion
- [LLM Integration](llm-integration.md) - LLM setup and usage
- [Vector Search](vector-search/indexing.md) - Index configuration







