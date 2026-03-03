#!/usr/bin/env python3
"""
Load a Hugging Face dataset and create embeddings in NeuronDB

Example: Load a text dataset and generate embeddings for semantic search
"""

from datasets import load_dataset
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'neurondb',
    'user': 'pge',
    'password': 'test'
}

# Popular datasets you can use:
# - "squad" - Question answering dataset
# - "wikitext" - Wikipedia text
# - "ag_news" - News articles
# - "imdb" - Movie reviews
# - "glue" - Various NLP tasks

def load_and_embed_dataset(
    dataset_name="ag_news",
    split="train",
    text_column="text",
    max_rows=100,
    model_name="all-MiniLM-L6-v2"
):
    """
    Load a Hugging Face dataset and create embeddings
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split (train, test, validation)
        text_column: Column containing the text to embed
        max_rows: Maximum number of rows to process
        model_name: SentenceTransformer model name
    """
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    # Limit to max_rows
    if len(dataset) > max_rows:
        dataset = dataset.select(range(max_rows))
    
    print(f"Loaded {len(dataset)} rows")
    
    # Load embedding model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    
    print(f"Embedding dimension: {embedding_dim}")
    
    # Connect to NeuronDB
    print("Connecting to NeuronDB...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Create table
    table_name = f"{dataset_name.replace('-', '_').replace('/', '_')}_embeddings"
    print(f"Creating table: {table_name}")
    
    cur.execute(f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} (
            id SERIAL PRIMARY KEY,
            text TEXT,
            label TEXT,
            embedding vector({embedding_dim})
        );
    """)
    conn.commit()
    
    # Process and insert data
    print("Generating embeddings and inserting data...")
    for i, row in enumerate(dataset):
        if i % 10 == 0:
            print(f"Processing row {i+1}/{len(dataset)}")
        
        # Get text
        text = row.get(text_column, "")
        if not text:
            # Try common column names
            text = row.get('sentence', row.get('document', row.get('content', str(row))))
        
        # Get label if exists
        label = row.get('label', '')
        if isinstance(label, int):
            label = str(label)
        
        # Generate embedding
        embedding = model.encode(text)
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        # Insert
        cur.execute(f"""
            INSERT INTO {table_name} (text, label, embedding)
            VALUES (%s, %s, %s::vector)
        """, (text[:1000], str(label), embedding_str))  # Limit text to 1000 chars
    
    conn.commit()
    
    print(f"\n✅ Successfully loaded {len(dataset)} rows into {table_name}")
    
    # Test semantic search
    print("\n🔍 Testing semantic search...")
    test_query = "technology and computers"
    query_embedding = model.encode(test_query)
    query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
    
    cur.execute(f"""
        SELECT 
            text,
            label,
            1 - (embedding <=> %s::vector) as similarity
        FROM {table_name}
        ORDER BY embedding <=> %s::vector
        LIMIT 5
    """, (query_embedding_str, query_embedding_str))
    
    results = cur.fetchall()
    print(f"\nTop 5 results for query: '{test_query}'")
    for i, (text, label, similarity) in enumerate(results, 1):
        print(f"\n{i}. Similarity: {similarity:.4f} | Label: {label}")
        print(f"   Text: {text[:200]}...")
    
    cur.close()
    conn.close()
    
    return table_name


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Hugging Face dataset into NeuronDB")
    parser.add_argument("--dataset", default="ag_news", help="Dataset name (default: ag_news)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--text-column", default="text", help="Text column name (default: text)")
    parser.add_argument("--max-rows", type=int, default=100, help="Max rows to load (default: 100)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model (default: all-MiniLM-L6-v2)")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Hugging Face Dataset Loader for NeuronDB                   ║
║   Loading dataset with embeddings for semantic search        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        table_name = load_and_embed_dataset(
            dataset_name=args.dataset,
            split=args.split,
            text_column=args.text_column,
            max_rows=args.max_rows,
            model_name=args.model
        )
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║   ✅ SUCCESS!                                                 ║
║                                                               ║
║   Dataset loaded into table: {table_name:<30} ║
║                                                               ║
║   Try querying in NeuronDB SQL console:                      ║
║   SELECT * FROM {table_name:<37} LIMIT 10; ║
║                                                               ║
║   Or semantic search:                                        ║
║   SELECT text, 1-(embedding <=> '[...]'::vector) as sim     ║
║   FROM {table_name:<46}    ║
║   ORDER BY sim DESC LIMIT 5;                                 ║
╚══════════════════════════════════════════════════════════════╝
        """)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()






