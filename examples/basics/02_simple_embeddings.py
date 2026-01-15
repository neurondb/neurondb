#!/usr/bin/env python3
"""
Example 2: Text to Embeddings
=============================
Learn how to:
- Generate embeddings from text using SentenceTransformers
- Store embeddings in NeuronDB
- Use embeddings for similarity search

Run: python 02_simple_embeddings.py

Requirements:
    pip install psycopg2-binary sentence-transformers
"""

import psycopg2
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Database connection (defaults match Docker Compose setup)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5433')),  # Docker Compose default port
    'database': os.getenv('DB_NAME', 'neurondb'),
    'user': os.getenv('DB_USER', 'neurondb'),  # Docker Compose default user
    'password': os.getenv('DB_PASSWORD', 'neurondb')  # Docker Compose default password
}

# Embedding model (small, fast model for examples)
MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions

print("=" * 60)
print("Example 2: Text to Embeddings")
print("=" * 60)

# Step 1: Load embedding model
print(f"\n1. Loading embedding model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"✓ Model loaded (dimension: {embedding_dim})")

# Step 2: Connect to database
print("\n2. Connecting to database...")
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Step 3: Create table
print("\n3. Creating table...")
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS text_embeddings (
        id SERIAL PRIMARY KEY,
        text TEXT,
        embedding vector({embedding_dim})
    );
""")
conn.commit()
print("✓ Table created")

# Step 4: Generate embeddings for some texts
print("\n4. Generating embeddings...")
texts = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "PostgreSQL is a powerful database system",
    "Neural networks are used for deep learning",
]

for text in texts:
    # Generate embedding
    embedding = model.encode(text)
    
    # Convert to list format for PostgreSQL
    embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
    
    # Insert into database
    cur.execute(
        "INSERT INTO text_embeddings (text, embedding) VALUES (%s, %s::vector)",
        (text, embedding_str)
    )
    print(f"  ✓ Embedded: {text[:50]}...")

conn.commit()
print(f"✓ Inserted {len(texts)} embeddings")

# Step 5: Search for similar text
print("\n5. Searching for similar texts...")
query_text = "What is artificial intelligence?"
print(f"  Query: {query_text}")

# Generate query embedding
query_embedding = model.encode(query_text)
query_embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

# Find most similar texts
cur.execute("""
    SELECT 
        text,
        embedding <=> %s::vector AS distance,
        1 - (embedding <=> %s::vector) AS similarity
    FROM text_embeddings
    ORDER BY distance
    LIMIT 3
""", (query_embedding_str, query_embedding_str))

results = cur.fetchall()
print("\n  Most similar texts:")
for i, (text, distance, similarity) in enumerate(results, 1):
    print(f"    {i}. {text[:60]}...")
    print(f"       Similarity: {similarity:.4f}")

# Cleanup
print("\n6. Cleaning up...")
cur.execute("DROP TABLE IF EXISTS text_embeddings")
conn.commit()
print("✓ Table dropped")

cur.close()
conn.close()

print("\n" + "=" * 60)
print("Example complete! ✓")
print("=" * 60)









