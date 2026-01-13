#!/usr/bin/env python3
"""
Indexing: HNSW Index
====================
Learn how to create and use HNSW indexes for fast vector search.

Run: python 01_hnsw_index.py
"""

import psycopg2
import os
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5433')),  # Docker Compose default port
    'database': os.getenv('DB_NAME', 'neurondb'),
    'user': os.getenv('DB_USER', 'neurondb'),  # Docker Compose default user
    'password': os.getenv('DB_PASSWORD', 'neurondb')  # Docker Compose default password
}

MODEL_NAME = "all-MiniLM-L6-v2"

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("=" * 60)
print("Indexing: HNSW Index")
print("=" * 60)

# Load model
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()

# Create table
print("\n1. Creating table with vectors...")
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector({embedding_dim})
    );
""")
conn.commit()
print("✓ Table created")

# Insert sample data
print("\n2. Inserting sample documents...")
texts = [
    "Machine learning algorithms",
    "Python programming language",
    "Database systems",
    "Neural networks",
    "Vector embeddings",
]

for text in texts:
    embedding = model.encode(text)
    embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector)",
        (text, embedding_str)
    )
conn.commit()
print(f"✓ Inserted {len(texts)} documents")

# Create HNSW index
print("\n3. Creating HNSW index...")
print("  Parameters:")
print("    - m = 16 (number of connections per node)")
print("    - ef_construction = 64 (index build quality)")
cur.execute(f"""
    CREATE INDEX documents_embedding_idx 
    ON documents 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")
conn.commit()
print("✓ HNSW index created")

# Show index info
print("\n4. Index information:")
cur.execute("""
    SELECT 
        indexname,
        indexdef
    FROM pg_indexes
    WHERE tablename = 'documents';
""")
for idx_name, idx_def in cur.fetchall():
    print(f"  Index: {idx_name}")
    print(f"  Type: HNSW")

# Search with index
print("\n5. Performing search (uses index automatically)...")
query = "artificial intelligence"
query_embedding = model.encode(query)
query_embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

cur.execute("""
    SELECT 
        content,
        embedding <=> %s::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT 3;
""", (query_embedding_str,))

results = cur.fetchall()
print("\n  Top 3 results:")
for content, dist in results:
    print(f"    {content} (distance: {dist:.4f})")

# Cleanup
cur.execute("DROP TABLE IF EXISTS documents CASCADE")
conn.commit()

cur.close()
conn.close()
print("\n✓ Example complete!")








