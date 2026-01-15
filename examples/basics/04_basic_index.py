#!/usr/bin/env python3
"""
Example 4: Creating and Using Indexes
======================================
Learn how to:
- Create HNSW indexes for fast vector search
- Understand index parameters
- See performance difference with/without index

Run: python 04_basic_index.py

Requirements:
    pip install psycopg2-binary sentence-transformers
"""

import psycopg2
import os
import time
from sentence_transformers import SentenceTransformer

# Database connection (defaults match Docker Compose setup)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5433')),  # Docker Compose default port
    'database': os.getenv('DB_NAME', 'neurondb'),
    'user': os.getenv('DB_USER', 'neurondb'),  # Docker Compose default user
    'password': os.getenv('DB_PASSWORD', 'neurondb')  # Docker Compose default password
}

MODEL_NAME = "all-MiniLM-L6-v2"

print("=" * 60)
print("Example 4: Creating and Using Indexes")
print("=" * 60)

# Load model and connect
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Create table
print("\n1. Creating table...")
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector({embedding_dim})
    );
""")
conn.commit()
print("✓ Table created")

# Insert sample documents
print("\n2. Inserting sample documents...")
documents = [
    "Machine learning algorithms can learn from data",
    "Python is a versatile programming language",
    "Databases store and retrieve information efficiently",
    "Neural networks are inspired by the human brain",
    "PostgreSQL is an open-source relational database",
    "Vector embeddings represent text as numbers",
    "Search engines use ranking algorithms",
    "Natural language processing enables text understanding",
    "Deep learning uses multiple neural network layers",
    "SQL queries retrieve data from databases",
] * 10  # Create 100 documents for better performance comparison

for doc in documents:
    embedding = model.encode(doc)
    embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector)",
        (doc, embedding_str)
    )

conn.commit()
print(f"✓ Inserted {len(documents)} documents")

# Search without index
print("\n3. Searching WITHOUT index...")
query = "artificial intelligence and machine learning"
query_embedding = model.encode(query)
query_embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

start = time.time()
cur.execute("""
    SELECT content, embedding <=> %s::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT 5
""", (query_embedding_str,))
results = cur.fetchall()
time_without_index = time.time() - start

print(f"  Time: {time_without_index:.4f} seconds")
print("  Top 5 results:")
for i, (content, dist) in enumerate(results[:5], 1):
    print(f"    {i}. {content[:60]}... (distance: {dist:.4f})")

# Create HNSW index
print("\n4. Creating HNSW index...")
print("  This may take a moment...")
start = time.time()
cur.execute(f"""
    CREATE INDEX documents_embedding_idx 
    ON documents 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")
conn.commit()
index_time = time.time() - start
print(f"✓ Index created in {index_time:.2f} seconds")

# Search with index
print("\n5. Searching WITH index...")
start = time.time()
cur.execute("""
    SELECT content, embedding <=> %s::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT 5
""", (query_embedding_str,))
results = cur.fetchall()
time_with_index = time.time() - start

print(f"  Time: {time_with_index:.4f} seconds")
print("  Top 5 results:")
for i, (content, dist) in enumerate(results[:5], 1):
    print(f"    {i}. {content[:60]}... (distance: {dist:.4f})")

# Performance comparison
print("\n6. Performance comparison:")
speedup = time_without_index / time_with_index if time_with_index > 0 else 0
print(f"  Without index: {time_without_index:.4f}s")
print(f"  With index:    {time_with_index:.4f}s")
print(f"  Speedup:       {speedup:.2f}x faster")

# Index information
print("\n7. Index information:")
cur.execute("""
    SELECT 
        indexname,
        indexdef
    FROM pg_indexes
    WHERE tablename = 'documents'
""")
indexes = cur.fetchall()
for idx_name, idx_def in indexes:
    print(f"  Index: {idx_name}")
    print(f"  Definition: {idx_def[:80]}...")

# Cleanup
print("\n8. Cleaning up...")
cur.execute("DROP TABLE IF EXISTS documents CASCADE")
conn.commit()
print("✓ Table and index dropped")

cur.close()
conn.close()

print("\n" + "=" * 60)
print("Example complete! ✓")
print("=" * 60)
print("\nNote: Indexes are essential for fast search on large datasets.")
print("HNSW (Hierarchical Navigable Small World) is the recommended index type.")









