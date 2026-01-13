#!/usr/bin/env python3
"""
Indexing: IVF Index
===================
Learn how to create and use IVF (Inverted File) indexes.

Run: python 02_ivf_index.py
"""

import psycopg2
import os

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5433')),  # Docker Compose default port
    'database': os.getenv('DB_NAME', 'neurondb'),
    'user': os.getenv('DB_USER', 'neurondb'),  # Docker Compose default user
    'password': os.getenv('DB_PASSWORD', 'neurondb')  # Docker Compose default password
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("=" * 60)
print("Indexing: IVF Index")
print("=" * 60)

print("\nIVF (Inverted File) indexes are useful for:")
print("  - Large datasets (millions of vectors)")
print("  - When you need to balance memory and speed")
print("  - Clustering-based approximate search")

print("\n1. Creating table...")
cur.execute("""
    CREATE TABLE IF NOT EXISTS large_vectors (
        id SERIAL PRIMARY KEY,
        embedding vector(128)
    );
""")
conn.commit()
print("✓ Table created")

# Note: IVF index creation syntax may vary
print("\n2. IVF Index Creation:")
print("  CREATE INDEX idx_name")
print("  ON table_name")
print("  USING ivf (embedding vector_cosine_ops)")
print("  WITH (lists = 100);  -- number of clusters")

print("\nFor complete IVF examples, see:")
print("  - NeuronDB/docs/vector-search/indexing.md")
print("  - NeuronDB/demo/vector/sql/")

print("\nNote: IVF indexes are best for very large datasets.")
print("For most use cases, HNSW is recommended.")

cur.execute("DROP TABLE IF EXISTS large_vectors")
conn.commit()

cur.close()
conn.close()
print("\n✓ Example complete!")








