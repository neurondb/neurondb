#!/usr/bin/env python3
"""
Vector Types: Basic vector
==========================
Learn how to use the standard vector type for storing embeddings.

Run: python 01_basic_vector.py
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
print("Vector Type: Basic vector")
print("=" * 60)

# Create table with vector column
print("\n1. Creating table with vector column...")
cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id SERIAL PRIMARY KEY,
        name TEXT,
        embedding vector(384)  -- 384-dimensional vectors
    );
""")
conn.commit()
print("✓ Table created")

# Insert vectors
print("\n2. Inserting vectors...")
items = [
    ('apple', '[0.1, 0.2, 0.3]' + ',0.0' * 381),  # Simplified for demo
    ('banana', '[0.2, 0.1, 0.3]' + ',0.0' * 381),
    ('car', '[0.9, 0.1, 0.0]' + ',0.0' * 381),
]

for name, vec in items:
    cur.execute(
        "INSERT INTO items (name, embedding) VALUES (%s, %s::vector)",
        (name, vec)
    )
print(f"✓ Inserted {len(items)} items")

# Query vectors
print("\n3. Querying vectors...")
cur.execute("SELECT name, embedding FROM items LIMIT 3")
for row in cur.fetchall():
    print(f"  {row[0]}: {str(row[1])[:50]}...")

# Cleanup
cur.execute("DROP TABLE IF EXISTS items")
conn.commit()

cur.close()
conn.close()
print("\n✓ Example complete!")








