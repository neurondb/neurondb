#!/usr/bin/env python3
"""
Example 3: Similarity Search
============================
Learn how to:
- Perform similarity search using different distance metrics
- Find nearest neighbors
- Filter search results

Run: python 03_similarity_search.py

Requirements:
    pip install psycopg2-binary sentence-transformers
"""

import psycopg2
import os
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
print("Example 3: Similarity Search")
print("=" * 60)

# Load model and connect
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Create table with sample data
print("\n1. Creating table and inserting sample data...")
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS products (
        id SERIAL PRIMARY KEY,
        name TEXT,
        description TEXT,
        category TEXT,
        embedding vector({embedding_dim})
    );
""")

# Sample products
products = [
    ("Laptop", "High-performance laptop for work and gaming", "Electronics"),
    ("Smartphone", "Latest smartphone with advanced camera", "Electronics"),
    ("Coffee Maker", "Automatic coffee maker for home use", "Appliances"),
    ("Running Shoes", "Comfortable running shoes for athletes", "Sports"),
    ("Book", "Programming guide for Python developers", "Books"),
    ("Tablet", "Portable tablet for reading and browsing", "Electronics"),
]

for name, desc, category in products:
    # Combine name and description for embedding
    text = f"{name}: {desc}"
    embedding = model.encode(text)
    embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
    
    cur.execute(
        "INSERT INTO products (name, description, category, embedding) VALUES (%s, %s, %s, %s::vector)",
        (name, desc, category, embedding_str)
    )

conn.commit()
print(f"✓ Inserted {len(products)} products")

# Example 1: Basic similarity search
print("\n2. Example 1: Basic similarity search")
query = "I need a device to browse the internet"
print(f"  Query: {query}")

query_embedding = model.encode(query)
query_embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

cur.execute("""
    SELECT 
        name,
        description,
        embedding <=> %s::vector AS distance
    FROM products
    ORDER BY distance
    LIMIT 3
""", (query_embedding_str,))

results = cur.fetchall()
print("\n  Top 3 results:")
for i, (name, desc, dist) in enumerate(results, 1):
    print(f"    {i}. {name} - {desc}")
    print(f"       Distance: {dist:.4f}")

# Example 2: Filtered search (by category)
print("\n3. Example 2: Filtered search (Electronics only)")
query = "portable computing device"
print(f"  Query: {query}")

query_embedding = model.encode(query)
query_embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

cur.execute("""
    SELECT 
        name,
        category,
        embedding <=> %s::vector AS distance
    FROM products
    WHERE category = 'Electronics'
    ORDER BY distance
    LIMIT 3
""", (query_embedding_str,))

results = cur.fetchall()
print("\n  Top Electronics:")
for i, (name, cat, dist) in enumerate(results, 1):
    print(f"    {i}. {name} ({cat}) - Distance: {dist:.4f}")

# Example 3: Using different distance metrics
print("\n4. Example 3: Different distance metrics")
query = "coffee brewing machine"
print(f"  Query: {query}")

query_embedding = model.encode(query)
query_embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

# Cosine distance (<=>)
cur.execute("""
    SELECT name, embedding <=> %s::vector AS cosine_distance
    FROM products
    ORDER BY cosine_distance
    LIMIT 1
""", (query_embedding_str,))
result = cur.fetchone()
print(f"  Cosine distance: {result[0]} (distance: {result[1]:.4f})")

# L2 distance (<->)
cur.execute("""
    SELECT name, embedding <-> %s::vector AS l2_distance
    FROM products
    ORDER BY l2_distance
    LIMIT 1
""", (query_embedding_str,))
result = cur.fetchone()
print(f"  L2 distance: {result[0]} (distance: {result[1]:.4f})")

# Cleanup
print("\n5. Cleaning up...")
cur.execute("DROP TABLE IF EXISTS products")
conn.commit()
print("✓ Table dropped")

cur.close()
conn.close()

print("\n" + "=" * 60)
print("Example complete! ✓")
print("=" * 60)








