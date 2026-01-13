#!/usr/bin/env python3
"""
Example 1: Basic Vector Operations
===================================
Learn how to:
- Create a table with vector columns
- Insert vectors into the table
- Query vectors

Run: python 01_basic_vectors.py
"""

import psycopg2
import os

# Database connection (adjust these to match your setup)
# Default values match Docker Compose setup (port 5433, user neurondb)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5433')),  # Docker Compose default port
    'database': os.getenv('DB_NAME', 'neurondb'),
    'user': os.getenv('DB_USER', 'neurondb'),  # Docker Compose default user
    'password': os.getenv('DB_PASSWORD', 'neurondb')  # Docker Compose default password
}

# Connect to database
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("=" * 60)
print("Example 1: Basic Vector Operations")
print("=" * 60)

# Step 1: Create a table with a vector column
print("\n1. Creating table with vector column...")
cur.execute("""
    CREATE TABLE IF NOT EXISTS simple_vectors (
        id SERIAL PRIMARY KEY,
        name TEXT,
        embedding vector(3)  -- 3-dimensional vectors
    );
""")
conn.commit()
print("✓ Table created")

# Step 2: Insert some vectors
print("\n2. Inserting vectors...")
vectors = [
    ('apple', '[1.0, 0.0, 0.0]'),
    ('banana', '[0.0, 1.0, 0.0]'),
    ('car', '[0.0, 0.0, 1.0]'),
    ('vehicle', '[0.0, 0.0, 0.9]'),
]

for name, vec in vectors:
    cur.execute(
        "INSERT INTO simple_vectors (name, embedding) VALUES (%s, %s::vector)",
        (name, vec)
    )
conn.commit()
print(f"✓ Inserted {len(vectors)} vectors")

# Step 3: Query all vectors
print("\n3. Querying all vectors...")
cur.execute("SELECT id, name, embedding FROM simple_vectors")
rows = cur.fetchall()
for row in rows:
    print(f"  {row[1]}: {row[2]}")

# Step 4: Calculate vector distance
print("\n4. Calculating distance between 'apple' and 'banana'...")
cur.execute("""
    SELECT 
        name,
        embedding <-> (SELECT embedding FROM simple_vectors WHERE name = 'apple') AS distance
    FROM simple_vectors
    WHERE name = 'banana'
""")
result = cur.fetchone()
print(f"  Distance: {result[1]:.4f}")

# Step 5: Find nearest neighbor
print("\n5. Finding nearest neighbor to 'car'...")
cur.execute("""
    SELECT 
        name,
        embedding <-> (SELECT embedding FROM simple_vectors WHERE name = 'car') AS distance
    FROM simple_vectors
    WHERE name != 'car'
    ORDER BY distance
    LIMIT 1
""")
result = cur.fetchone()
print(f"  Nearest to 'car': {result[0]} (distance: {result[1]:.4f})")

# Cleanup
print("\n6. Cleaning up...")
cur.execute("DROP TABLE IF EXISTS simple_vectors")
conn.commit()
print("✓ Table dropped")

cur.close()
conn.close()

print("\n" + "=" * 60)
print("Example complete! ✓")
print("=" * 60)








