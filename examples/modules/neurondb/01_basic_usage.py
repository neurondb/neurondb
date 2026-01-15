#!/usr/bin/env python3
"""
NeuronDB Module: Basic Usage
=============================
Learn how to use NeuronDB extension in PostgreSQL.

Run: python 01_basic_usage.py
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
print("NeuronDB Module: Basic Usage")
print("=" * 60)

# Check if extension is installed
print("\n1. Checking NeuronDB extension...")
try:
    cur.execute("""
        SELECT extname, extversion 
        FROM pg_extension 
        WHERE extname = 'neurondb';
    """)
    result = cur.fetchone()
    if result:
        print(f"✓ NeuronDB extension installed (version: {result[1]})")
    else:
        print("  Extension not found. Install with: CREATE EXTENSION neurondb;")
except Exception as e:
    print(f"  Error: {e}")

# Show available vector functions
print("\n2. Available vector functions:")
try:
    cur.execute("""
        SELECT routine_name 
        FROM information_schema.routines 
        WHERE routine_schema IN ('public', 'neurondb')
        AND routine_name LIKE '%vector%'
        ORDER BY routine_name
        LIMIT 10;
    """)
    functions = cur.fetchall()
    if functions:
        for func, in functions:
            print(f"    - {func}")
    else:
        print("    (Check neurondb schema)")
except Exception as e:
    print(f"    Note: {e}")

# Basic vector operations
print("\n3. Basic vector operations:")
cur.execute("""
    SELECT 
        '[1,2,3]'::vector AS vec,
        vector_dims('[1,2,3]'::vector) AS dimensions,
        vector_norm('[3,4]'::vector) AS norm;
""")
result = cur.fetchone()
if result:
    print(f"    Vector: {result[0]}")
    print(f"    Dimensions: {result[1]}")
    print(f"    Norm of [3,4]: {result[2]:.2f}")

cur.close()
conn.close()
print("\n✓ Example complete!")









