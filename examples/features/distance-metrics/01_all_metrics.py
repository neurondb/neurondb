#!/usr/bin/env python3
"""
Distance Metrics: All Metrics
==============================
Learn how to use different distance metrics for vector similarity.

Run: python 01_all_metrics.py
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
print("Distance Metrics: All Metrics")
print("=" * 60)

# Create test vectors
print("\n1. Creating test vectors...")
cur.execute("""
    CREATE TEMP TABLE test_vectors (
        name TEXT,
        vec vector(3)
    );
    
    INSERT INTO test_vectors VALUES
        ('vec1', '[1, 2, 3]'::vector),
        ('vec2', '[4, 5, 6]'::vector),
        ('vec3', '[0, 0, 0]'::vector);
""")
conn.commit()
print("✓ Test vectors created")

# Compare all distance metrics
print("\n2. Comparing distance metrics...")
cur.execute("""
    SELECT 
        'L2 (Euclidean)' AS metric,
        vector_l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS distance,
        'Geometric distance' AS description
    UNION ALL
    SELECT 
        'Cosine',
        vector_cosine_distance('[1,2,3]'::vector, '[4,5,6]'::vector),
        'Angular distance (good for text)'
    UNION ALL
    SELECT 
        'Inner Product',
        vector_inner_product('[1,2,3]'::vector, '[4,5,6]'::vector),
        'Dot product (negative)'
    UNION ALL
    SELECT 
        'L1 (Manhattan)',
        vector_l1_distance('[1,2,3]'::vector, '[4,5,6]'::vector),
        'Sum of absolute differences'
    UNION ALL
    SELECT 
        'Chebyshev',
        vector_chebyshev_distance('[1,2,3]'::vector, '[4,5,6]'::vector)::float,
        'Maximum dimension difference'
    ORDER BY distance;
""")

results = cur.fetchall()
print("\n  Distance Metrics Comparison:")
for metric, distance, desc in results:
    print(f"    {metric:20} = {distance:8.4f}  ({desc})")

# Using operators
print("\n3. Using distance operators...")
cur.execute("""
    SELECT 
        v1.name AS vec1,
        v2.name AS vec2,
        v1.vec <-> v2.vec AS l2_distance,
        v1.vec <=> v2.vec AS cosine_distance,
        v1.vec <#> v2.vec AS inner_product
    FROM test_vectors v1, test_vectors v2
    WHERE v1.name < v2.name
    ORDER BY l2_distance;
""")

results = cur.fetchall()
print("\n  Vector Comparisons:")
for vec1, vec2, l2, cos, ip in results:
    print(f"    {vec1} vs {vec2}:")
    print(f"      L2: {l2:.4f}, Cosine: {cos:.4f}, Inner Product: {ip:.4f}")

cur.close()
conn.close()
print("\n✓ Example complete!")









