#!/usr/bin/env python3
"""
ML Algorithms: K-Means Clustering
==================================
Learn how to use K-Means clustering for grouping similar data.

Run: python 01_clustering_kmeans.py

Note: This requires a dataset. See the SQL demo for full examples.
"""

import psycopg2
import os
import numpy as np

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
print("ML Algorithms: K-Means Clustering")
print("=" * 60)

# Create sample dataset
print("\n1. Creating sample dataset...")
cur.execute("""
    CREATE TEMP TABLE sample_data (
        id SERIAL PRIMARY KEY,
        features vector(3),
        label TEXT
    );
    
    -- Create 3 clusters of points
    INSERT INTO sample_data (features, label) VALUES
        ('[1.0, 1.0, 1.0]'::vector, 'cluster1'),
        ('[1.1, 1.0, 1.0]'::vector, 'cluster1'),
        ('[0.9, 1.1, 1.0]'::vector, 'cluster1'),
        ('[5.0, 5.0, 5.0]'::vector, 'cluster2'),
        ('[5.1, 5.0, 5.0]'::vector, 'cluster2'),
        ('[4.9, 5.1, 5.0]'::vector, 'cluster2'),
        ('[10.0, 10.0, 10.0]'::vector, 'cluster3'),
        ('[10.1, 10.0, 10.0]'::vector, 'cluster3'),
        ('[9.9, 10.1, 10.0]'::vector, 'cluster3');
""")
conn.commit()
print("✓ Dataset created (9 points in 3 clusters)")

# Run K-Means clustering
print("\n2. Running K-Means clustering (K=3)...")
try:
    cur.execute("""
        SELECT cluster_kmeans('sample_data', 'features', 3, 50) AS clusters;
    """)
    result = cur.fetchone()
    clusters = result[0] if result else []
    
    print(f"✓ Clustering complete: {len(clusters)} clusters found")
    
    # Show cluster assignments
    print("\n3. Cluster assignments:")
    cur.execute("""
        SELECT 
            id,
            label,
            unnest(%s::int[]) AS cluster
        FROM sample_data
        ORDER BY id;
    """, (clusters,))
    
    assignments = cur.fetchall()
    for row_id, label, cluster in assignments:
        print(f"    Point {row_id} ({label}): Cluster {cluster}")
        
except Exception as e:
    print(f"  Note: K-Means function may require specific setup.")
    print(f"  Error: {e}")
    print("\n  For full K-Means examples, see:")
    print("  NeuronDB/demo/ML/sql/002_kmeans_clustering.sql")

cur.close()
conn.close()
print("\n✓ Example complete!")








