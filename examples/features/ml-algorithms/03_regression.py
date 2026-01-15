#!/usr/bin/env python3
"""
ML Algorithms: Regression
==========================
Learn how to use regression algorithms for predicting continuous values.

Run: python 03_regression.py
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
print("ML Algorithms: Regression")
print("=" * 60)

print("\nNeuronDB supports multiple regression algorithms:")
print("  - Linear Regression")
print("  - Ridge Regression")
print("  - Lasso Regression")
print("  - Deep Learning models")

print("\nFor complete regression examples, see:")
print("  - NeuronDB/demo/ML/sql/008_linear_regression.sql")
print("  - NeuronDB/demo/ML/sql/014_ridge_lasso.sql")

# Create simple example
print("\n1. Creating sample regression dataset...")
cur.execute("""
    CREATE TEMP TABLE regression_sample (
        id SERIAL PRIMARY KEY,
        features vector(3),
        target FLOAT
    );
    
    -- Simple linear relationship: target = sum of features
    INSERT INTO regression_sample (features, target) VALUES
        ('[1, 1, 1]'::vector, 3.0),
        ('[2, 2, 2]'::vector, 6.0),
        ('[3, 3, 3]'::vector, 9.0),
        ('[4, 4, 4]'::vector, 12.0),
        ('[5, 5, 5]'::vector, 15.0);
""")
conn.commit()
print("✓ Sample dataset created")

print("\n2. Regression functions available:")
print("  - train_linear_regression(table, features_col, target_col)")
print("  - predict_linear_regression(coefficients, features)")
print("  - evaluate_linear_regression(table, features_col, target_col, coefficients)")

print("\nFor full examples with training and evaluation, see:")
print("  NeuronDB/demo/ML/sql/008_linear_regression.sql")

cur.close()
conn.close()
print("\n✓ Example complete!")









