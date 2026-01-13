#!/usr/bin/env python3
"""
ML Algorithms: Classification
===============================
Learn how to use classification algorithms (Logistic Regression, SVM, etc.)

Run: python 02_classification.py

Note: See SQL demos for complete classification examples.
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
print("ML Algorithms: Classification")
print("=" * 60)

print("\nNeuronDB supports multiple classification algorithms:")
print("  - Logistic Regression")
print("  - SVM (Support Vector Machine)")
print("  - Naive Bayes")
print("  - Decision Trees")
print("  - Random Forest")
print("  - Neural Networks")

print("\nFor complete classification examples, see:")
print("  - NeuronDB/demo/ML/sql/009_logistic_regression.sql")
print("  - NeuronDB/demo/ML/sql/011_decision_tree.sql")
print("  - NeuronDB/demo/ML/sql/013_svm.sql")
print("  - NeuronDB/demo/ML/sql/016_random_forest.sql")

# Show available functions
print("\n1. Checking available classification functions...")
try:
    cur.execute("""
        SELECT routine_name 
        FROM information_schema.routines 
        WHERE routine_schema = 'public' 
        AND routine_name LIKE '%classif%' OR routine_name LIKE '%logistic%' OR routine_name LIKE '%svm%'
        ORDER BY routine_name;
    """)
    functions = cur.fetchall()
    if functions:
        print("  Available functions:")
        for func, in functions:
            print(f"    - {func}")
    else:
        print("  Functions may be in neurondb schema")
except Exception as e:
    print(f"  Note: {e}")

cur.close()
conn.close()
print("\n✓ Example complete!")








