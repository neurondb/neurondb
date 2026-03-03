#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/tmp/ann-benchmarks')
import ann_benchmarks.datasets as datasets
print("Loading dataset...")
hdf5_file, dimension = datasets.get_dataset('sift-128-euclidean')
print(f"Dataset loaded: dim={dimension}")
X_train = hdf5_file['train'][:1000]  # Just 1000 vectors
print(f"Loaded {len(X_train)} training vectors")
hdf5_file.close()
print("Test complete")
