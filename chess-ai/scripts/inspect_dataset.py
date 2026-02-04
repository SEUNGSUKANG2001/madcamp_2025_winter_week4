"""
Inspect dataset samples to verify value encoding.
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from datasets import load_from_disk

# Load dataset from processed directory
dataset_path = "/mnt/c/Users/seungsu/madcamp_2025_winter_week4/data/processed/lichess-2500-180_dataset"

print(f"Loading dataset from: {dataset_path}")
dataset = load_from_disk(dataset_path)

print(f"\nDataset info:")
print(f"  Total samples: {len(dataset)}")
print(f"  Columns: {dataset.column_names}")
print(f"  Features: {dataset.features}")

# Print first 10 samples
print(f"\n{'='*80}")
print("First 10 samples:")
print(f"{'='*80}\n")

for i in range(min(10, len(dataset))):
    sample = dataset[i]
    print(f"Sample {i}:")
    print(sample)



