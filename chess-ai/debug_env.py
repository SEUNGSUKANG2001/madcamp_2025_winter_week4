
import os
import sys

# Path to check
path = "data/processed_position/lichess-2500-180_dataset_part_0_of_20"
abs_path = os.path.abspath(path)

print(f"Checking path: {path}")
print(f"Absolute path: {abs_path}")
print(f"Exists: {os.path.exists(path)}")
print(f"Is Directory: {os.path.isdir(path)}")

try:
    import datasets
    print("datasets library: INSTALLED")
except ImportError as e:
    print(f"datasets library: NOT INSTALLED ({e})")

try:
    import h5py
    print("h5py library: INSTALLED")
except ImportError as e:
    print(f"h5py library: NOT INSTALLED ({e})")
