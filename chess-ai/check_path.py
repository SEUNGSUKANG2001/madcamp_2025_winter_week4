
import os
import sys

print(f"CWD: {os.getcwd()}")

print("\n--- Data Directory ---")
if os.path.exists("data"):
    print(f"data contents: {os.listdir('data')}")
else:
    print("data directory not found")

print("\n--- Processed Position Directory ---")
pp_path = "data/processed_position"
if os.path.exists(pp_path):
    print(f"contents: {os.listdir(pp_path)}")
else:
    print(f"{pp_path} not found")

target_name = "lichess-2500-180_dataset_part_0_of_20"
full_rel_path = os.path.join(pp_path, target_name)

print(f"\n--- Checking Target: {full_rel_path} ---")
if os.path.exists(full_rel_path):
    print("EXISTS")
    print(f"Is Dir: {os.path.isdir(full_rel_path)}")
    print(f"Is File: {os.path.isfile(full_rel_path)}")
    try:
        print(f"Contents: {os.listdir(full_rel_path)}")
    except Exception as e:
        print(f"Cannot listdir: {e}")
else:
    print("DOES NOT EXIST")
    # Fuzzy match check
    if os.path.exists(pp_path):
        print(f"Checking for similar names in {pp_path}...")
        for name in os.listdir(pp_path):
            print(f" - Found: '{name}' (len={len(name)})")
            if name.strip() == target_name.strip():
                print("   (Matches stripped target name!)")
            else:
                print(f"   (Differs)")

