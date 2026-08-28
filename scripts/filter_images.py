import os
import shutil
import pandas as pd

# Paths
CSV_PATH = "data/train_subsample.csv"
SOURCE_DIR = "data/train_images"
OUTPUT_DIR = "data/filtered_images"

# Read IDs from CSV
df = pd.read_csv(CSV_PATH)
ids = set(df["id_code"].astype(str))

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

copied = 0
missing = 0

# Check every file in the image directory
for filename in os.listdir(SOURCE_DIR):

    # Get ID without extension
    image_id = os.path.splitext(filename)[0]

    if image_id in ids:
        source_path = os.path.join(SOURCE_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        shutil.copy2(source_path, output_path)
        copied += 1

print(f"IDs in CSV       : {len(ids)}")
print(f"Images copied    : {copied}")
print(f"Images missing   : {len(ids) - copied}")
print(f"Output directory : {OUTPUT_DIR}")