"""
Owner: P1 (Admin)
Day 4: runs the full pipeline (via main.run_pipeline) on a batch of real
images and saves reports + overlay images to disk. This is for two things:
  1. QA — sanity-check the pipeline against several real images before
     anyone trusts the demo to work live.
  2. Demo prep — pre-generated Grad-CAM/lesion overlays as a fallback in
     case live inference is slow or the venue wifi is bad.

Usage:
    python scripts/batch_sanity_check.py \
        --csv data/train_subsample.csv \
        --image_dir data/train_images \
        --n 8 \
        --checkpoint models/best_checkpoint.keras

Produces (per image), under results/batch_check/:
    {image_id}_report.txt
    {image_id}_gradcam.png
    {image_id}_lesions.png
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline import load_dataset, load_image
from src.model import load_trained_model
from main import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/train_subsample.csv")
    parser.add_argument("--image_dir", type=str, default="data/train_images")
    parser.add_argument("--n", type=int, default=8, help="Number of images to sample")
    parser.add_argument("--checkpoint", type=str, default="models/best_checkpoint.keras")
    parser.add_argument("--outdir", type=str, default="results/batch_check")
    args = parser.parse_args()

    model = None
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        model = load_trained_model(args.checkpoint)
    else:
        print(f"No checkpoint at {args.checkpoint} — running with placeholder predictions.")

    df = load_dataset(args.csv, args.image_dir)
    # Sample across a spread of classes, not just the first N rows, so the
    # sanity check actually exercises different grades. Built manually
    # (rather than via groupby().apply()) since some pandas versions drop
    # the grouping column from the result — confirmed during testing.
    per_class_n = max(1, args.n // df["label"].nunique())
    parts = []
    for label_value, group in df.groupby("label"):
        parts.append(group.sample(min(len(group), per_class_n), random_state=42))
    sample = pd.concat(parts).head(args.n)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Running pipeline on {len(sample)} images...")

    errors = []
    for _, row in sample.iterrows():
        image_id = row["image_id"]
        try:
            image = load_image(row["filepath"])
        except FileNotFoundError as e:
            errors.append((image_id, str(e)))
            continue

        result = run_pipeline(image, model=model)

        with open(os.path.join(args.outdir, f"{image_id}_report.txt"), "w") as f:
            f.write(result["report"])

        Image.fromarray(result["gradcam"]).save(os.path.join(args.outdir, f"{image_id}_gradcam.png"))
        Image.fromarray(result["lesions"]["overlay_image"]).save(
            os.path.join(args.outdir, f"{image_id}_lesions.png")
        )

        true_label = row["label"]
        pred_grade = result["prediction"]["dr_grade"]
        match = "OK" if pred_grade == true_label else "MISMATCH"
        print(f"  {image_id}: true={true_label} pred={pred_grade} [{match}] "
              f"quality={result['quality']['status']} rec={result['recommendation']}")

    if errors:
        print(f"\n[WARNING] {len(errors)} image(s) failed to load:")
        for image_id, err in errors:
            print(f"  {image_id}: {err}")

    print(f"\nSaved {len(sample) - len(errors)} reports + overlays to {args.outdir}/")


if __name__ == "__main__":
    main()
