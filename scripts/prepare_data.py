"""
Owner: P2 (but anyone can run this once)
One-time script: subsamples the full APTOS train.csv down to a small,
stratified subset (~800-1200 images) suitable for a 6-day hackathon's
compute budget, oversampling the rare severe/proliferative classes
slightly so the model sees enough examples of them.

Usage:
    python scripts/prepare_data.py \
        --full_csv data/train.csv \
        --image_dir data/train_images \
        --output_csv data/train_subsample.csv \
        --target_total 1000

This does NOT copy image files anywhere new — it just writes a filtered
CSV. load_dataset() in src/data_pipeline.py already knows how to join
image_id -> filepath, so point it at this smaller CSV instead of the
full one and everything downstream (splitting, augmentation, training)
just works on the smaller set.
"""
import argparse
import os
import pandas as pd
import numpy as np


def subsample_stratified(df: pd.DataFrame, target_total: int, minority_boost: float = 1.5, seed: int = 42) -> pd.DataFrame:
    """
    Samples a stratified subset of `df` (expects a 'diagnosis' or 'label'
    column with values 0-4). Rare classes (3, 4) get a `minority_boost`
    multiplier applied to their natural proportional share, so the tiny
    hackathon subset doesn't end up with near-zero severe/proliferative
    examples.
    """
    label_col = "diagnosis" if "diagnosis" in df.columns else "label"
    rng = np.random.RandomState(seed)

    counts = df[label_col].value_counts().sort_index()
    proportions = counts / counts.sum()

    # Boost rare classes (3, 4), then renormalize so total still ~= target_total
    boosted = proportions.copy()
    for rare_class in (3, 4):
        if rare_class in boosted.index:
            boosted[rare_class] *= minority_boost
    boosted = boosted / boosted.sum()

    target_per_class = (boosted * target_total).round().astype(int)

    sampled_parts = []
    for cls, n_target in target_per_class.items():
        class_df = df[df[label_col] == cls]
        n_available = len(class_df)
        n_take = min(n_target, n_available)
        if n_take < n_target:
            print(f"  [warning] class {cls}: wanted {n_target}, only {n_available} available — taking all of them")
        sampled_parts.append(class_df.sample(n=n_take, random_state=seed))

    result = pd.concat(sampled_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_csv", type=str, default="data/train.csv")
    parser.add_argument("--image_dir", type=str, default="data/train_images")
    parser.add_argument("--output_csv", type=str, default="data/train_subsample.csv")
    parser.add_argument("--target_total", type=int, default=1000)
    parser.add_argument("--minority_boost", type=float, default=1.5)
    args = parser.parse_args()

    if not os.path.exists(args.full_csv):
        raise FileNotFoundError(
            f"Couldn't find {args.full_csv}. Download APTOS first:\n"
            f"  kaggle competitions download -c aptos2019-blindness-detection -p data/\n"
            f"  cd data && unzip aptos2019-blindness-detection.zip"
        )

    df = pd.read_csv(args.full_csv)
    print(f"Full dataset: {len(df)} images")
    print("Class distribution (full):")
    label_col = "diagnosis" if "diagnosis" in df.columns else "label"
    print(df[label_col].value_counts().sort_index())

    subsample = subsample_stratified(df, args.target_total, args.minority_boost)

    print(f"\nSubsampled dataset: {len(subsample)} images")
    print("Class distribution (subsample):")
    print(subsample[label_col].value_counts().sort_index())

    subsample.to_csv(args.output_csv, index=False)
    print(f"\nSaved to {args.output_csv}")

    # Sanity check: confirm every image referenced actually exists on disk
    missing = 0
    id_col = "id_code" if "id_code" in subsample.columns else "image_id"
    for image_id in subsample[id_col]:
        found = any(
            os.path.exists(os.path.join(args.image_dir, f"{image_id}{ext}"))
            for ext in (".png", ".jpg", ".jpeg")
        )
        if not found:
            missing += 1
    if missing:
        print(f"\n[WARNING] {missing} referenced images not found in {args.image_dir} — check image_dir path.")
    else:
        print(f"\nAll {len(subsample)} referenced images found in {args.image_dir}.")


if __name__ == "__main__":
    main()
