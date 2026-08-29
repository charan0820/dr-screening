"""
Owner: P3
Day 4: runs compute_quality_score() across your real subsample and plots
the score distribution, so you can sanity-check (and if needed, adjust)
the thresholds in src/quality.py against real fundus images rather than
guesses. Also prints a few example failure-reason breakdowns.

Usage:
    python scripts/quality_diagnostics.py \
        --csv data/train_subsample.csv \
        --image_dir data/train_images
"""
import argparse
import os
import sys
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline import load_dataset, load_image
from src.quality import compute_quality_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/train_subsample.csv")
    parser.add_argument("--image_dir", type=str, default="data/train_images")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    df = load_dataset(args.csv, args.image_dir)
    scores = []
    statuses = Counter()
    failure_reasons = Counter()

    print(f"Scoring {len(df)} images...")
    for i, row in df.iterrows():
        try:
            image = load_image(row["filepath"])
        except FileNotFoundError:
            continue
        result = compute_quality_score(image)
        scores.append(result["score"])
        statuses[result["status"]] += 1
        if result["failure_reason"]:
            failure_reasons[result["failure_reason"]] += 1

        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(df)}")

    print("\nStatus distribution:")
    for status, count in statuses.items():
        print(f"  {status}: {count} ({100 * count / len(scores):.1f}%)")

    print("\nFailure reason distribution (among flagged images):")
    for reason, count in failure_reasons.items():
        print(f"  {reason}: {count}")

    os.makedirs(args.outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=30, color="#de638a", edgecolor="white")
    ax.axvline(80, color="green", linestyle="--", label="good threshold (80)")
    ax.axvline(60, color="orange", linestyle="--", label="borderline threshold (60)")
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Number of images")
    ax.set_title("Quality Score Distribution — Real Subsample")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(args.outdir, "quality_score_distribution.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    # A quick heads-up if the current thresholds look badly miscalibrated
    ungradable_fraction = statuses.get("ungradable", 0) / max(len(scores), 1)
    if ungradable_fraction > 0.5:
        print(
            "\n[NOTE] Over half your images are flagged ungradable. Since APTOS "
            "images are generally clinical-grade quality, this likely means the "
            "thresholds in src/quality.py are too strict for this dataset — "
            "consider loosening them rather than assuming the images are bad."
        )


if __name__ == "__main__":
    main()
