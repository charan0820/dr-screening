"""
Owner: P5
Day 5: the mini ablation study — directly tests RQ2/RQ3 at hackathon scale:
does quality-aware preprocessing (enhancement) actually help, compared to
feeding the CNN raw images unchanged?

Two conditions on the SAME held-out test set, SAME trained model:
  A. "baseline"      — raw images, no quality-based enhancement
  B. "quality-aware"  — images enhanced via src.enhancement before prediction

This isolates the effect of the enhancement step specifically (not the full
lesion/uncertainty stack, which isn't a separate trainable component in this
prototype) — an honest, small-scale version of the ablation the full spec
calls for, appropriately scoped to 6 days.

Usage:
    python scripts/ablation_study.py \
        --csv data/train_subsample.csv \
        --image_dir data/train_images \
        --checkpoint models/best_checkpoint.keras
"""
import argparse
import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline import load_dataset, stratified_split, load_image
from src.quality import compute_quality_score
from src.enhancement import enhance_image
from src.model import load_trained_model


def evaluate_condition(model, df, apply_enhancement: bool) -> dict:
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        try:
            image = load_image(row["filepath"]).astype(np.float32)
        except FileNotFoundError:
            continue

        if apply_enhancement:
            q = compute_quality_score(image.astype(np.uint8))
            image = enhance_image(image.astype(np.uint8), q.get("failure_reason")).astype(np.float32)

        batch = np.expand_dims(image, axis=0)
        probs = model.predict(batch, verbose=0)[0]
        y_pred.append(int(np.argmax(probs)))
        y_true.append(row["label"])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "n": len(y_true),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/train_subsample.csv")
    parser.add_argument("--image_dir", type=str, default="data/train_images")
    parser.add_argument("--checkpoint", type=str, default="models/best_checkpoint.keras")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"No checkpoint at {args.checkpoint}. The ablation study needs a "
            "real trained model — run scripts/train_model.py first."
        )

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_trained_model(args.checkpoint)

    df = load_dataset(args.csv, args.image_dir)
    splits = stratified_split(df)
    test_df = splits["test"]
    print(f"Evaluating on {len(test_df)} held-out test images...")

    print("\nCondition A: baseline (raw images, no enhancement)")
    baseline = evaluate_condition(model, test_df, apply_enhancement=False)
    print(f"  accuracy={baseline['accuracy']:.3f}  macro_f1={baseline['macro_f1']:.3f}  qwk={baseline['qwk']:.3f}")

    print("\nCondition B: quality-aware (adaptive enhancement applied)")
    quality_aware = evaluate_condition(model, test_df, apply_enhancement=True)
    print(f"  accuracy={quality_aware['accuracy']:.3f}  macro_f1={quality_aware['macro_f1']:.3f}  qwk={quality_aware['qwk']:.3f}")

    result = {"baseline": baseline, "quality_aware": quality_aware}
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {out_path}")

    # Report honestly — do NOT claim improvement unless the numbers show it.
    qwk_delta = quality_aware["qwk"] - baseline["qwk"]
    if qwk_delta > 0.02:
        print(f"\n[RESULT] Quality-aware pipeline improved QWK by {qwk_delta:+.3f} over baseline.")
    elif qwk_delta < -0.02:
        print(f"\n[RESULT] Quality-aware pipeline performed WORSE by {qwk_delta:+.3f} QWK than baseline. "
              "Report this honestly — do not hide a negative result.")
    else:
        print(f"\n[RESULT] No meaningful difference detected (QWK delta={qwk_delta:+.3f}) at this sample size. "
              "This is a valid and reportable finding, not a failure — say so explicitly rather than "
              "overstating a small/noisy difference either direction.")


if __name__ == "__main__":
    main()
