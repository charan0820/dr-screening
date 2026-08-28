"""
Owner: P4
Day 3: the actual script you run to train the model and produce a
checkpoint. Everything it calls (build_model, train, evaluate) already
exists in src/model.py and src/train.py from Day 2 — this script just
wires them together with the real data pipeline and runs them.

Usage:
    python scripts/train_model.py \
        --csv data/train_subsample.csv \
        --image_dir data/train_images \
        --epochs 15

Produces:
    models/best_checkpoint.keras   (main.py will load this automatically)
    results/eval_metrics.json      (accuracy, macro F1, QWK, confusion matrix)
"""
import argparse
import json
import os
import sys
import numpy as np

# Allow running as `python scripts/train_model.py` from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline import load_dataset, stratified_split, load_image
from src.model import build_model
from src.train import train, evaluate


def build_arrays(df, target_size=224):
    """Loads every image referenced in df into a single (X, y) numpy pair.
    Fine at this dataset's small scale (~1000 images) — no need for a
    tf.data generator or batching complexity for a hackathon prototype."""
    images, labels = [], []
    for _, row in df.iterrows():
        img = load_image(row["filepath"], target_size=target_size)
        images.append(img)
        labels.append(row["label"])
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/train_subsample.csv")
    parser.add_argument("--image_dir", type=str, default="data/train_images")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--target_size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(args.csv, args.image_dir)
    splits = stratified_split(df)
    print({k: len(v) for k, v in splits.items()})

    print("Loading images into memory (fine at ~1000-image scale)...")
    X_train, y_train = build_arrays(splits["train"], args.target_size)
    X_val, y_val = build_arrays(splits["val"], args.target_size)
    X_test, y_test = build_arrays(splits["test"], args.target_size)

    print("Building model...")
    model = build_model(num_classes=5, input_shape=(args.target_size, args.target_size, 3))

    print("Training...")
    train_result = train(model, (X_train, y_train), (X_val, y_val), epochs=args.epochs)
    print(f"Best checkpoint saved to: {train_result['best_checkpoint_path']}")

    print("Evaluating on held-out test set...")
    metrics = evaluate(model, (X_test, y_test))
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Quadratic Weighted Kappa: {metrics['qwk']:.3f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    # Save metrics (confusion matrix -> list for JSON serializability)
    metrics_to_save = dict(metrics)
    metrics_to_save["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    with open("results/eval_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print("Saved metrics to results/eval_metrics.json")


if __name__ == "__main__":
    main()
