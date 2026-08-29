"""
Owner: P4
Day 4: turns evaluate()'s raw metrics into plots worth showing in the demo —
confusion matrix heatmap and per-class F1 bar chart. Run this right after
scripts/train_model.py finishes.

Usage:
    python scripts/evaluate_and_visualize.py --metrics results/eval_metrics.json

Produces:
    results/confusion_matrix.png
    results/per_class_f1.png
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def plot_confusion_matrix(cm: np.ndarray, output_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title("DR Grade Confusion Matrix (held-out test set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(f1_scores: list, output_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_NAMES, f1_scores, color="#4a3267")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-Class F1 (held-out test set)")
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="results/eval_metrics.json")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        raise FileNotFoundError(
            f"Couldn't find {args.metrics}. Run scripts/train_model.py first — "
            "it writes this file automatically after evaluate()."
        )

    with open(args.metrics) as f:
        metrics = json.load(f)

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Quadratic Weighted Kappa: {metrics['qwk']:.3f}")

    cm = np.array(metrics["confusion_matrix"])
    os.makedirs(args.outdir, exist_ok=True)
    cm_path = os.path.join(args.outdir, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)
    print(f"Saved {cm_path}")

    f1_scores = metrics["per_class"]["f1_per_class"]
    f1_path = os.path.join(args.outdir, "per_class_f1.png")
    plot_per_class_f1(f1_scores, f1_path)
    print(f"Saved {f1_path}")

    # Flag clinically important error types directly in the console output —
    # a 0<->4 confusion is a much bigger deal than a 1<->2 confusion.
    if cm.shape == (5, 5):
        severe_errors = cm[0, 4] + cm[4, 0]
        if severe_errors > 0:
            print(f"\n[NOTE] {severe_errors} case(s) confused between Grade 0 and Grade 4 "
                  "— the most clinically serious error type. Worth reviewing these "
                  "specific images for your failure-analysis discussion.")


if __name__ == "__main__":
    main()
