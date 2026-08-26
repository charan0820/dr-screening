"""
Owner: P4
Training, evaluation, and single-image inference.
See CONTRACTS.md for required signatures.
"""
import numpy as np


def train(model, train_data, val_data, epochs: int = 15) -> dict:
    """
    TODO(P4): model.fit with class weighting (APTOS is imbalanced —
    grades 3/4 are rare). Use ModelCheckpoint to save best-val-kappa model.
    """
    # --- STUB ---
    return {"history": {}, "best_checkpoint_path": "models/dummy_checkpoint.h5"}


def evaluate(model, test_data) -> dict:
    """
    TODO(P4): Compute accuracy, macro F1, quadratic weighted kappa,
    confusion matrix, per-class sensitivity/specificity on the held-out
    test set ONLY (never used for threshold tuning).
    """
    # --- STUB ---
    return {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "qwk": 0.0,
        "confusion_matrix": np.zeros((5, 5)),
        "per_class": {},
    }


def predict(model, image: np.ndarray) -> dict:
    """
    TODO(P4): model.predict on a single preprocessed image.
    Referable DR definition (project's operational threshold): grade >= 2.
    """
    # --- STUB ---
    probs = [0.6, 0.2, 0.1, 0.05, 0.05]
    grade = int(np.argmax(probs))
    return {
        "dr_grade": grade,
        "referable": grade >= 2,
        "probabilities": probs,
    }
