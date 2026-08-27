"""
Owner: P4
Training, evaluation, and single-image inference.
Day 2: real logic wired in, but every function keeps a "no trained model
yet" fallback so `python main.py --dummy` keeps working for the whole team
until Day 3's actual training run produces a checkpoint. Once a real model
is loaded, these functions automatically use it instead of the fallback.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

REFERABLE_THRESHOLD_GRADE = 2  # project's operational definition: grade >= 2 is referable


def train(model, train_data, val_data, epochs: int = 15) -> dict:
    """
    train_data / val_data: expected as (X, y) numpy tuples for this
    prototype's scale (small enough to fit in memory — no need for a
    tf.data generator at this dataset size).
    """
    checkpoint_path = "models/best_checkpoint.keras"
    callbacks = [
        __import__("tensorflow").keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True
        ),
        __import__("tensorflow").keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
    ]

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Class weighting for APTOS imbalance (grades 3/4 are rare)
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    return {"history": history.history, "best_checkpoint_path": checkpoint_path}


def evaluate(model, test_data) -> dict:
    """
    test_data: (X_test, y_test). Computed ONLY on the held-out test set —
    never used for threshold tuning (that happens on the val set only).
    """
    X_test, y_test = test_data
    probs = model.predict(X_test)
    preds = np.argmax(probs, axis=1)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "qwk": float(cohen_kappa_score(y_test, preds, weights="quadratic")),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1, 2, 3, 4]),
        "per_class": {
            "f1_per_class": f1_score(y_test, preds, average=None, labels=[0, 1, 2, 3, 4]).tolist()
        },
    }


def predict(model, image: np.ndarray) -> dict:
    """
    Single-image inference. If `model` is None (no trained checkpoint yet —
    true for the whole team until Day 3), returns a clearly-labeled
    placeholder so the rest of the pipeline stays testable.
    """
    if model is None:
        probs = [0.6, 0.2, 0.1, 0.05, 0.05]
        grade = int(np.argmax(probs))
        return {
            "dr_grade": grade,
            "referable": grade >= REFERABLE_THRESHOLD_GRADE,
            "probabilities": probs,
            "_note": "placeholder — no trained model loaded yet",
        }

    batch = np.expand_dims(image, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    grade = int(np.argmax(probs))
    return {
        "dr_grade": grade,
        "referable": grade >= REFERABLE_THRESHOLD_GRADE,
        "probabilities": probs.tolist(),
    }
