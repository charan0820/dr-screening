"""
Owner: P5
Monte Carlo dropout uncertainty estimation.
Day 2: real MC-dropout implementation. Falls back to a placeholder if no
trained model exists yet.
"""
import numpy as np

# Tune these experimentally against your own val set — starting points only.
LOW_ENTROPY_MAX = 0.5
MEDIUM_ENTROPY_MAX = 1.2


def mc_dropout_predict(model, image: np.ndarray, n_passes: int = 10) -> dict:
    """
    Runs n_passes forward passes with dropout active at inference time
    (training=True), computes mean probabilities and predictive entropy,
    then buckets entropy into low/medium/high.
    """
    if model is None:
        mean_probs = [0.6, 0.2, 0.1, 0.05, 0.05]
        entropy = 0.9
        level = _bucket(entropy)
        return {
            "mean_probs": mean_probs,
            "entropy": entropy,
            "uncertainty_level": level,
            "_note": "placeholder — no trained model loaded yet",
        }

    batch = np.expand_dims(image, axis=0)
    all_probs = []
    for _ in range(n_passes):
        probs = model(batch, training=True).numpy()[0]  # training=True keeps dropout active
        all_probs.append(probs)

    all_probs = np.array(all_probs)
    mean_probs = all_probs.mean(axis=0)
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-8)))

    return {
        "mean_probs": mean_probs.tolist(),
        "entropy": entropy,
        "uncertainty_level": _bucket(entropy),
    }


def _bucket(entropy: float) -> str:
    if entropy < LOW_ENTROPY_MAX:
        return "low"
    elif entropy < MEDIUM_ENTROPY_MAX:
        return "medium"
    else:
        return "high"
