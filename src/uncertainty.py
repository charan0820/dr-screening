"""
Owner: P5

Monte Carlo dropout uncertainty estimation.

See CONTRACTS.md for required signature.
"""

import numpy as np


def mc_dropout_predict(
    model, image: np.ndarray, n_passes: int = 10
) -> dict:
    """
    TODO(P5): Run n_passes forward passes with dropout active
    at inference.

    Day 1 stub: returns dummy probabilities and uncertainty
    so the pipeline can run end-to-end.
    """

    mean_probs = [0.6, 0.2, 0.1, 0.05, 0.05]

    entropy = 0.9

    if entropy < 0.5:
        level = "low"
    elif entropy < 1.2:
        level = "medium"
    else:
        level = "high"

    return {
        "mean_probs": mean_probs,
        "entropy": entropy,
        "uncertainty_level": level,
    }