"""
Owner: P5

Grad-CAM explainability overlay.

See CONTRACTS.md for required signature.
"""

import numpy as np


def generate_gradcam(
    model, image: np.ndarray, class_idx: int
) -> np.ndarray:
    """
    TODO(P5): Standard Grad-CAM against the last conv layer
    of the backbone.

    Day 1 stub: returns the input image unchanged so the
    pipeline can run end-to-end.
    """

    return image