"""
Owner: P5

Heuristic (non-trained) lesion overlay for demo/explainability
purposes.

This is NOT a validated lesion detector — classical blob/contour
heuristics only, for visual evidence in the GUI.

See CONTRACTS.md for required signature.
"""

import numpy as np


def detect_lesions(image: np.ndarray) -> dict:
    """
    TODO(P5): Classical heuristics, e.g.:

    - dark blob detection -> hemorrhage/MA candidates
    - bright blob detection -> hard exudate candidates
    - draw markers on a copy of the image

    Day 1 stub: returns dummy lesion counts and the input image.
    """

    return {
        "microaneurysm_count": 3,
        "hemorrhage_count": 1,
        "exudate_count": 2,
        "overlay_image": image,
    }