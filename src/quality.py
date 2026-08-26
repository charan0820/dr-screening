"""
Owner: P3
Image quality scoring + recapture-instruction generation.
See CONTRACTS.md for required signatures.
"""
import numpy as np


def compute_quality_score(image: np.ndarray) -> dict:
    """
    TODO(P3): Implement using classical metrics:
      - Laplacian variance (focus/blur)
      - brightness / illumination stats
      - contrast (std dev of intensity)
      - retinal-area percentage (rough field-of-view check)
    Combine into a normalized 0-100 score. Thresholds are configurable
    (e.g. >=80 good, 60-79 borderline-enhance, <60 ungradable) — tune
    experimentally, don't assume these are clinically validated.
    """
    # --- STUB ---
    return {
        "score": 75.0,
        "status": "borderline",
        "failure_reason": "low_contrast",
    }


def get_recapture_message(failure_reason: str) -> str:
    """
    TODO(P3): Map each failure_reason to a clear instruction.
    """
    # --- STUB ---
    messages = {
        "blur": "Image ungradable due to excessive blur. Please stabilize the camera and recapture.",
        "low_illumination": "Image too dark. Please improve lighting and recapture.",
        "low_contrast": "Low contrast detected. Please recapture with correct focus/exposure.",
        "incomplete_field": "Retinal field incomplete. Please recenter and recapture.",
        "glare": "Excessive glare detected. Please adjust camera angle and recapture.",
    }
    return messages.get(failure_reason, "Please recapture the image.")
