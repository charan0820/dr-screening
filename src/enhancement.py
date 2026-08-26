"""
Owner: P3
Adaptive enhancement — selects a method based on the diagnosed quality issue.
See CONTRACTS.md for required signature.
"""
import numpy as np


def enhance_image(image: np.ndarray, failure_reason: str | None) -> np.ndarray:
    """
    TODO(P3): Implement adaptive enhancement:
      - failure_reason == 'low_contrast' or 'low_illumination' -> CLAHE
      - failure_reason == 'blur' -> mild sharpening (careful: don't invent
        lesion-like structures — flag this risk in the report)
      - failure_reason is None -> return image unchanged
    Use cv2.createCLAHE on the L-channel in LAB space as the main technique.
    """
    # --- STUB: returns input unchanged ---
    return image
