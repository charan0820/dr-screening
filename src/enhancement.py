"""
Owner: P3
Adaptive enhancement — selects a method based on the diagnosed quality issue.
Day 2: real CLAHE-based implementation replaces Day 1 pass-through stub.
"""
import cv2
import numpy as np


def _apply_clahe(image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """CLAHE on the L-channel in LAB space — standard retinal-enhancement move."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _mild_sharpen(image: np.ndarray) -> np.ndarray:
    """Conservative unsharp mask. Deliberately mild — aggressive sharpening
    risks inventing lesion-like edge artifacts, which we must not do
    without evaluating that risk (see Section 6 requirements)."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(image, 1.3, blurred, -0.3, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_image(image: np.ndarray, failure_reason: str | None) -> np.ndarray:
    """
    Routes to a specific enhancement based on the diagnosed failure reason.
    Returns the image unchanged if no issue was flagged.
    """
    if failure_reason in ("low_contrast", "low_illumination"):
        return _apply_clahe(image)
    elif failure_reason == "glare":
        # Mild CLAHE with a lower clip limit tends to help without
        # over-amplifying already-blown-out regions.
        return _apply_clahe(image, clip_limit=1.5)
    elif failure_reason == "blur":
        return _mild_sharpen(image)
    else:
        return image
