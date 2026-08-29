"""
Owner: P5
Heuristic (non-trained) lesion overlay for demo/explainability purposes.
Day 2: real classical blob-detection heuristics replace the Day 1 stub.
NOT a validated lesion detector — clearly labeled as a research/demo
heuristic wherever it's shown (report_generator.py already does this).
"""
import cv2
import numpy as np


def _detect_dark_blobs(gray: np.ndarray, min_area=8, max_area=250):
    """Candidate microaneurysm/hemorrhage blobs: small dark regions."""
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if min_area < cv2.contourArea(c) < max_area]


def _detect_bright_blobs(gray: np.ndarray, min_area=15, max_area=400):
    """Candidate hard-exudate blobs: small bright regions."""
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if min_area < cv2.contourArea(c) < max_area]


def detect_lesions(image: np.ndarray) -> dict:
    """
    Classical heuristic overlay — draws candidate markers, does NOT claim
    clinical-grade lesion detection. Dark blobs -> MA/hemorrhage candidates
    (red), bright blobs -> hard-exudate candidates (yellow).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    overlay = image.copy()

    dark_blobs = _detect_dark_blobs(gray)
    bright_blobs = _detect_bright_blobs(gray)

    # Rough split: very small dark blobs -> MA candidates, slightly larger -> hemorrhage
    ma_candidates = [c for c in dark_blobs if cv2.contourArea(c) < 40]
    hemorrhage_candidates = [c for c in dark_blobs if cv2.contourArea(c) >= 40]

    for c in ma_candidates:
        (x, y), r = cv2.minEnclosingCircle(c)
        cv2.circle(overlay, (int(x), int(y)), max(int(r), 2), (255, 0, 0), 1)  # red

    for c in hemorrhage_candidates:
        (x, y), r = cv2.minEnclosingCircle(c)
        cv2.circle(overlay, (int(x), int(y)), max(int(r), 2), (139, 0, 0), 2)  # dark red

    for c in bright_blobs:
        (x, y), r = cv2.minEnclosingCircle(c)
        cv2.circle(overlay, (int(x), int(y)), max(int(r), 2), (255, 255, 0), 1)  # yellow

    return {
        "microaneurysm_count": len(ma_candidates),
        "hemorrhage_count": len(hemorrhage_candidates),
        "exudate_count": len(bright_blobs),
        "overlay_image": overlay,
    }


def combine_evidence(
    original: np.ndarray,
    gradcam_image: np.ndarray,
    lesion_overlay_image: np.ndarray,
) -> np.ndarray:
    """
    Combines original, Grad-CAM, and lesion overlay images
    side by side into one evidence image.

    All inputs are expected to be RGB uint8 images.
    """

    h = original.shape[0]

    resized = [
        cv2.resize(img, (h, h))
        for img in (original, gradcam_image, lesion_overlay_image)
    ]
    combined = np.hstack(resized)

    return combined