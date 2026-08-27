"""
Owner: P3
Image quality scoring + recapture-instruction generation.
Day 2: real classical-CV implementation replaces Day 1 stub.
Thresholds below are configurable starting points — tune experimentally
against your own subsample, they are NOT clinically validated values.
"""
import cv2
import numpy as np

# --- Configurable thresholds (tune during Day 2/3 testing) ---
GOOD_THRESHOLD = 80
BORDERLINE_THRESHOLD = 60
BLUR_LAPLACIAN_MIN = 100.0      # below this -> flagged as blurry
DARK_MEAN_MAX = 60.0            # below this mean brightness -> too dark
BRIGHT_MEAN_MIN = 200.0         # above this -> overexposed/glare
LOW_CONTRAST_STD_MAX = 30.0     # below this std dev -> low contrast
MIN_RETINAL_AREA_FRACTION = 0.35  # below this -> incomplete field of view


def _laplacian_variance(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _retinal_area_fraction(gray: np.ndarray) -> float:
    """Rough field-of-view check: fraction of pixels above a low-intensity
    threshold, i.e. not part of the black border/background."""
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(mask)) / mask.size


def compute_quality_score(image: np.ndarray) -> dict:
    """
    Combines Laplacian variance (focus), brightness, contrast, and retinal
    field coverage into a single 0-100 score, plus a status label and a
    single dominant failure reason for the recapture message.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lap_var = _laplacian_variance(gray)
    mean_brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    retinal_fraction = _retinal_area_fraction(gray)

    # Normalize each metric to 0-100, then combine with simple weights.
    # These caps are heuristic starting points for a small-scale prototype.
    focus_score = min(100.0, (lap_var / 300.0) * 100.0)
    brightness_score = 100.0 - min(100.0, abs(mean_brightness - 130) / 130 * 100.0)
    contrast_score = min(100.0, (contrast_std / 60.0) * 100.0)
    field_score = min(100.0, (retinal_fraction / 0.6) * 100.0)

    score = 0.35 * focus_score + 0.25 * brightness_score + 0.20 * contrast_score + 0.20 * field_score
    score = round(float(np.clip(score, 0, 100)), 1)

    # Determine dominant failure reason by checking the worst offender first
    failure_reason = None
    if lap_var < BLUR_LAPLACIAN_MIN:
        failure_reason = "blur"
    elif mean_brightness < DARK_MEAN_MAX:
        failure_reason = "low_illumination"
    elif mean_brightness > BRIGHT_MEAN_MIN:
        failure_reason = "glare"
    elif contrast_std < LOW_CONTRAST_STD_MAX:
        failure_reason = "low_contrast"
    elif retinal_fraction < MIN_RETINAL_AREA_FRACTION:
        failure_reason = "incomplete_field"

    if score >= GOOD_THRESHOLD:
        status = "good"
    elif score >= BORDERLINE_THRESHOLD:
        status = "borderline"
    else:
        status = "ungradable"

    return {
        "score": score,
        "status": status,
        "failure_reason": failure_reason,
        # exposed for debugging/tuning, not part of the locked contract
        "_debug_metrics": {
            "laplacian_variance": lap_var,
            "mean_brightness": mean_brightness,
            "contrast_std": contrast_std,
            "retinal_area_fraction": retinal_fraction,
        },
    }


def get_recapture_message(failure_reason: str) -> str:
    messages = {
        "blur": "Image ungradable due to excessive blur. Please stabilize the camera and recapture.",
        "low_illumination": "Image too dark. Please improve lighting and recapture.",
        "low_contrast": "Low contrast detected. Please recapture with correct focus/exposure.",
        "incomplete_field": "Retinal field incomplete. Please recenter and recapture.",
        "glare": "Excessive glare detected. Please adjust camera angle and recapture.",
    }
    return messages.get(failure_reason, "Please recapture the image.")
