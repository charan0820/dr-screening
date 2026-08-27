"""
Owner: P2
Handles APTOS ingestion, stratified splitting, and augmentation.
Day 2: real implementations replace Day 1 stubs.
See CONTRACTS.md — signatures unchanged from Day 1.
"""
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

try:
    import albumentations as A
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False


def load_dataset(csv_path: str, image_dir: str) -> pd.DataFrame:
    """
    Reads APTOS's train.csv (columns: id_code, diagnosis) and builds full
    filepaths. Returns columns: ['image_id', 'filepath', 'label'].
    """
    df = pd.read_csv(csv_path)

    # APTOS 2019's official column names
    if "id_code" in df.columns and "diagnosis" in df.columns:
        df = df.rename(columns={"id_code": "image_id", "diagnosis": "label"})
    elif "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Unexpected CSV columns {list(df.columns)}. "
            "Expected APTOS's ['id_code', 'diagnosis'] or ['image_id', 'label']."
        )

    def _to_path(image_id):
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(image_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                return candidate
        # fall back to .png even if not found yet, so caller gets a clear error later
        return os.path.join(image_dir, f"{image_id}.png")

    df["filepath"] = df["image_id"].apply(_to_path)
    return df[["image_id", "filepath", "label"]]


def stratified_split(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42) -> dict:
    """
    Stratified split on 'label' to preserve class balance across
    train/val/test. NOTE: APTOS has no patient ID field, so this is an
    image-level split, not patient-level — call this out as a known
    limitation in the final report/thesis.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val_size, stratify=train_val["label"], random_state=seed
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def get_augmentation_pipeline(training: bool = True):
    """
    Medically-reasonable augmentation only — nothing that could obscure or
    fabricate lesion-relevant detail (no aggressive elastic warps, no heavy
    color inversion, etc.).
    """
    if not _HAS_ALBUMENTATIONS:
        # Fallback: return None: caller should skip augmentation if the
        # library isn't installed rather than crash the whole pipeline.
        return None

    if training:
        return A.Compose([
            A.Rotate(limit=25, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),
            A.GaussNoise(std_range=(0.03, 0.08), p=0.15),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
        ])
    else:
        return A.Compose([])  # no augmentation at val/test time


def load_image(filepath: str, target_size: int = 224) -> np.ndarray:
    """
    Loads an image from disk, converts BGR->RGB, resizes to target_size.
    Raises FileNotFoundError clearly rather than letting cv2 return None
    silently (a common source of confusing downstream crashes).
    """
    image = cv2.imread(filepath)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {filepath}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size, target_size))
    return image
