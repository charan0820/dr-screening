"""
Owner: P2
Handles APTOS ingestion, stratified patient/image-level splitting, and augmentation.
See CONTRACTS.md for the required function signatures — do not change them
without telling the team.
"""
import numpy as np
import pandas as pd


def load_dataset(csv_path: str, image_dir: str) -> pd.DataFrame:
    """
    TODO(P2): Read the APTOS train.csv, join with image_dir paths.
    Returns a DataFrame with columns: ['image_id', 'filepath', 'label']
    """
    # --- STUB: returns fake data so downstream modules can run today ---
    return pd.DataFrame({
        "image_id": [f"img_{i}" for i in range(10)],
        "filepath": [f"data/dummy/img_{i}.png" for i in range(10)],
        "label": np.random.randint(0, 5, size=10),
    })


def stratified_split(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42) -> dict:
    """
    TODO(P2): Real stratified split preserving class balance across
    train/val/test. Document that APTOS has no patient ID, so this is
    image-level split (mention as a known limitation in the report).
    """
    # --- STUB ---
    n = len(df)
    return {
        "train": df.iloc[: int(n * 0.7)],
        "val": df.iloc[int(n * 0.7): int(n * 0.85)],
        "test": df.iloc[int(n * 0.85):],
    }


def get_augmentation_pipeline(training: bool = True):
    """
    TODO(P2): Return an albumentations.Compose object.
    Rotation, flip, brightness/contrast, mild blur/noise — nothing that
    could obscure or invent lesion-relevant detail.
    """
    # --- STUB ---
    return None


def load_image(filepath: str) -> np.ndarray:
    """
    TODO(P2): cv2.imread + BGR->RGB conversion + resize.
    """
    # --- STUB: dummy RGB image ---
    return np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
