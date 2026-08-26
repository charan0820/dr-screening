"""
Owner: P4
Model architecture — transfer learning backbone + classification head.
See CONTRACTS.md for required signatures.
"""


def build_model(num_classes: int = 5, backbone: str = "efficientnet_b0"):
    """
    TODO(P4): Build a tf.keras.Model using EfficientNetB0 (imagenet weights,
    include_top=False) + GlobalAveragePooling + Dense(num_classes, softmax).
    Freeze backbone initially, unfreeze last N layers for fine-tuning once
    training stabilizes.
    """
    # --- STUB ---
    return None


def load_trained_model(checkpoint_path: str):
    """
    TODO(P4): tf.keras.models.load_model(checkpoint_path)
    """
    # --- STUB ---
    return None
