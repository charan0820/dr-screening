"""
Owner: P4
Model architecture — EfficientNetB0 transfer-learning backbone + classification head.
Day 2: real architecture replaces Day 1 stub. Actual training happens Day 3.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(num_classes: int = 5, backbone: str = "efficientnet_b0", input_shape=(224, 224, 3)):
    """
    Transfer-learning model: frozen ImageNet backbone + GAP + dropout + dense
    softmax head. Backbone frozen for now — unfreeze last N layers on Day 3
    once initial training stabilizes.
    """
    if backbone == "efficientnet_b0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model(checkpoint_path: str):
    """Loads a saved .h5/.keras checkpoint for inference."""
    return tf.keras.models.load_model(checkpoint_path)
