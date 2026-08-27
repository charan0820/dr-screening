"""
Owner: P5
Grad-CAM explainability overlay.
Day 2: real tf.GradientTape-based implementation. Falls back to returning
the original image if no trained model exists yet (Day 3+ delivers that).
"""
import numpy as np
import cv2
import tensorflow as tf


def _find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:  # (batch, H, W, channels)
            return layer.name
    raise ValueError("No 4D (conv) layer found in model for Grad-CAM.")


def generate_gradcam(model, image: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Standard Grad-CAM: gradients of the target class w.r.t. the last conv
    layer's activations, weighted sum -> ReLU -> normalize -> resize ->
    blended heatmap overlay.
    """
    if model is None or class_idx is None:
        # No trained model yet — return original image unchanged so the
        # rest of the pipeline (GUI, report) stays runnable.
        return image

    last_conv_name = _find_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_name).output, model.output]
    )

    batch = np.expand_dims(image, axis=0).astype(np.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    return overlay
