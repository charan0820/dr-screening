"""
Owner: P5
Grad-CAM explainability overlay.
Day 2: real tf.GradientTape-based implementation. Falls back to returning
the original image if no trained model exists yet (Day 3+ delivers that).
"""
import numpy as np
import cv2
import tensorflow as tf


def _find_last_conv_index(model):
    """Finds the index (in model.layers) of the last layer with a 4D
    output (batch, H, W, channels) — works even when the conv backbone
    (e.g. EfficientNetB0) is nested as a single sub-model layer, since
    that sub-model's own output is itself 4D when include_top=False."""
    last_idx = None
    for i, layer in enumerate(model.layers):
        try:
            shape = layer.output.shape
        except (AttributeError, ValueError):
            continue
        if shape is not None and len(shape) == 4:
            last_idx = i
    if last_idx is None:
        raise ValueError("No 4D (conv) layer found in model for Grad-CAM.")
    return last_idx


def _build_grad_model(model, last_conv_idx):
    """
    Retraces the model's graph from a fresh Input tensor, reusing the same
    layer objects (weights shared, nothing retrained) so we get valid
    references to an intermediate conv output. This sidesteps a Keras 3
    limitation where you can't pull an intermediate tensor out of an
    already-built graph when the conv backbone is a nested sub-model.
    """
    input_shape = model.input_shape[1:]
    inp = tf.keras.Input(shape=input_shape)
    x = inp
    conv_output = None
    for i, layer in enumerate(model.layers):
        if i == 0:
            continue  # original InputLayer — skip, we already made a fresh one
        x = layer(x, training=False)
        if i == last_conv_idx:
            conv_output = x
    return tf.keras.Model(inp, [conv_output, x])


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

    last_conv_idx = _find_last_conv_index(model)
    grad_model = _build_grad_model(model, last_conv_idx)

    # Model input always float32; overlay base always uint8 regardless of
    # what dtype the caller passed in.
    batch = np.expand_dims(image, axis=0).astype(np.float32)
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
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

    heatmap = cv2.resize(heatmap, (image_uint8.shape[1], image_uint8.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_uint8, 0.6, heatmap_color, 0.4, 0)
    return overlay
