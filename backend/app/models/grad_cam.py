"""
grad_cam.py
===========
Gradient-weighted Class Activation Mapping (Grad-CAM) for GeroFusion v3.

Generates heatmaps showing which spatial regions of a patient image most
strongly influenced the model's prediction. This explainability tool is
critical for clinical trust: clinicians can verify that the network attends
to anatomically meaningful regions (e.g., facial muscles for frailty,
spinal curvature for fall risk).

Reference:
    Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.
"""

import io
import base64
import numpy as np

import tensorflow as tf
from tensorflow import keras

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ── Last-conv layer names per backbone ──────────────────────────────
LAST_CONV_LAYERS = {
    "resnet50":       "conv5_block3_out",
    "mobilenetv2":    "out_relu",
    "efficientnetb0": "top_activation",
    "efficientnetb4": "top_activation",
    "custom":         "gn_pw",     # GeriatricNet pointwise conv
}


def _all_layers(model: keras.Model):
    """Yield all layers including those inside nested sub-models."""
    for layer in model.layers:
        yield layer
        if hasattr(layer, 'layers'):
            yield from _all_layers(layer)


def get_last_conv_layer(model: keras.Model, cnn_type: str) -> str:
    """Find the last convolutional layer name for a given backbone."""
    target = LAST_CONV_LAYERS.get(cnn_type, "")
    for layer in reversed(list(_all_layers(model))):
        if target and layer.name == target:
            return layer.name
        if not target and isinstance(layer, (
            keras.layers.Conv2D, keras.layers.DepthwiseConv2D
        )):
            return layer.name
    raise ValueError(f"Cannot find last conv layer for {cnn_type}")


def compute_gradcam(
    model: keras.Model,
    image_array: np.ndarray,
    clinical_array: np.ndarray,
    class_idx: int,
    cnn_type: str = "custom",
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single image.

    Parameters
    ──────────
    model         : compiled Keras hybrid model
    image_array   : (1, 224, 224, 3) float32 [0, 1]
    clinical_array: (1, 10) float32 normalised
    class_idx     : syndrome index to explain (0–4)
    cnn_type      : backbone name for last-conv lookup

    Returns
    ───────
    heatmap : (224, 224) float32 [0, 1] — Grad-CAM activation map
    """
    last_conv_name = get_last_conv_layer(model, cnn_type)

    # Find the layer object (may be inside a nested sub-model)
    conv_layer = None
    for layer in _all_layers(model):
        if layer.name == last_conv_name:
            conv_layer = layer
            break
    if conv_layer is None:
        raise ValueError(f"Layer {last_conv_name} not found")

    # Build grad model: maps inputs → [last_conv_output, final_predictions]
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            conv_layer.output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        inputs = [tf.cast(image_array, tf.float32),
                  tf.cast(clinical_array, tf.float32)]
        conv_outputs, predictions = grad_model(inputs)
        class_score = predictions[:, class_idx]

    # Gradients of class score w.r.t. last conv feature maps
    grads = tape.gradient(class_score, conv_outputs)        # (1, H, W, C)

    # Global average pooling over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # (C,)

    # Weight feature maps by their gradient importance
    conv_outputs = conv_outputs[0]                           # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                           # (H, W)

    # ReLU — only positive influences
    heatmap = tf.nn.relu(heatmap)

    # Normalise to [0, 1]
    heatmap = heatmap.numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # Resize to input image size
    if CV2_OK:
        heatmap = cv2.resize(heatmap, (224, 224))
    elif PIL_OK:
        from PIL import Image as PILImage
        heatmap_img = PILImage.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((224, 224), PILImage.BILINEAR)
        heatmap = np.array(heatmap_img) / 255.0
    else:
        # Simple nearest-neighbour upscale
        h, w = heatmap.shape
        heatmap = np.repeat(np.repeat(heatmap, 224 // h + 1, axis=0),
                            224 // w + 1, axis=1)[:224, :224]

    return heatmap.astype(np.float32)


def overlay_heatmap(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto the original image.

    Parameters
    ──────────
    image_array : (224, 224, 3) float32 [0, 1]
    heatmap     : (224, 224) float32 [0, 1]
    alpha       : blending weight for the heatmap

    Returns
    ───────
    overlay : (224, 224, 3) uint8 RGB
    """
    img = (image_array[0] * 255).astype(np.uint8) if image_array.ndim == 4 else (image_array * 255).astype(np.uint8)

    if CV2_OK:
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img, 1 - alpha, colormap, alpha, 0)
    else:
        # Fallback: red channel heatmap
        overlay = img.copy().astype(np.float32)
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + heatmap * 120, 0, 255)
        overlay[:, :, 1] = np.clip(overlay[:, :, 1] - heatmap * 60, 0, 255)
        overlay = overlay.astype(np.uint8)

    return overlay


def gradcam_to_base64(
    model: keras.Model,
    image_array: np.ndarray,
    clinical_array: np.ndarray,
    class_idx: int,
    cnn_type: str = "custom",
) -> str:
    """
    Full Grad-CAM pipeline: compute → overlay → encode as base64 PNG.
    Returns a base64 string suitable for embedding in JSON API responses.
    """
    if not PIL_OK:
        return ""

    try:
        heatmap = compute_gradcam(model, image_array, clinical_array, class_idx, cnn_type)
        overlay = overlay_heatmap(image_array, heatmap)

        pil_img = Image.fromarray(overlay)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return b64
    except Exception as e:
        print(f"  Grad-CAM error: {e}")
        return ""
