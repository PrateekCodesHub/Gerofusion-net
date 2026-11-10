"""
hybrid_model.py
===============
Geriatric Syndrome GeroFusion v3 — Multi-Modal Deep Learning Architecture

This module implements ALL model variants used in the GeroFusion system.

CNN Backbones (image branch):
  1. ResNet50       – Deep residual network, transfer learning from ImageNet
  2. MobileNetV2    – Lightweight depthwise separable convolutions
  3. EfficientNetB0 – Compound-scaled model (NASNet-family baseline)
  4. EfficientNetB4 – Larger compound-scaled variant
  5. GeriatricNet   – Custom-designed CNN tailored to this clinical domain

ANN Variants (tabular/clinical branch):
  1. Standard ANN   – 2-layer (64→32), baseline clinical encoder
  2. Deep ANN       – 4-layer (128→64→32→16), deeper extraction
  3. Attention ANN  – Self-attention pooling over feature embeddings
  4. Wide ANN       – Wide layers (256→128) with residual skip
  5. ResidualANN    – Skip connections in fully-connected layers

Additional Modalities:
  - Video Model     – TimeDistributed CNN + BiGRU for gait sequences
  - Audio Model     – 2D CNN on Mel-spectrogram for speech biomarkers

Fusion:
  CNN Features + ANN Features (+ Video/Audio optional)
      → Concatenation Layer
      → Dense(256) → Dense(128) → Softmax(5 classes)

Author  : Student Implementation
Project : Geriatric Syndrome GeroFusion v3
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 224, 224, 3
NUM_CLINICAL_FEATURES = 10   # age, sys_bp, dia_bp, hr, bmi, cog, mob, grip, gait, depr
NUM_CLASSES           = 5    # Frailty, Sarcopenia, Cognitive Impairment, Fall Risk, Depression
NUM_AUDIO_FRAMES      = 128  # mel-spectrogram time frames
NUM_MEL_BINS          = 64   # mel-spectrogram frequency bins
VIDEO_FRAMES          = 16   # frames per gait clip
VIDEO_H, VIDEO_W      = 112, 112

SYNDROME_LABELS = [
    "Frailty",
    "Sarcopenia",
    "Cognitive Impairment",
    "Fall Risk",
    "Depression",
]

CLINICAL_FEATURE_NAMES = [
    "age", "systolic_bp", "diastolic_bp", "heart_rate", "bmi",
    "cognitive_score", "mobility_score", "grip_strength",
    "gait_speed", "depression_score",
]


# ══════════════════════════════════════════════════════════════════════
#  ANN BRANCH — Structured Clinical Data Encoders
# ══════════════════════════════════════════════════════════════════════

def build_standard_ann(input_shape: int = NUM_CLINICAL_FEATURES) -> keras.Model:
    """
    Standard 2-layer ANN for clinical tabular data.

    Architecture
    ────────────
    Input  : 10 neurons — one per clinical feature
    Dense1 : 64 neurons, ReLU, BatchNorm, Dropout(0.3)
    Dense2 : 32 neurons, ReLU, BatchNorm, Dropout(0.2)
    Output : 32-dim feature vector (passed to fusion layer)
    """
    inp = layers.Input(shape=(input_shape,), name="clinical_input")

    # Hidden layer 1: 64 neurons capture combinations of clinical features
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4), name="ann_dense1")(inp)
    x = layers.BatchNormalization(name="ann_bn1")(x)
    x = layers.Activation("relu", name="ann_relu1")(x)
    x = layers.Dropout(0.3, name="ann_drop1")(x)

    # Hidden layer 2: 32 neurons compress to salient clinical representation
    x = layers.Dense(32, kernel_regularizer=regularizers.l2(1e-4), name="ann_dense2")(x)
    x = layers.BatchNormalization(name="ann_bn2")(x)
    x = layers.Activation("relu", name="ann_relu2")(x)
    x = layers.Dropout(0.2, name="ann_drop2")(x)

    return keras.Model(inputs=inp, outputs=x, name="standard_ann")


def build_deep_ann(input_shape: int = NUM_CLINICAL_FEATURES) -> keras.Model:
    """
    Deep 4-layer ANN: 128 → 64 → 32 → 16 neurons.

    Rationale: more layers let the network learn higher-order
    interactions between clinical markers (e.g., combined effect
    of low grip strength AND low gait speed on Sarcopenia risk).
    """
    inp = layers.Input(shape=(input_shape,), name="clinical_input")

    x = layers.Dense(128, name="dnn_d1")(inp)
    x = layers.BatchNormalization(name="dnn_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, name="dnn_d2")(x)
    x = layers.BatchNormalization(name="dnn_bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, name="dnn_d3")(x)
    x = layers.BatchNormalization(name="dnn_bn3")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(16, name="dnn_d4")(x)
    x = layers.BatchNormalization(name="dnn_bn4")(x)
    x = layers.Activation("relu")(x)

    return keras.Model(inputs=inp, outputs=x, name="deep_ann")


def build_attention_ann(input_shape: int = NUM_CLINICAL_FEATURES) -> keras.Model:
    """
    ANN with soft-attention: learns which clinical features matter most
    per patient and weights them dynamically before classification.

    Attention mechanism:
        score_i = softmax( W_a · h_i )
        context = sum( score_i * h_i )

    This mimics how a clinician focuses on the most abnormal readings.
    """
    inp = layers.Input(shape=(input_shape,), name="clinical_input")

    # Project to 64-d
    x = layers.Dense(64, activation="relu", name="att_proj")(inp)
    x = layers.BatchNormalization()(x)

    # Attention scores (sigmoid so multiple features can be highlighted)
    att_scores = layers.Dense(64, activation="sigmoid", name="att_scores")(x)
    x = layers.Multiply(name="att_weighted")([x, att_scores])

    x = layers.Dense(32, activation="relu", name="att_fc")(x)
    x = layers.Dropout(0.3)(x)

    return keras.Model(inputs=inp, outputs=x, name="attention_ann")


def build_wide_ann(input_shape: int = NUM_CLINICAL_FEATURES) -> keras.Model:
    """
    Wide ANN: 256 → 128 neurons — larger receptive field for clinical data.
    A skip connection between input projection and second layer
    allows gradients to flow cleanly through the wider network.
    """
    inp = layers.Input(shape=(input_shape,), name="clinical_input")

    # Wide first layer: 256 neurons
    x = layers.Dense(256, activation="relu", name="wide_d1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Second layer: 128 neurons
    x = layers.Dense(128, activation="relu", name="wide_d2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    # Skip projection: align input dimension to 128 for residual add
    skip = layers.Dense(128, name="wide_skip")(inp)
    x = layers.Add(name="wide_residual")([x, skip])
    x = layers.Activation("relu")(x)

    # Compress to 32-d output embedding
    x = layers.Dense(32, activation="relu", name="wide_out")(x)

    return keras.Model(inputs=inp, outputs=x, name="wide_ann")


def build_residual_ann(input_shape: int = NUM_CLINICAL_FEATURES) -> keras.Model:
    """
    ResidualANN: fully-connected layers with explicit residual connections.
    Inspired by ResNet but applied to tabular clinical data.
    Helps avoid vanishing gradients in deeper clinical encoders.
    """
    inp = layers.Input(shape=(input_shape,), name="clinical_input")

    # Initial projection to 64-d
    x = layers.Dense(64, activation="relu", name="res_init")(inp)

    # Residual block 1
    res = x
    x = layers.Dense(64, activation="relu", name="res_b1_fc1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, name="res_b1_fc2")(x)
    x = layers.Add()([x, res])
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    # Residual block 2: downsample to 32
    res2 = layers.Dense(32, name="res_skip2")(x)
    x = layers.Dense(32, activation="relu", name="res_b2_fc1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, name="res_b2_fc2")(x)
    x = layers.Add()([x, res2])
    x = layers.Activation("relu")(x)

    return keras.Model(inputs=inp, outputs=x, name="residual_ann")


# ══════════════════════════════════════════════════════════════════════
#  CNN BRANCH — Image Feature Extractors
# ══════════════════════════════════════════════════════════════════════

def _freeze_base(base_model, freeze_ratio: float = 0.8):
    """Freeze the bottom `freeze_ratio` fraction of layers for fine-tuning."""
    n = len(base_model.layers)
    cutoff = int(n * freeze_ratio)
    for layer in base_model.layers[:cutoff]:
        layer.trainable = False
    for layer in base_model.layers[cutoff:]:
        layer.trainable = True


def build_resnet50_branch() -> keras.Model:
    """
    ResNet50 CNN branch.

    Input  : 224 × 224 × 3 RGB patient image
    Base   : ResNet50 pretrained on ImageNet (50-layer residual network)
    Head   : GlobalAveragePooling → Dense(256, ReLU) → Dense(128, ReLU)
    Output : 128-dim visual feature vector

    Why ResNet50?
    - Deep residual connections allow very deep networks to train without
      vanishing gradients.
    - ImageNet pretraining transfers texture and shape priors to medical images.
    - Top accuracy in the comparison but largest memory footprint (~98 MB).
    """
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
    )
    _freeze_base(base, freeze_ratio=0.80)  # Fine-tune top 20% of layers

    inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name="resnet_gap")(x)
    x = layers.Dense(256, activation="relu", name="resnet_fc1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", name="resnet_fc2")(x)

    return keras.Model(inputs=inp, outputs=x, name="resnet50_branch")


def build_mobilenetv2_branch() -> keras.Model:
    """
    MobileNetV2 CNN branch.

    Input  : 224 × 224 × 3 RGB image
    Base   : MobileNetV2 (depthwise separable convolutions, 3.4M params)
    Head   : GlobalAveragePooling → Dense(128, ReLU) → Dense(64, ReLU)
    Output : 64-dim visual feature vector

    Why MobileNetV2?
    - Depthwise separable convolutions reduce FLOPs by ~8× vs standard conv.
    - Ideal for edge deployment and GTX 1650 real-time inference.
    - Slightly lower accuracy than ResNet50 but trains ~3× faster.
    """
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        alpha=1.0,
    )
    _freeze_base(base, freeze_ratio=0.75)

    inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name="mob_gap")(x)
    x = layers.Dense(128, activation="relu", name="mob_fc1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu", name="mob_fc2")(x)

    return keras.Model(inputs=inp, outputs=x, name="mobilenetv2_branch")


def build_efficientnet_branch(variant: str = "B0") -> keras.Model:
    """
    EfficientNet branch (B0 or B4).

    Input  : 224 × 224 × 3 RGB image
    Base   : EfficientNetB0 (~5.3M params) or EfficientNetB4 (~19M params)
    Head   : GlobalAveragePooling → Dense(128, ReLU) → Dense(64, ReLU)
    Output : 64-dim visual feature vector

    Why EfficientNet?
    - Compound scaling: simultaneously scales depth, width, and resolution.
    - EfficientNetB0 achieves ImageNet accuracy comparable to ResNet50
      with 8× fewer parameters.
    - EfficientNetB4 is the best accuracy/size tradeoff in this suite.
    """
    if variant == "B4":
        base = tf.keras.applications.EfficientNetB4(
            include_top=False, weights="imagenet",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        )
        out_dim = 128
    else:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        )
        out_dim = 64

    _freeze_base(base, freeze_ratio=0.80)

    inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name=f"eff{variant}_gap")(x)
    x = layers.Dense(256, activation="relu", name=f"eff{variant}_fc1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(out_dim, activation="relu", name=f"eff{variant}_fc2")(x)

    return keras.Model(inputs=inp, outputs=x, name=f"efficientnet{variant}_branch")


def build_geriatricnet_branch() -> keras.Model:
    """
    GeriatricNet — Custom CNN designed for geriatric image analysis.

    Architecture
    ────────────
    Input       : 224 × 224 × 3 RGB image

    Block 1 (Edge detection):
      Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)
      Captures low-level edges, contours in posture images.

    Block 2 (Texture extraction):
      Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2)
      Extracts facial texture and muscle definition patterns.

    Block 3 (Semantic features):
      Conv2D(128, 3×3) → BN → ReLU → MaxPool(2×2)
      Captures higher-level patterns (slouched posture, facial droop).

    Block 4 (Global context):
      Conv2D(256, 3×3) → BN → ReLU
      Learns full-body spatial context.

    Depthwise Separable Block (efficiency):
      DepthwiseConv2D(3×3) → PointwiseConv2D(256)
      Reduces parameters while preserving representational power.

    Head:
      GlobalAveragePooling → Dense(256) → Dense(128)

    Output : 128-dim visual feature vector

    This custom architecture is narrower than ResNet50 but achieves
    competitive accuracy while being interpretable via Grad-CAM.
    """
    inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")

    # Block 1 — low-level edge detection
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="gn_conv1")(inp)
    x = layers.BatchNormalization(name="gn_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="gn_conv1b")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="gn_pool1")(x)  # 112 × 112

    # Block 2 — texture extraction
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="gn_conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="gn_conv2b")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="gn_pool2")(x)  # 56 × 56
    x = layers.Dropout(0.2)(x)

    # Block 3 — semantic pattern detection
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="gn_conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="gn_conv3b")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="gn_pool3")(x)  # 28 × 28
    x = layers.Dropout(0.25)(x)

    # Block 4 — global context
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="gn_conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="gn_pool4")(x)  # 14 × 14

    # Depthwise Separable Block — efficiency layer
    x = layers.DepthwiseConv2D((3, 3), padding="same", name="gn_dw")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, (1, 1), padding="same", name="gn_pw")(x)  # pointwise
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="gn_pool5")(x)  # 7 × 7

    # Head
    x = layers.GlobalAveragePooling2D(name="gn_gap")(x)
    x = layers.Dense(256, activation="relu", name="gn_fc1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", name="gn_fc2")(x)
    x = layers.Dropout(0.3)(x)

    return keras.Model(inputs=inp, outputs=x, name="geriatricnet_branch")


# ══════════════════════════════════════════════════════════════════════
#  AUDIO BRANCH — Mel-Spectrogram CNN for Speech Biomarkers
# ══════════════════════════════════════════════════════════════════════

def build_audio_branch(
    n_frames: int = NUM_AUDIO_FRAMES,
    n_mels: int = NUM_MEL_BINS,
) -> keras.Model:
    """
    Audio CNN branch: 2D convolutions on Mel-spectrogram images.

    Input  : (128, 64, 1) — time × mel_bins × channel
    Output : 32-dim audio feature vector

    Speech biomarkers captured:
    - Pitch variation (monotone speech → cognitive decline indicator)
    - Speech rate (slowed articulation → frailty/cognitive impairment)
    - Voice tremor frequency (Parkinson-related gait disorders)
    - Silence ratio (depression marker)
    """
    inp = layers.Input(shape=(n_frames, n_mels, 1), name="audio_input")

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="aud_c1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="aud_c2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="aud_c3")(x)
    x = layers.GlobalAveragePooling2D(name="aud_gap")(x)
    x = layers.Dense(64, activation="relu", name="aud_fc1")(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(32, activation="relu", name="aud_fc2")(x)

    return keras.Model(inputs=inp, outputs=x, name="audio_branch")


# ══════════════════════════════════════════════════════════════════════
#  VIDEO BRANCH — TimeDistributed CNN + BiGRU for Gait Sequences
# ══════════════════════════════════════════════════════════════════════

def build_video_branch(
    n_frames: int = VIDEO_FRAMES,
    frame_h: int = VIDEO_H,
    frame_w: int = VIDEO_W,
) -> keras.Model:
    """
    Video branch: models temporal gait and posture dynamics.

    Architecture
    ────────────
    Input  : (16, 112, 112, 3) — 16 frames × 112×112 RGB

    TimeDistributed(MobileNetV2-lite):
        Each frame independently encoded by a lightweight CNN.
        Shared weights ensure temporal consistency.

    BiGRU(64):
        Bidirectional GRU captures both forward and backward
        temporal dynamics in gait patterns (e.g., stumble → recover).

    Output : 64-dim temporal feature vector

    Why this design?
    - Pure 3D CNNs (C3D/I3D) require 8–16 GB GPU memory.
    - TimeDistributed + GRU is memory-efficient for GTX 1650 (4 GB VRAM).
    - BiGRU reads both directions: useful for detecting asymmetric gait.
    """
    inp = layers.Input(shape=(n_frames, frame_h, frame_w, 3), name="video_input")

    # Small shared CNN per frame (lighter than MobileNetV2 for memory)
    frame_cnn = keras.Sequential([
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
    ], name="frame_cnn")

    x = layers.TimeDistributed(frame_cnn, name="td_cnn")(inp)

    # Bidirectional GRU for temporal modeling
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False), name="bigru")(x)
    x = layers.Dense(64, activation="relu", name="vid_fc")(x)
    x = layers.Dropout(0.4)(x)

    return keras.Model(inputs=inp, outputs=x, name="video_branch")


# ══════════════════════════════════════════════════════════════════════
#  HYBRID FUSION MODEL
# ══════════════════════════════════════════════════════════════════════

CNN_BUILDERS = {
    "resnet50":      build_resnet50_branch,
    "mobilenetv2":   build_mobilenetv2_branch,
    "efficientnetb0": lambda: build_efficientnet_branch("B0"),
    "efficientnetb4": lambda: build_efficientnet_branch("B4"),
    "custom":        build_geriatricnet_branch,
}

ANN_BUILDERS = {
    "standard":  build_standard_ann,
    "deep":      build_deep_ann,
    "attention": build_attention_ann,
    "wide":      build_wide_ann,
    "residual":  build_residual_ann,
}


def build_hybrid_model(
    cnn_type: str = "resnet50",
    ann_type: str = "standard",
    use_audio: bool = False,
    use_video: bool = False,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.4,
) -> keras.Model:
    """
    Builds the full Hybrid CNN-ANN (+ optional Audio/Video) model.

    Feature Fusion
    ──────────────
    CNN features  (128-d or 64-d depending on backbone)
    ANN features  (32-d or 16-d depending on variant)
    Audio features (32-d, if use_audio=True)
    Video features (64-d, if use_video=True)
            ↓ Concatenate
    Fusion Dense(256) → BN → ReLU → Dropout(0.4)
    Fusion Dense(128) → BN → ReLU → Dropout(0.3)
    Output Dense(num_classes) → Softmax

    Parameters
    ──────────
    cnn_type   : one of 'resnet50', 'mobilenetv2', 'efficientnetb0',
                 'efficientnetb4', 'custom'
    ann_type   : one of 'standard', 'deep', 'attention', 'wide', 'residual'
    use_audio  : include mel-spectrogram audio branch
    use_video  : include temporal gait video branch
    """
    # ── Build sub-branches ──────────────────────────────────────────
    cnn_branch = CNN_BUILDERS[cnn_type]()
    ann_branch = ANN_BUILDERS[ann_type]()

    image_input    = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
    clinical_input = layers.Input(shape=(NUM_CLINICAL_FEATURES,), name="clinical_input")

    cnn_features = cnn_branch(image_input)
    ann_features = ann_branch(clinical_input)

    to_concat = [cnn_features, ann_features]
    all_inputs = [image_input, clinical_input]

    if use_audio:
        audio_branch = build_audio_branch()
        audio_input = layers.Input(
            shape=(NUM_AUDIO_FRAMES, NUM_MEL_BINS, 1), name="audio_input"
        )
        audio_features = audio_branch(audio_input)
        to_concat.append(audio_features)
        all_inputs.append(audio_input)

    if use_video:
        video_branch = build_video_branch()
        video_input = layers.Input(
            shape=(VIDEO_FRAMES, VIDEO_H, VIDEO_W, 3), name="video_input"
        )
        video_features = video_branch(video_input)
        to_concat.append(video_features)
        all_inputs.append(video_input)

    # ── Fusion ─────────────────────────────────────────────────────
    if len(to_concat) > 1:
        fused = layers.Concatenate(name="fusion_concat")(to_concat)
    else:
        fused = to_concat[0]

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), name="fusion_d1")(fused)
    x = layers.BatchNormalization(name="fusion_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate, name="fusion_drop1")(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4), name="fusion_d2")(x)
    x = layers.BatchNormalization(name="fusion_bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate * 0.75, name="fusion_drop2")(x)

    # ── Output ─────────────────────────────────────────────────────
    output = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model_name = f"gerofusion_{cnn_type}_{ann_type}"
    if use_audio:
        model_name += "_audio"
    if use_video:
        model_name += "_video"

    return keras.Model(inputs=all_inputs, outputs=output, name=model_name)


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-4,
    label_smoothing: float = 0.1,
) -> keras.Model:
    """
    Compiles model with Adam optimiser, categorical cross-entropy
    with label smoothing (reduces overconfident predictions).
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
