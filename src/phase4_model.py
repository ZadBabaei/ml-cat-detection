"""
Phase 4: Model Architecture
=============================
Cat Detection Project — TensorFlow/Keras

This script defines THREE models so you can compare approaches:
  A) Logistic Regression baseline  (simplest possible model)
  B) Small CNN from scratch        (the main learning exercise)
  C) Transfer Learning (MobileNetV2) (best real-world approach)

Each layer is explained in detail so you understand what it does
and WHY it's there.

Usage:
    python src/phase4_model.py
"""

import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==============================================================
# CONFIG
# ==============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = 32   # CIFAR-10 image size
NUM_CHANNELS = 3


# ==============================================================
# MODEL A — Logistic Regression (baseline)
# ==============================================================
def build_baseline_model():
    """
    The simplest possible "model": flatten the image into a 1-D vector
    and pass it through a single neuron with a sigmoid activation.

    This is equivalent to logistic regression:
        P(cat) = sigmoid(W · pixels + b)

    WHY: It gives us a lower bound on performance. If our CNN can't
    beat this, something is wrong with the CNN.

    Expected accuracy: ~90% (by always predicting "not_cat" since
    cats are only 10% of the data). So this baseline shows us that
    raw accuracy is misleading with imbalanced data!
    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)),
        # Flatten: (32, 32, 3) → (3072,)
        # Turns the 2D image into a flat vector of numbers
        layers.Flatten(),
        # Dense(1) + sigmoid: single output neuron
        # sigmoid squashes output to [0, 1] = probability of being a cat
        layers.Dense(1, activation="sigmoid"),
    ], name="baseline_logistic_regression")

    return model


# ==============================================================
# MODEL B — Small CNN from scratch  ⭐ MAIN MODEL
# ==============================================================
def build_cnn_model():
    """
    A small Convolutional Neural Network built from scratch.

    ARCHITECTURE OVERVIEW:
    ┌─────────────────────────────────────────────────────────┐
    │  Input (32×32×3)                                        │
    │    ↓                                                    │
    │  Conv2D(32 filters, 3×3) + ReLU  → "detect edges"      │
    │  Conv2D(32 filters, 3×3) + ReLU  → "detect textures"   │
    │  MaxPool(2×2)                    → shrink to 16×16      │
    │  Dropout(0.25)                   → prevent overfitting  │
    │    ↓                                                    │
    │  Conv2D(64 filters, 3×3) + ReLU  → "detect shapes"     │
    │  Conv2D(64 filters, 3×3) + ReLU  → "detect parts"      │
    │  MaxPool(2×2)                    → shrink to 8×8        │
    │  Dropout(0.25)                   → prevent overfitting  │
    │    ↓                                                    │
    │  Flatten                         → 1-D vector           │
    │  Dense(128) + ReLU               → combine features     │
    │  Dropout(0.5)                    → strong regularization│
    │  Dense(1) + Sigmoid              → probability of cat   │
    └─────────────────────────────────────────────────────────┘

    LAYER-BY-LAYER EXPLANATION:
    """
    model = keras.Sequential(name="cat_detector_cnn")

    # --- Input ---
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)))

    # --- Block 1: Low-level features (edges, colors) ---

    # Conv2D(32, (3,3)):
    #   - 32 filters: the network learns 32 different 3×3 patterns
    #   - Each filter slides across the image and produces a "feature map"
    #   - padding="same": output has same spatial size as input
    #   - Early layers learn simple things: horizontal edges, color blobs
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))

    # Second Conv2D: stacking two convolutions lets the network learn
    # slightly more complex patterns (combinations of edges = textures)
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))

    # MaxPool2D(2,2):
    #   - Takes each 2×2 block and keeps only the maximum value
    #   - Reduces spatial size by half: 32×32 → 16×16
    #   - WHY: Makes the model more robust to small shifts in position
    #     and reduces computation for subsequent layers
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Dropout(0.25):
    #   - During training, randomly "turns off" 25% of neurons
    #   - Forces the network to not rely on any single neuron
    #   - This is REGULARIZATION — it prevents overfitting
    #   - During inference (prediction), all neurons are active
    model.add(layers.Dropout(0.25))

    # --- Block 2: Higher-level features (shapes, object parts) ---

    # 64 filters: more filters because we need to detect more complex patterns
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 16×16 → 8×8
    model.add(layers.Dropout(0.25))

    # --- Classification head ---

    # Flatten: reshape from (8, 8, 64) = 4096 values → (4096,) 1-D vector
    # We need a flat vector to feed into Dense layers
    model.add(layers.Flatten())

    # Dense(128):
    #   - A "fully connected" layer: every input connects to every neuron
    #   - 128 neurons combine all the spatial features into a decision
    #   - ReLU activation: f(x) = max(0, x) — simple, fast, effective
    model.add(layers.Dense(128, activation="relu"))

    # Dropout(0.5):
    #   - Stronger dropout (50%) in the dense layer
    #   - Dense layers have many parameters → more prone to overfitting
    model.add(layers.Dropout(0.5))

    # Dense(1) + Sigmoid:
    #   - Single output neuron for binary classification
    #   - Sigmoid: squashes output to [0, 1]
    #   - Output = P(image is a cat)
    #   - If output > 0.5 → predict "cat", else → predict "not cat"
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


# ==============================================================
# MODEL C — Transfer Learning (MobileNetV2)
# ==============================================================
def build_transfer_model():
    """
    Transfer Learning: use a model pretrained on ImageNet (1.2M images,
    1000 classes) and adapt it for our cat/not-cat task.

    WHY TRANSFER LEARNING WORKS:
    - ImageNet models already know how to detect edges, textures,
      shapes, and even animal features.
    - We "freeze" those learned features and only train a small
      classification head on top.
    - This works well even with small datasets.

    NOTE: MobileNetV2 expects at least 32×32 images, which matches
    our CIFAR-10 data. For best results, resize to 96×96 or 224×224.
    """
    # Load MobileNetV2 without the top classification layer
    # include_top=False: remove the original 1000-class head
    # weights="imagenet": use pretrained weights (requires download)
    # For sandbox/offline: weights=None (random init, defeats the purpose)
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS),
        include_top=False,
        weights=None,  # Change to "imagenet" on your machine for real transfer learning!
    )

    # Freeze all pretrained layers: they won't be updated during training
    base_model.trainable = False

    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)),
        base_model,
        # Global Average Pooling: collapse spatial dims (H×W) into a single vector
        # Much fewer parameters than Flatten → less overfitting
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="transfer_mobilenetv2")

    return model


# ==============================================================
# COMPILE HELPER
# ==============================================================
def compile_model(model, learning_rate=0.001):
    """
    Compile the model: define how it learns.

    LOSS: BinaryCrossentropy
      - Standard loss for binary classification
      - Measures how far the predicted probability is from the true label
      - Loss = -[y·log(p) + (1-y)·log(1-p)]
      - Perfect prediction → loss = 0

    OPTIMIZER: Adam
      - Adaptive learning rate optimizer
      - Combines the best of SGD + Momentum + RMSprop
      - learning_rate=0.001 is a good default starting point

    METRICS: We track accuracy during training for quick feedback,
    but we'll use better metrics (precision, recall, F1) in Phase 8.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("\n🐱 Cat Detection — Phase 4: Model Architecture\n")

    # Build all three models
    models = {
        "A) Baseline (Logistic Regression)": build_baseline_model(),
        "B) CNN from scratch ⭐": build_cnn_model(),
        "C) Transfer Learning (MobileNetV2)": build_transfer_model(),
    }

    for name, model in models.items():
        compile_model(model)
        print("\n" + "=" * 60)
        print(f"  {name}")
        print("=" * 60)
        model.summary()

        # Count parameters
        total = model.count_params()
        trainable = sum(
            tf.size(w).numpy() for w in model.trainable_weights
        )
        non_trainable = total - trainable
        print(f"\n  Total params     : {total:>10,}")
        print(f"  Trainable params : {trainable:>10,}")
        print(f"  Non-trainable    : {non_trainable:>10,}")

    print("\n" + "=" * 60)
    print("  WHICH MODEL SHOULD YOU USE?")
    print("=" * 60)
    print("  • Start with Model B (CNN from scratch) — it's the best")
    print("    for learning how CNNs work.")
    print("  • Use Model A as a baseline to compare against.")
    print("  • Try Model C later for the best real-world performance")
    print("    (set weights='imagenet' on your own machine).")

    print("\n✅ Phase 4 complete!")
    print("   Models are defined and compiled.")
    print("   👉 Next step: Phase 5 — Training & Backpropagation\n")


if __name__ == "__main__":
    main()
