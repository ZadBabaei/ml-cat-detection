"""
Phase 3: Preprocessing & Data Augmentation
============================================
Cat Detection Project — TensorFlow/Keras

This script:
  1. Loads the saved splits from Phase 1 (processed_splits.npz)
  2. Normalizes pixel values to [0, 1]
  3. Builds a tf.data pipeline with data augmentation (training only)
  4. Visualizes augmented samples so you can see what augmentation does
  5. Saves the pipeline config for use in later phases

KEY CONCEPTS EXPLAINED:
-----------------------
• Normalization: Neural networks work better when inputs are small numbers
  (0-1) instead of raw pixel values (0-255). This helps gradients flow
  smoothly and speeds up training.

• Data Augmentation: We artificially create "new" training images by
  applying random transformations (flip, rotate, zoom, brightness).
  This teaches the model to be robust to variations and reduces overfitting.
  IMPORTANT: We only augment the TRAINING set. Validation and test sets
  stay untouched so our evaluation is fair and consistent.

• tf.data Pipeline: TensorFlow's efficient data loading system. It
  prefetches and processes data in the background while the GPU trains,
  so the GPU never sits idle waiting for data.

Usage:
    python src/phase3_preprocessing.py
"""

import numpy as np
from pathlib import Path
from collections import Counter

import tensorflow as tf

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ==============================================================
# CONFIG
# ==============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed_splits.npz"

IMG_SIZE = 32         # CIFAR-10 images are 32x32; we keep them as-is
BATCH_SIZE = 32       # Number of images per training step
AUTOTUNE = tf.data.AUTOTUNE  # Let TF decide optimal prefetch buffer


# ==============================================================
# STEP 1 — Load saved data from Phase 1
# ==============================================================
def load_splits():
    """Load the .npz file saved by Phase 1."""
    print(f"  Loading data from: {DATA_PATH}")
    data = np.load(DATA_PATH)

    x_train, y_train = data["x_train"], data["y_train"]
    x_val,   y_val   = data["x_val"],   data["y_val"]
    x_test,  y_test  = data["x_test"],  data["y_test"]

    print(f"  Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_val, y_val, x_test, y_test


# ==============================================================
# STEP 2 — Normalize pixel values
# ==============================================================
def normalize(images):
    """
    Convert pixel values from [0, 255] integers to [0.0, 1.0] floats.

    WHY: Raw pixel values (0-255) create large gradients that make
    training unstable. Normalizing to [0,1] keeps the numbers small
    and helps the optimizer converge faster.
    """
    return images.astype(np.float32) / 255.0


# ==============================================================
# STEP 3 — Data Augmentation Layer
# ==============================================================
def build_augmentation_layer():
    """
    Create a Keras Sequential model that applies random transformations.

    Each transformation has a probability of being applied, so every
    time the same image passes through, it looks slightly different.

    These are the augmentations and WHY we use each:

    • RandomFlip("horizontal"):
      Cats can face left or right — the model should handle both.

    • RandomRotation(0.05):
      Slight tilt (±18°). Real photos aren't always perfectly level.

    • RandomZoom((-0.1, 0.1)):
      Slight zoom in/out. Cats appear at different distances.

    • RandomBrightness(0.2):
      Brightness variation. Photos are taken in different lighting.

    • RandomContrast(0.2):
      Contrast variation. Different cameras / conditions.
    """
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom((-0.1, 0.1)),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="data_augmentation")

    return augmentation


# ==============================================================
# STEP 4 — Build tf.data Pipelines
# ==============================================================
def build_dataset(x, y, batch_size, augment=False, shuffle=False):
    """
    Build a tf.data.Dataset pipeline.

    Parameters:
        x: images (float32, already normalized)
        y: labels (int32)
        batch_size: how many images per batch
        augment: if True, apply data augmentation (training only!)
        shuffle: if True, shuffle the data each epoch

    Pipeline steps:
        1. Create dataset from numpy arrays
        2. Shuffle (training only) — so the model sees images in random order
        3. Batch — group images into mini-batches
        4. Augment (training only) — random transformations
        5. Prefetch — load next batch while GPU processes current one
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=42)

    ds = ds.batch(batch_size)

    if augment:
        aug_layer = build_augmentation_layer()
        # Apply augmentation to images only (not labels), then clip to [0,1]
        ds = ds.map(
            lambda img, lbl: (tf.clip_by_value(aug_layer(img, training=True), 0.0, 1.0), lbl),
            num_parallel_calls=AUTOTUNE
        )

    ds = ds.prefetch(AUTOTUNE)
    return ds


# ==============================================================
# STEP 5 — Visualize augmented samples
# ==============================================================
def visualize_augmentation(x_train, y_train):
    """
    Pick one cat image and show it with 8 different random augmentations.
    This helps you understand what augmentation actually does.
    """
    if not HAS_MPL:
        print("  Skipping visualization (matplotlib not installed).")
        return

    aug_layer = build_augmentation_layer()

    # Find a cat image
    cat_idx = np.where(y_train == 1)[0][0]
    original = x_train[cat_idx]

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()

    # First image: original
    axes[0].imshow(np.clip(original, 0, 1))
    axes[0].set_title("ORIGINAL", fontsize=11, fontweight="bold", color="blue")
    axes[0].axis("off")

    # Remaining 9: augmented versions of the same image
    for i in range(1, 10):
        # Add batch dimension, augment, remove batch dimension
        augmented = aug_layer(tf.expand_dims(original, 0), training=True)
        augmented = augmented[0].numpy()
        axes[i].imshow(np.clip(augmented, 0, 1))
        axes[i].set_title(f"Augmented #{i}", fontsize=10)
        axes[i].axis("off")

    plt.suptitle(
        "Data Augmentation: Same image with random transformations\n"
        "(flip, rotate, zoom, brightness, contrast)",
        fontsize=13
    )
    plt.tight_layout()

    save_path = PROJECT_ROOT / "augmentation_demo.png"
    plt.savefig(save_path, dpi=120)
    print(f"\n  Augmentation demo saved to: {save_path}")
    plt.close()


# ==============================================================
# STEP 6 — Summary of the full pipeline
# ==============================================================
def print_pipeline_summary(train_ds, val_ds, test_ds):
    """Print what the pipeline produces."""
    print("\n" + "=" * 55)
    print("  PIPELINE SUMMARY")
    print("=" * 55)

    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        for batch_x, batch_y in ds.take(1):
            print(f"  {name:>5} dataset:")
            print(f"    Batch image shape : {batch_x.shape}")
            print(f"    Batch label shape : {batch_y.shape}")
            print(f"    Image value range : [{batch_x.numpy().min():.3f}, {batch_x.numpy().max():.3f}]")
            print(f"    Label dtype       : {batch_y.dtype}")
            print()

    print("  Training pipeline:   shuffle → batch → augment → prefetch")
    print("  Validation pipeline: batch → prefetch (no shuffle, no augment)")
    print("  Test pipeline:       batch → prefetch (no shuffle, no augment)")
    print("=" * 55)


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("\n🐱 Cat Detection — Phase 3: Preprocessing & Augmentation\n")

    # Load
    x_train, y_train, x_val, y_val, x_test, y_test = load_splits()

    # Normalize
    print("\n  Normalizing pixel values: [0, 255] → [0.0, 1.0]")
    x_train = normalize(x_train)
    x_val   = normalize(x_val)
    x_test  = normalize(x_test)
    print(f"  Done. Sample pixel range: [{x_train.min():.1f}, {x_train.max():.1f}]")

    # Visualize augmentation
    visualize_augmentation(x_train, y_train)

    # Build pipelines
    print("\n  Building tf.data pipelines ...")
    train_ds = build_dataset(x_train, y_train, BATCH_SIZE, augment=True,  shuffle=True)
    val_ds   = build_dataset(x_val,   y_val,   BATCH_SIZE, augment=False, shuffle=False)
    test_ds  = build_dataset(x_test,  y_test,  BATCH_SIZE, augment=False, shuffle=False)

    # Summary
    print_pipeline_summary(train_ds, val_ds, test_ds)

    # Save normalized data for next phases (so they don't have to redo this)
    save_path = PROJECT_ROOT / "data" / "normalized_splits.npz"
    np.savez_compressed(
        save_path,
        x_train=x_train, y_train=y_train,
        x_val=x_val,     y_val=y_val,
        x_test=x_test,   y_test=y_test,
    )
    size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved normalized splits to: {save_path}  ({size_mb:.1f} MB)")

    print("\n✅ Phase 3 complete!")
    print("   Data is normalized, augmentation pipeline is built.")
    print("   👉 Next step: Phase 4 — Model Architecture\n")


if __name__ == "__main__":
    main()
