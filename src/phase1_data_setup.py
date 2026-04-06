"""
Phase 1: Data Setup & Exploration
===================================
Cat Detection Project — TensorFlow/Keras

TWO MODES:
  1. On your own machine: Set USE_SYNTHETIC = False
     → Downloads CIFAR-10 automatically via TensorFlow.
  2. In a sandbox / no internet: Set USE_SYNTHETIC = True  (default here)
     → Generates random dummy images so the full pipeline runs.

CIFAR-10 classes (for reference):
  0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
  5: dog, 6: frog, 7: horse, 8: ship, 9: truck

We treat class 3 (cat) as "cat" and everything else as "not_cat".

Usage:
    python src/phase1_data_setup.py
"""

import os
import numpy as np
from pathlib import Path
from collections import Counter

# --- Optional viz ---
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠️  matplotlib not installed. Visualization will be skipped.")
    print("   Install with: pip install matplotlib")


# ==============================================================
# CONFIG
# ==============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAT_CLASS_INDEX = 3   # CIFAR-10 label for "cat"
RANDOM_SEED = 42

# ⬇️ CHANGE THIS to False when running on your own machine with internet
USE_SYNTHETIC = True


# ==============================================================
# STEP 1 — Load data and create binary labels
# ==============================================================
def load_data_cifar10():
    """Download real CIFAR-10 data (requires internet + TensorFlow)."""
    import tensorflow as tf
    from tensorflow.keras.datasets import cifar10

    print("  Downloading / loading CIFAR-10 ...")
    (x_train, y_train_raw), (x_test, y_test_raw) = cifar10.load_data()

    y_train_raw = y_train_raw.flatten()
    y_test_raw  = y_test_raw.flatten()

    y_train = (y_train_raw == CAT_CLASS_INDEX).astype(np.int32)
    y_test  = (y_test_raw  == CAT_CLASS_INDEX).astype(np.int32)

    print(f"  Loaded {len(x_train)} training + {len(x_test)} test images.")
    return x_train, y_train, x_test, y_test


def load_data_synthetic(n_train=5000, n_test=1000, cat_ratio=0.10):
    """
    Generate synthetic random images that mimic CIFAR-10 shape (32x32x3).
    ~10% are labeled 'cat' (1), the rest 'not_cat' (0).
    The images are random noise — this is purely for pipeline testing.
    """
    print("  Generating synthetic data (random 32x32 images) ...")
    print("  ⚠️  These are fake images! Switch USE_SYNTHETIC = False for real data.\n")

    np.random.seed(RANDOM_SEED)

    # Generate random images (uint8, 0-255)
    x_train = np.random.randint(0, 256, (n_train, 32, 32, 3), dtype=np.uint8)
    x_test  = np.random.randint(0, 256, (n_test,  32, 32, 3), dtype=np.uint8)

    # Make "cat" images slightly reddish and "not_cat" slightly bluish
    # so visualizations look meaningfully different
    n_cat_train = int(n_train * cat_ratio)
    n_cat_test  = int(n_test  * cat_ratio)

    # Cat images: boost red channel
    x_train[:n_cat_train, :, :, 0] = np.clip(
        x_train[:n_cat_train, :, :, 0].astype(np.int16) + 80, 0, 255
    ).astype(np.uint8)
    x_test[:n_cat_test, :, :, 0] = np.clip(
        x_test[:n_cat_test, :, :, 0].astype(np.int16) + 80, 0, 255
    ).astype(np.uint8)

    # Not-cat images: boost blue channel
    x_train[n_cat_train:, :, :, 2] = np.clip(
        x_train[n_cat_train:, :, :, 2].astype(np.int16) + 80, 0, 255
    ).astype(np.uint8)
    x_test[n_cat_test:, :, :, 2] = np.clip(
        x_test[n_cat_test:, :, :, 2].astype(np.int16) + 80, 0, 255
    ).astype(np.uint8)

    # Labels
    y_train = np.zeros(n_train, dtype=np.int32)
    y_train[:n_cat_train] = 1
    y_test = np.zeros(n_test, dtype=np.int32)
    y_test[:n_cat_test] = 1

    # Shuffle so cats aren't all at the front
    train_idx = np.random.permutation(n_train)
    test_idx  = np.random.permutation(n_test)
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    x_test,  y_test  = x_test[test_idx],   y_test[test_idx]

    print(f"  Generated {n_train} training + {n_test} test images.")
    return x_train, y_train, x_test, y_test


def load_data():
    """Pick real or synthetic data based on USE_SYNTHETIC flag."""
    if USE_SYNTHETIC:
        return load_data_synthetic()
    else:
        return load_data_cifar10()


# ==============================================================
# STEP 2 — Explore the data
# ==============================================================
def explore(x_train, y_train, x_test, y_test):
    """Print basic stats about the dataset."""
    print("\n" + "=" * 55)
    print("  DATA EXPLORATION REPORT")
    print("=" * 55)
    print(f"  Image shape    : {x_train[0].shape}  (32x32 pixels, 3 color channels)")
    print(f"  Pixel range    : [{x_train.min()}, {x_train.max()}]")
    print(f"  Data type      : {x_train.dtype}")
    print()

    for name, y in [("Train", y_train), ("Test", y_test)]:
        counts = Counter(y.tolist())
        total = len(y)
        n_cat = counts.get(1, 0)
        n_not = counts.get(0, 0)
        print(f"  {name:>5} set : {total:>6} images")
        print(f"         cat     : {n_cat:>6}  ({n_cat/total*100:.1f}%)")
        print(f"         not_cat : {n_not:>6}  ({n_not/total*100:.1f}%)")
        print()

    print("  Note: cats are only ~10% of the data, so the dataset is")
    print("  imbalanced. We'll handle this during training (Phase 5).")
    print("=" * 55)


# ==============================================================
# STEP 3 — Split training data into train + validation
# ==============================================================
def split_train_val(x_train, y_train, val_ratio=0.15):
    """
    The dataset already provides a separate test set, so we only need
    to carve a validation set out of the training data.
    """
    np.random.seed(RANDOM_SEED)
    n = len(x_train)
    indices = np.random.permutation(n)
    n_val = int(n * val_ratio)

    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    x_val,  y_val  = x_train[val_idx],  y_train[val_idx]
    x_tr,   y_tr   = x_train[train_idx], y_train[train_idx]

    print("\n" + "=" * 55)
    print("  DATA SPLIT SUMMARY")
    print("=" * 55)
    for name, y in [("Train", y_tr), ("Validation", y_val)]:
        n_cat = int((y == 1).sum())
        n_not = int((y == 0).sum())
        print(f"  {name:>10} : {len(y):>6} images  (cat: {n_cat}, not_cat: {n_not})")
    print(f"  {'Test':>10} : (separate test set — untouched until Phase 9)")
    print("=" * 55)

    return x_tr, y_tr, x_val, y_val


# ==============================================================
# STEP 4 — Visualize sample images
# ==============================================================
def show_samples(x_train, y_train, n=10):
    """Show a grid of sample images, half cats and half not-cats."""
    if not HAS_MPL:
        print("  Skipping visualization (matplotlib not installed).")
        return

    cat_indices = np.where(y_train == 1)[0]
    not_indices = np.where(y_train == 0)[0]

    np.random.seed(RANDOM_SEED)
    chosen_cats = np.random.choice(cat_indices, min(n // 2, len(cat_indices)), replace=False)
    chosen_nots = np.random.choice(not_indices, min(n // 2, len(not_indices)), replace=False)
    chosen = np.concatenate([chosen_cats, chosen_nots])
    np.random.shuffle(chosen)

    fig, axes = plt.subplots(2, len(chosen) // 2, figsize=(14, 6))
    axes = axes.flatten()

    for ax, idx in zip(axes, chosen):
        ax.imshow(x_train[idx])
        label = "CAT" if y_train[idx] == 1 else "not cat"
        color = "green" if y_train[idx] == 1 else "red"
        ax.set_title(label, fontsize=12, color=color, fontweight="bold")
        ax.axis("off")

    title = "Sample Images"
    if USE_SYNTHETIC:
        title += " (SYNTHETIC — reddish = cat, bluish = not cat)"
    else:
        title += " from CIFAR-10 (Cat vs Not-Cat)"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    save_path = PROJECT_ROOT / "sample_images.png"
    plt.savefig(save_path, dpi=120)
    print(f"\n  Sample grid saved to: {save_path}")
    plt.close()


# ==============================================================
# STEP 5 — Save processed data as .npz for easy loading later
# ==============================================================
def save_splits(x_train, y_train, x_val, y_val, x_test, y_test):
    """Save all splits to a single .npz file for fast reloading."""
    save_path = PROJECT_ROOT / "data" / "processed_splits.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        x_train=x_train, y_train=y_train,
        x_val=x_val,     y_val=y_val,
        x_test=x_test,   y_test=y_test,
    )
    size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved processed splits to: {save_path}  ({size_mb:.1f} MB)")
    print(f"     Load later with: data = np.load('{save_path}')")
    print(f"     Then: x_train = data['x_train'], y_train = data['y_train'], etc.")


# ==============================================================
# MAIN
# ==============================================================
def main():
    mode = "Synthetic Data" if USE_SYNTHETIC else "CIFAR-10"
    print(f"\n🐱 Cat Detection — Phase 1: Data Setup ({mode})\n")

    # Load
    x_train, y_train, x_test, y_test = load_data()

    # Explore
    explore(x_train, y_train, x_test, y_test)

    # Visualize
    show_samples(x_train, y_train)

    # Split train → train + val
    x_train, y_train, x_val, y_val = split_train_val(x_train, y_train)

    # Save
    save_splits(x_train, y_train, x_val, y_val, x_test, y_test)

    print("\n✅ Phase 1 complete!")
    print("   Your data is explored, split, and saved.")
    if USE_SYNTHETIC:
        print("   💡 Tip: Set USE_SYNTHETIC = False to use real CIFAR-10 data on your machine.")
    print("   👉 Next step: Phase 3 — Preprocessing & Data Augmentation\n")


if __name__ == "__main__":
    main()
