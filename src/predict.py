"""
predict.py — Run the trained cat detector on any image
=======================================================

Usage:
    py src/predict.py path/to/your/image.jpg

Examples:
    py src/predict.py C:/Users/Mehrzad/Pictures/myphoto.jpg
    py src/predict.py images/cat1.png
    py src/predict.py images/dog.jpg

The script will print whether the image contains a cat and
how confident the model is. It also shows a small preview.

IMPORTANT: You must have run phase5_6_7_train_tune.py first
           so that models/final_best_model.keras exists.
           If you haven't trained yet, run:
               py src/phase1_data_setup.py
               py src/phase3_preprocessing.py
               py src/phase5_6_7_train_tune.py
"""

import sys
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide TensorFlow info messages

# ── Validate argument ──────────────────────────────────────────
if len(sys.argv) < 2:
    print("\nUsage:  py src/predict.py <path_to_image>")
    print("Example: py src/predict.py C:/Pictures/cat.jpg\n")
    sys.exit(1)

IMAGE_PATH = Path(sys.argv[1])
if not IMAGE_PATH.exists():
    print(f"\nError: File not found — {IMAGE_PATH}\n")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_best_model.keras"

if not MODEL_PATH.exists():
    print(f"\nNo trained model found at: {MODEL_PATH}")
    print("Please run the training script first:")
    print("  py src/phase5_6_7_train_tune.py\n")
    sys.exit(1)

# ── Imports ────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Tip: install Pillow for better image support: pip install Pillow")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Threshold (must match what was chosen in phase8_9_evaluate.py)
# The model outputs probabilities. With CIFAR-10's class imbalance
# the best threshold is often much lower than 0.5.
# Default here is 0.5; adjust if your training found a better value.
THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load and preprocess the image
# ─────────────────────────────────────────────────────────────
def load_and_preprocess(image_path: Path):
    """
    Load any image file and prepare it for the model.

    The model was trained on 32×32 RGB images with pixel values
    in [0.0, 1.0], so we must apply the same transforms here.
    This is called 'inference preprocessing' and it must exactly
    match the training preprocessing — otherwise the model sees
    inputs it was never trained on and gives bad results.
    """
    if HAS_PIL:
        img = Image.open(image_path).convert("RGB")   # ensure 3 channels
        img_resized = img.resize((32, 32), Image.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)
    else:
        # Fallback: use TensorFlow directly
        raw = tf.io.read_file(str(image_path))
        img_tensor = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img_resized = tf.image.resize(img_tensor, [32, 32])
        img_array = img_resized.numpy().astype(np.float32)
        img = None
        img_resized = None

    # Normalize to [0, 1] — same as training
    img_normalized = img_array / 255.0

    # Add batch dimension: (32, 32, 3) → (1, 32, 32, 3)
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch, img_array, img


# ─────────────────────────────────────────────────────────────
# STEP 2 — Run prediction
# ─────────────────────────────────────────────────────────────
def predict(model, img_batch, threshold=THRESHOLD):
    """
    Forward pass: run the image through the model.
    Returns the probability (0.0–1.0) of the image being a cat.
    """
    probability = model.predict(img_batch, verbose=0)[0][0]
    is_cat = probability >= threshold
    return float(probability), bool(is_cat)


# ─────────────────────────────────────────────────────────────
# STEP 3 — Show result
# ─────────────────────────────────────────────────────────────
def print_result(image_path, probability, is_cat, threshold):
    """Print the prediction in a clear, readable format."""
    bar_filled = int(probability * 30)
    bar = "█" * bar_filled + "░" * (30 - bar_filled)

    print()
    print("╔══════════════════════════════════════════╗")
    print("║         CAT DETECTION RESULT             ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Image   : {str(image_path.name)[:40]:<40}  ║"[:46] + "║")
    print(f"║  Result  : {'🐱 CAT DETECTED' if is_cat else '❌ NOT A CAT':<40}  ║"[:46] + "║")
    print(f"║  Cat prob: {probability*100:>5.1f}%  [{bar}]  ║")
    print(f"║  Threshold used: {threshold}                         ║"[:46] + "║")
    print("╚══════════════════════════════════════════╝")
    print()

    if is_cat:
        conf = probability * 100
        if conf > 85:
            print(f"  High confidence ({conf:.1f}%) — very likely a cat!")
        elif conf > 65:
            print(f"  Moderate confidence ({conf:.1f}%) — probably a cat.")
        else:
            print(f"  Low confidence ({conf:.1f}%) — uncertain, might be a cat.")
    else:
        not_cat_pct = (1 - probability) * 100
        if not_cat_pct > 85:
            print(f"  High confidence ({not_cat_pct:.1f}%) — does not contain a cat.")
        else:
            print(f"  Low confidence ({not_cat_pct:.1f}%) — uncertain, probably not a cat.")

    print()


def save_result_image(image_path, img_array, probability, is_cat):
    """Save a visualization of the result to a PNG file."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.clip(img_array / 255.0, 0, 1))

    label = f"{'CAT' if is_cat else 'NOT CAT'}  ({probability*100:.1f}%)"
    color = "green" if is_cat else "red"
    ax.set_title(label, fontsize=16, color=color, fontweight="bold", pad=12)
    ax.axis("off")

    output_path = PROJECT_ROOT / f"prediction_{image_path.stem}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Result image saved to: {output_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print(f"\n  Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)

    print(f"  Processing image : {IMAGE_PATH}")
    img_batch, img_array, _ = load_and_preprocess(IMAGE_PATH)

    probability, is_cat = predict(model, img_batch, threshold=THRESHOLD)

    print_result(IMAGE_PATH, probability, is_cat, THRESHOLD)

    save_result_image(IMAGE_PATH, img_array, probability, is_cat)


if __name__ == "__main__":
    main()
