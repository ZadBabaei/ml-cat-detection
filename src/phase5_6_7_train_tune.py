"""
Phases 5 + 6 + 7: Training, Overfitting Diagnosis, & Hyperparameter Tuning
============================================================================
Cat Detection Project — TensorFlow/Keras

This script covers three phases in one:

  PHASE 5 — Training & Backpropagation
    • Trains the CNN on the cat/not-cat data
    • Explains what happens each epoch (forward pass, loss, backprop, update)
    • Uses callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

  PHASE 6 — Overfitting vs. Underfitting Diagnosis
    • Plots training vs. validation loss/accuracy curves
    • Automatically detects overfitting or underfitting from the curves
    • Suggests fixes based on the diagnosis

  PHASE 7 — Hyperparameter Tuning
    • Runs a grid/random search over key hyperparameters
    • Compares results and picks the best configuration
    • Retrains the final model with the best hyperparameters

Usage:
    python src/phase5_6_7_train_tune.py

TIP: On your own machine with real CIFAR-10 data, increase EPOCHS and
     the hyperparameter grid for better results.
"""

import os
import time
import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
DATA_PATH = PROJECT_ROOT / "data" / "normalized_splits.npz"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Training defaults (Phase 5)
DEFAULT_EPOCHS = 30        # 30 epochs for real data; reduce to 8 for quick testing
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001

# Set to None for full dataset, or a number to limit for quick testing
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

# Hyperparameter grid (Phase 7)
HP_GRID = {
    "learning_rate": [0.01, 0.001, 0.0001],
    "batch_size": [32, 64],
    "dropout_rate": [0.3, 0.5],
}
# For speed we'll do random search: try N random combos instead of all
HP_MAX_TRIALS = 6  # increase for more thorough search


# ==============================================================
# DATA LOADING
# ==============================================================
def load_data():
    """Load normalized splits from Phase 3."""
    print(f"  Loading data from: {DATA_PATH}")
    data = np.load(DATA_PATH)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val,   y_val   = data["x_val"],   data["y_val"]
    x_test,  y_test  = data["x_test"],  data["y_test"]

    # Subsample for speed in sandbox (remove this on your machine)
    if MAX_TRAIN_SAMPLES and len(x_train) > MAX_TRAIN_SAMPLES:
        idx = np.random.RandomState(42).permutation(len(x_train))[:MAX_TRAIN_SAMPLES]
        x_train, y_train = x_train[idx], y_train[idx]
    if MAX_VAL_SAMPLES and len(x_val) > MAX_VAL_SAMPLES:
        idx = np.random.RandomState(42).permutation(len(x_val))[:MAX_VAL_SAMPLES]
        x_val, y_val = x_val[idx], y_val[idx]

    print(f"  Using: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
    return (x_train, y_train, x_val, y_val, x_test, y_test)


def make_dataset(x, y, batch_size, augment=False, shuffle=False):
    """Build a tf.data pipeline (same as Phase 3)."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=42)
    ds = ds.batch(batch_size)
    if augment:
        aug = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom((-0.1, 0.1)),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ])
        ds = ds.map(
            lambda img, lbl: (tf.clip_by_value(aug(img, training=True), 0.0, 1.0), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ==============================================================
# MODEL BUILDER (parameterized for hyperparameter tuning)
# ==============================================================
def build_cnn(learning_rate=0.001, dropout_rate=0.5):
    """
    Build and compile the CNN with configurable hyperparameters.
    This is the same architecture from Phase 4, but with tunable
    dropout and learning rate.
    """
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),  # lighter dropout in conv layers
        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        # Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation="sigmoid"),
    ], name="cat_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ================================================================
# PHASE 5 — TRAINING & BACKPROPAGATION
# ================================================================
def compute_class_weights(y):
    """
    Compute class weights to handle imbalanced data.

    With 90% not-cat and 10% cat, the model learns to always predict
    "not cat" because that's correct 90% of the time. Class weights
    fix this by telling the optimizer:
      "A mistake on a cat image costs 9× more than a mistake on a non-cat."

    Formula: weight_for_class = total_samples / (num_classes × count_of_class)
      not_cat (0): 50000 / (2 × 45000) = 0.556
      cat (1):     50000 / (2 × 5000)  = 5.0

    This forces the model to pay attention to the minority class (cats).
    """
    n_total = len(y)
    n_cat = int((y == 1).sum())
    n_not = n_total - n_cat
    weight_not = n_total / (2.0 * n_not)
    weight_cat = n_total / (2.0 * n_cat)
    weights = {0: weight_not, 1: weight_cat}
    print(f"  Class weights: not_cat={weight_not:.3f}, cat={weight_cat:.3f}")
    return weights


def train_model(
    model, train_ds, val_ds,
    epochs=DEFAULT_EPOCHS,
    model_name="cat_cnn",
    class_weight=None,
):
    """
    Train the model and return the training history.

    WHAT HAPPENS EACH EPOCH (simplified):
    ┌──────────────────────────────────────────────────────┐
    │  For each batch of images:                           │
    │    1. FORWARD PASS                                   │
    │       Image → Conv layers → Dense layers → P(cat)    │
    │                                                      │
    │    2. LOSS CALCULATION                               │
    │       loss = BinaryCrossentropy(true_label, P(cat))  │
    │       e.g. true=1 (cat), predicted=0.3 → high loss   │
    │            true=1 (cat), predicted=0.9 → low loss    │
    │                                                      │
    │    3. BACKPROPAGATION                                │
    │       Compute ∂loss/∂weight for EVERY weight in the  │
    │       network, using the chain rule, layer by layer   │
    │       from output back to input.                     │
    │                                                      │
    │       Chain rule example (simplified):               │
    │       ∂loss/∂w1 = ∂loss/∂output × ∂output/∂hidden   │
    │                   × ∂hidden/∂w1                      │
    │                                                      │
    │    4. WEIGHT UPDATE (Adam optimizer)                  │
    │       w_new = w_old - learning_rate × gradient       │
    │       (Adam also uses momentum and adaptive rates)   │
    │                                                      │
    │  After all batches → 1 epoch complete.               │
    │  Evaluate on validation set (no weight updates).     │
    └──────────────────────────────────────────────────────┘

    CALLBACKS EXPLAINED:
    """
    checkpoint_path = MODELS_DIR / f"{model_name}_best.keras"

    callbacks = [
        # EarlyStopping:
        #   Monitors val_loss. If it doesn't improve for 'patience' epochs,
        #   stop training. This prevents wasting time and overfitting.
        #   restore_best_weights=True: go back to the best epoch's weights.
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        # ModelCheckpoint:
        #   Save the model every time val_loss improves.
        #   So even if training crashes, we have the best model saved.
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),

        # ReduceLROnPlateau:
        #   If val_loss plateaus for 3 epochs, reduce the learning rate
        #   by half. This helps "fine-tune" when the model is close to
        #   a good solution but the learning rate is too large to converge.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,      # new_lr = old_lr × 0.5
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n  Training '{model_name}' for up to {epochs} epochs ...")
    print(f"  (EarlyStopping will stop if val_loss doesn't improve for 5 epochs)\n")

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    elapsed = time.time() - start

    print(f"\n  Training finished in {elapsed:.1f}s ({len(history.history['loss'])} epochs)")
    print(f"  Best model saved to: {checkpoint_path}")

    return history


# ================================================================
# PHASE 6 — OVERFITTING vs. UNDERFITTING DIAGNOSIS
# ================================================================
def diagnose_fit(history, model_name="cat_cnn"):
    """
    Plot training curves and diagnose overfitting or underfitting.

    HOW TO READ THE CURVES:
    ─────────────────────────────────────────────────────
    UNDERFITTING (high bias):
      • Both train_loss and val_loss are HIGH
      • Both train_acc and val_acc are LOW
      • The model is too simple to learn the patterns
      FIX: Bigger model, more epochs, less regularization

    OVERFITTING (high variance):
      • train_loss keeps going DOWN
      • val_loss starts going UP (diverges from train_loss)
      • train_acc is HIGH but val_acc is much LOWER
      • The model memorized training data instead of generalizing
      FIX: More data/augmentation, more dropout, early stopping,
           simpler model, L2 regularization

    GOOD FIT:
      • Both train_loss and val_loss go down together
      • Small gap between train_acc and val_acc
      • val_loss levels off but doesn't go back up
    ─────────────────────────────────────────────────────
    """
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    # --- Detect overfitting / underfitting ---
    final_train_loss = h["loss"][-1]
    final_val_loss = h["val_loss"][-1]
    final_train_acc = h["accuracy"][-1]
    final_val_acc = h["val_accuracy"][-1]
    gap = final_train_acc - final_val_acc

    print("\n" + "=" * 55)
    print("  PHASE 6 — FIT DIAGNOSIS")
    print("=" * 55)
    print(f"  Final train loss : {final_train_loss:.4f}   accuracy: {final_train_acc:.4f}")
    print(f"  Final val loss   : {final_val_loss:.4f}   accuracy: {final_val_acc:.4f}")
    print(f"  Accuracy gap     : {gap:.4f}  (train - val)")
    print()

    if final_train_acc < 0.7 and final_val_acc < 0.7:
        diagnosis = "UNDERFITTING"
        print("  Diagnosis: ⚠️  UNDERFITTING (both accuracies are low)")
        print("  Suggested fixes:")
        print("    • Use a bigger model (more layers or filters)")
        print("    • Train for more epochs")
        print("    • Reduce dropout / regularization")
        print("    • Check that data preprocessing is correct")
    elif gap > 0.10:
        diagnosis = "OVERFITTING"
        print("  Diagnosis: ⚠️  OVERFITTING (train >> val accuracy)")
        print("  Suggested fixes:")
        print("    • Add more training data or stronger augmentation")
        print("    • Increase dropout rate")
        print("    • Add L2 regularization (weight_decay in optimizer)")
        print("    • Use early stopping (already enabled)")
        print("    • Try a simpler model")
    else:
        diagnosis = "GOOD FIT"
        print("  Diagnosis: ✅ GOOD FIT (train and val are close)")
        print("  The model is generalizing well!")

    print("=" * 55)

    # --- Plot curves ---
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        ax1.plot(epochs, h["loss"], "b-o", label="Train Loss", markersize=4)
        ax1.plot(epochs, h["val_loss"], "r-o", label="Val Loss", markersize=4)
        ax1.set_title("Loss Curve", fontsize=13)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Binary Cross-Entropy Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(epochs, h["accuracy"], "b-o", label="Train Accuracy", markersize=4)
        ax2.plot(epochs, h["val_accuracy"], "r-o", label="Val Accuracy", markersize=4)
        ax2.set_title("Accuracy Curve", fontsize=13)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Training Curves — {model_name}  [{diagnosis}]", fontsize=14)
        plt.tight_layout()

        save_path = PROJECT_ROOT / f"training_curves_{model_name}.png"
        plt.savefig(save_path, dpi=120)
        print(f"\n  Training curves saved to: {save_path}")
        plt.close()

    return diagnosis


# ================================================================
# PHASE 7 — HYPERPARAMETER TUNING
# ================================================================
def hyperparameter_search(x_train, y_train, x_val, y_val, class_weight=None):
    """
    Try different hyperparameter combinations and find the best one.

    HYPERPARAMETERS vs. PARAMETERS:
    ─────────────────────────────────────────────────────
    Parameters (weights):
      • Learned automatically during training via backpropagation
      • e.g. Conv2D filter weights, Dense layer weights and biases

    Hyperparameters:
      • Set BY US before training starts
      • NOT learned from data
      • e.g. learning rate, batch size, dropout rate, # of layers
    ─────────────────────────────────────────────────────

    SEARCH STRATEGIES:
    1. Grid Search: try every combination. Thorough but slow.
       With 3 learning rates × 3 batch sizes × 2 dropout rates = 18 trials.

    2. Random Search: pick N random combos. Often more efficient because
       some hyperparameters matter more than others, and random search
       explores the important dimensions better.

    3. Bayesian Optimization: use past results to intelligently pick
       the next combo. Best for expensive searches. (Optuna, Ray Tune)

    We use RANDOM SEARCH here for speed.
    """
    print("\n" + "=" * 60)
    print("  PHASE 7 — HYPERPARAMETER TUNING (Random Search)")
    print("=" * 60)

    # Generate all possible combos, then randomly sample
    all_combos = list(itertools.product(
        HP_GRID["learning_rate"],
        HP_GRID["batch_size"],
        HP_GRID["dropout_rate"],
    ))
    np.random.seed(42)
    np.random.shuffle(all_combos)
    trials = all_combos[:HP_MAX_TRIALS]

    print(f"  Total possible combinations: {len(all_combos)}")
    print(f"  Trying {len(trials)} random trials\n")

    results = []
    TUNE_EPOCHS = 5  # fewer epochs per trial for speed

    for i, (lr, bs, dropout) in enumerate(trials):
        print(f"  Trial {i+1}/{len(trials)}: lr={lr}, batch_size={bs}, dropout={dropout}")

        # Build model with these hyperparameters
        model = build_cnn(learning_rate=lr, dropout_rate=dropout)

        # Build datasets with this batch size
        train_ds = make_dataset(x_train, y_train, bs, augment=True, shuffle=True)
        val_ds   = make_dataset(x_val, y_val, bs)

        # Train (quietly)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=TUNE_EPOCHS,
            verbose=0,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3,
                    restore_best_weights=True, verbose=0,
                ),
            ],
        )

        # Record best validation loss and accuracy
        best_val_loss = min(history.history["val_loss"])
        best_val_acc  = max(history.history["val_accuracy"])
        actual_epochs = len(history.history["loss"])

        results.append({
            "learning_rate": lr,
            "batch_size": bs,
            "dropout_rate": dropout,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "epochs_run": actual_epochs,
        })

        print(f"    → val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f} ({actual_epochs} epochs)")

    # Sort by best validation loss (lower is better)
    results.sort(key=lambda r: r["best_val_loss"])

    print("\n" + "-" * 60)
    print("  RESULTS (sorted by best val_loss)")
    print("-" * 60)
    print(f"  {'Rank':>4}  {'LR':>8}  {'Batch':>5}  {'Dropout':>7}  {'Val Loss':>9}  {'Val Acc':>8}")
    for i, r in enumerate(results):
        marker = " ⭐" if i == 0 else ""
        print(f"  {i+1:>4}  {r['learning_rate']:>8.4f}  {r['batch_size']:>5}  "
              f"{r['dropout_rate']:>7.2f}  {r['best_val_loss']:>9.4f}  "
              f"{r['best_val_acc']:>7.4f}{marker}")

    best = results[0]
    print(f"\n  Best hyperparameters:")
    print(f"    Learning rate : {best['learning_rate']}")
    print(f"    Batch size    : {best['batch_size']}")
    print(f"    Dropout rate  : {best['dropout_rate']}")
    print("=" * 60)

    return best


# ================================================================
# MAIN — RUN ALL THREE PHASES
# ================================================================
def main():
    print("\n" + "=" * 60)
    print("  🐱 Cat Detection — Phases 5 + 6 + 7")
    print("=" * 60)

    # --- Load data ---
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    # --- Compute class weights to handle 90/10 imbalance ---
    cw = compute_class_weights(y_train)

    # ===========================================
    # PHASE 5 — Initial training with defaults
    # ===========================================
    print("\n" + "~" * 60)
    print("  PHASE 5 — TRAINING WITH DEFAULT HYPERPARAMETERS")
    print("~" * 60)

    train_ds = make_dataset(x_train, y_train, DEFAULT_BATCH_SIZE, augment=True, shuffle=True)
    val_ds   = make_dataset(x_val, y_val, DEFAULT_BATCH_SIZE)

    model = build_cnn(
        learning_rate=DEFAULT_LEARNING_RATE,
        dropout_rate=0.5,
    )

    history = train_model(model, train_ds, val_ds, epochs=DEFAULT_EPOCHS, model_name="default", class_weight=cw)

    # ===========================================
    # PHASE 6 — Diagnose overfitting/underfitting
    # ===========================================
    print("\n" + "~" * 60)
    print("  PHASE 6 — OVERFITTING / UNDERFITTING DIAGNOSIS")
    print("~" * 60)

    diagnosis = diagnose_fit(history, model_name="default")

    # ===========================================
    # PHASE 7 — Hyperparameter tuning
    # ===========================================
    print("\n" + "~" * 60)
    print("  PHASE 7 — HYPERPARAMETER TUNING")
    print("~" * 60)

    best_hp = hyperparameter_search(x_train, y_train, x_val, y_val, class_weight=cw)

    # ===========================================
    # FINAL — Retrain with best hyperparameters
    # ===========================================
    print("\n" + "~" * 60)
    print("  FINAL — RETRAINING WITH BEST HYPERPARAMETERS")
    print("~" * 60)

    best_train_ds = make_dataset(
        x_train, y_train, best_hp["batch_size"], augment=True, shuffle=True
    )
    best_val_ds = make_dataset(x_val, y_val, best_hp["batch_size"])

    best_model = build_cnn(
        learning_rate=best_hp["learning_rate"],
        dropout_rate=best_hp["dropout_rate"],
    )

    best_history = train_model(
        best_model, best_train_ds, best_val_ds,
        epochs=DEFAULT_EPOCHS,
        model_name="best_tuned",
        class_weight=cw,
    )

    best_diagnosis = diagnose_fit(best_history, model_name="best_tuned")

    # Save final model
    final_path = MODELS_DIR / "final_best_model.keras"
    best_model.save(final_path)
    print(f"\n  Final model saved to: {final_path}")

    # ===========================================
    # SUMMARY
    # ===========================================
    print("\n" + "=" * 60)
    print("  PHASES 5-6-7 SUMMARY")
    print("=" * 60)
    print(f"  Phase 5: Trained CNN with defaults → {diagnosis}")
    print(f"  Phase 6: Diagnosed fit from training curves")
    print(f"  Phase 7: Tuned hyperparameters ({HP_MAX_TRIALS} trials)")
    print(f"           Best: lr={best_hp['learning_rate']}, "
          f"batch={best_hp['batch_size']}, dropout={best_hp['dropout_rate']}")
    print(f"  Final:   Retrained with best HP → {best_diagnosis}")
    print()
    print("  Saved artifacts:")
    print(f"    • Best model: {final_path}")
    if HAS_MPL:
        print(f"    • Default training curves: {PROJECT_ROOT / 'training_curves_default.png'}")
        print(f"    • Tuned training curves:   {PROJECT_ROOT / 'training_curves_best_tuned.png'}")
    print()
    print("  👉 Next step: Phase 8 — Evaluation Metrics")
    print("=" * 60)


if __name__ == "__main__":
    main()
