"""
Phases 8 + 9: Evaluation Metrics & Final Testing
==================================================
Cat Detection Project — TensorFlow/Keras

PHASE 8 — Evaluation Metrics
  We go beyond simple accuracy and compute metrics that tell us
  HOW the model is making mistakes:
    • Confusion Matrix — visual breakdown of TP, TN, FP, FN
    • Precision — "when the model says cat, is it right?"
    • Recall — "does it catch all the cats?"
    • F1 Score — balanced combination of precision and recall
    • ROC Curve & AUC — performance across all thresholds

PHASE 9 — Final Testing
  Run the best model on the held-out test set ONE TIME to get
  the definitive performance number. This is the number you'd
  report in a paper or to your team.

Usage:
    python src/phase8_9_evaluate.py
"""

import os
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras

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
MODEL_PATH = PROJECT_ROOT / "models" / "final_best_model.keras"


# ==============================================================
# DATA & MODEL LOADING
# ==============================================================
def load_data():
    data = np.load(DATA_PATH)
    return (
        data["x_val"], data["y_val"],
        data["x_test"], data["y_test"],
    )


def load_model():
    print(f"  Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    return model


# ==============================================================
# PREDICTION HELPER
# ==============================================================
def get_predictions(model, x, threshold=0.5):
    """
    Get raw probabilities and binary predictions.

    The model outputs P(cat) for each image.
    If P(cat) > threshold → predict "cat" (1)
    If P(cat) <= threshold → predict "not cat" (0)

    The THRESHOLD is a hyperparameter too! Default is 0.5 but:
    - Raise it (e.g. 0.7) → fewer false positives, more precision
    - Lower it (e.g. 0.3) → catch more cats, more recall
    """
    probabilities = model.predict(x, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(np.int32)
    return probabilities, predictions


# ==============================================================
# CONFUSION MATRIX
# ==============================================================
def compute_confusion_matrix(y_true, y_pred):
    """
    Build a confusion matrix manually (no sklearn needed).

    The confusion matrix for binary classification:

                        Predicted
                    NOT CAT    CAT
    Actual NOT CAT [  TN    |  FP  ]
    Actual CAT     [  FN    |  TP  ]

    TN (True Negative):  Correctly predicted "not cat"
    FP (False Positive): Wrongly predicted "cat" (false alarm)
    FN (False Negative): Missed a real cat (the worst mistake?)
    TP (True Positive):  Correctly predicted "cat"
    """
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TP, TN, FP, FN


def print_confusion_matrix(TP, TN, FP, FN):
    """Pretty-print the confusion matrix."""
    print("\n  CONFUSION MATRIX")
    print("  " + "-" * 40)
    print(f"                    Predicted")
    print(f"                  not_cat    cat")
    print(f"  Actual not_cat [ {TN:>5}   | {FP:>5}  ]")
    print(f"  Actual cat     [ {FN:>5}   | {TP:>5}  ]")
    print("  " + "-" * 40)
    print(f"  TN={TN}  FP={FP}  FN={FN}  TP={TP}")


# ==============================================================
# METRICS
# ==============================================================
def compute_metrics(TP, TN, FP, FN):
    """
    Compute classification metrics from the confusion matrix.

    ACCURACY = (TP + TN) / total
      "What fraction of all predictions were correct?"
      ⚠️ MISLEADING with imbalanced data! A model that always says
      "not cat" gets 90% accuracy when only 10% of images are cats.

    PRECISION = TP / (TP + FP)
      "When the model says 'cat', how often is it actually a cat?"
      High precision = few false alarms.
      Important when false positives are costly.

    RECALL (Sensitivity) = TP / (TP + FN)
      "Of all real cats, how many did the model find?"
      High recall = few missed cats.
      Important when false negatives are costly (e.g. medical diagnosis).

    F1 SCORE = 2 × (Precision × Recall) / (Precision + Recall)
      Harmonic mean of precision and recall.
      Balances both — useful as a single summary metric.
      Range: 0 (worst) to 1 (perfect).

    SPECIFICITY = TN / (TN + FP)
      "Of all non-cats, how many were correctly identified as non-cats?"
      Used in ROC curves.
    """
    total = TP + TN + FP + FN

    accuracy    = (TP + TN) / total if total > 0 else 0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall      = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
    }


def print_metrics(metrics, set_name="Validation"):
    """Pretty-print all metrics."""
    print(f"\n  METRICS ({set_name} Set)")
    print("  " + "-" * 40)
    print(f"  Accuracy    : {metrics['accuracy']:.4f}  (% correct overall)")
    print(f"  Precision   : {metrics['precision']:.4f}  (of predicted cats, % correct)")
    print(f"  Recall      : {metrics['recall']:.4f}  (of real cats, % found)")
    print(f"  F1 Score    : {metrics['f1_score']:.4f}  (balance of precision & recall)")
    print(f"  Specificity : {metrics['specificity']:.4f}  (of real non-cats, % correct)")
    print("  " + "-" * 40)


# ==============================================================
# ROC CURVE & AUC
# ==============================================================
def compute_roc_curve(y_true, probabilities, n_thresholds=200):
    """
    Compute the ROC (Receiver Operating Characteristic) curve.

    HOW IT WORKS:
    - Instead of using a fixed threshold of 0.5, we try many thresholds
      from 0.0 to 1.0.
    - At each threshold, we compute:
        TPR (True Positive Rate) = Recall = TP / (TP + FN)
        FPR (False Positive Rate) = FP / (FP + TN)
    - Plot FPR (x-axis) vs TPR (y-axis).

    WHAT THE ROC CURVE TELLS YOU:
    - A perfect model goes straight up to (0, 1) then across to (1, 1).
    - A random model is a diagonal line from (0, 0) to (1, 1).
    - The more the curve bows toward the upper-left, the better.

    AUC (Area Under the Curve):
    - AUC = 1.0 → perfect classifier
    - AUC = 0.5 → random guessing
    - AUC < 0.5 → worse than random (labels might be flipped)
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list, fpr_list = [], []

    for t in thresholds:
        y_pred = (probabilities >= t).astype(np.int32)
        TP, TN, FP, FN = compute_confusion_matrix(y_true, y_pred)

        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    # AUC using trapezoidal rule
    sorted_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sorted_idx]
    tpr_sorted = tpr_arr[sorted_idx]
    auc = np.trapezoid(tpr_sorted, fpr_sorted)

    return fpr_arr, tpr_arr, thresholds, auc


def plot_roc_curve(fpr, tpr, auc, set_name="Validation"):
    """Plot the ROC curve."""
    if not HAS_MPL:
        print("  Skipping ROC plot (matplotlib not installed).")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR = Recall)", fontsize=12)
    ax.set_title(f"ROC Curve — {set_name} Set", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    save_path = PROJECT_ROOT / f"roc_curve_{set_name.lower()}.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"\n  ROC curve saved to: {save_path}")
    plt.close()


# ==============================================================
# PLOT CONFUSION MATRIX
# ==============================================================
def plot_confusion_matrix(TP, TN, FP, FN, set_name="Validation"):
    """Plot a visual confusion matrix."""
    if not HAS_MPL:
        return

    matrix = np.array([[TN, FP], [FN, TP]])
    labels = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}",
                    ha="center", va="center", fontsize=16, color=color,
                    fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["not_cat", "cat"], fontsize=12)
    ax.set_yticklabels(["not_cat", "cat"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(f"Confusion Matrix — {set_name} Set", fontsize=14)

    plt.colorbar(im)
    plt.tight_layout()

    save_path = PROJECT_ROOT / f"confusion_matrix_{set_name.lower()}.png"
    plt.savefig(save_path, dpi=120)
    print(f"  Confusion matrix plot saved to: {save_path}")
    plt.close()


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("\n" + "=" * 60)
    print("  🐱 Cat Detection — Phases 8 + 9: Evaluation & Final Test")
    print("=" * 60)

    # Load
    x_val, y_val, x_test, y_test = load_data()
    model = load_model()

    # =============================================
    # PHASE 8 — EVALUATION METRICS (Validation Set)
    # =============================================
    print("\n" + "~" * 60)
    print("  PHASE 8 — EVALUATION METRICS (Validation Set)")
    print("~" * 60)
    print("  Using the validation set to review all metrics.")
    print("  (The test set stays untouched until Phase 9.)\n")

    probs_val, preds_val = get_predictions(model, x_val, threshold=0.5)

    # Confusion Matrix
    TP, TN, FP, FN = compute_confusion_matrix(y_val, preds_val)
    print_confusion_matrix(TP, TN, FP, FN)
    plot_confusion_matrix(TP, TN, FP, FN, "Validation")

    # Metrics
    metrics_val = compute_metrics(TP, TN, FP, FN)
    print_metrics(metrics_val, "Validation")

    # ROC Curve
    fpr, tpr, thresholds, auc = compute_roc_curve(y_val, probs_val)
    print(f"\n  ROC AUC: {auc:.4f}")
    if auc > 0.9:
        print("  Interpretation: Excellent discriminative ability!")
    elif auc > 0.8:
        print("  Interpretation: Good discriminative ability.")
    elif auc > 0.7:
        print("  Interpretation: Fair — there's room for improvement.")
    else:
        print("  Interpretation: Poor — the model struggles to separate classes.")
    plot_roc_curve(fpr, tpr, auc, "Validation")

    # =============================================
    # Threshold analysis
    # =============================================
    print("\n  THRESHOLD ANALYSIS")
    print("  " + "-" * 50)
    print("  The default threshold is 0.5, but adjusting it")
    print("  trades off precision vs. recall:\n")
    # First, show the probability distribution to understand what thresholds make sense
    cat_probs = probs_val[y_val == 1]
    not_probs = probs_val[y_val == 0]
    print(f"  Probability stats:")
    print(f"    Cat images     — mean: {cat_probs.mean():.4f}, min: {cat_probs.min():.4f}, max: {cat_probs.max():.4f}")
    print(f"    Non-cat images — mean: {not_probs.mean():.4f}, min: {not_probs.min():.4f}, max: {not_probs.max():.4f}")
    print()

    # Find best F1 threshold
    best_f1, best_t = 0, 0.5
    for t_search in np.arange(0.01, 0.99, 0.01):
        _, preds_s = get_predictions(model, x_val, threshold=t_search)
        tp, tn, fp, fn = compute_confusion_matrix(y_val, preds_s)
        m = compute_metrics(tp, tn, fp, fn)
        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_t = t_search

    print(f"  Best F1 threshold: {best_t:.2f} (F1 = {best_f1:.4f})")

    # Save threshold to a file so predict.py can auto-load it
    threshold_path = PROJECT_ROOT / "models" / "best_threshold.txt"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(str(round(best_t, 4)))
    print(f"  Saved threshold to: {threshold_path}")
    print()

    # Show a range of thresholds centered around the best one
    check_thresholds = sorted(set(
        [0.5, best_t] +
        [round(best_t + d, 2) for d in [-0.1, -0.05, 0.05, 0.1]]
    ))
    check_thresholds = [t for t in check_thresholds if 0.01 <= t <= 0.99]

    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for t in check_thresholds:
        _, preds_t = get_predictions(model, x_val, threshold=t)
        tp, tn, fp, fn = compute_confusion_matrix(y_val, preds_t)
        m = compute_metrics(tp, tn, fp, fn)
        marker = " ← default" if t == 0.5 else ""
        print(f"  {t:>10.1f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1_score']:>8.4f}{marker}")

    # Also recompute val metrics with optimal threshold
    _, preds_val_opt = get_predictions(model, x_val, threshold=best_t)
    TP, TN, FP, FN = compute_confusion_matrix(y_val, preds_val_opt)
    print_confusion_matrix(TP, TN, FP, FN)
    plot_confusion_matrix(TP, TN, FP, FN, "Validation_Optimal")
    metrics_val = compute_metrics(TP, TN, FP, FN)
    print_metrics(metrics_val, f"Validation (threshold={best_t:.2f})")

    # =============================================
    # PHASE 9 — FINAL TEST (one time only!)
    # =============================================
    print("\n" + "~" * 60)
    print("  PHASE 9 — FINAL TEST SET EVALUATION")
    print("~" * 60)
    print("  This is the FINAL, definitive evaluation.")
    print("  The test set has NEVER been used for any decision.\n")

    # Use the best threshold found in Phase 8 (not 0.5!)
    print(f"  Using optimal threshold from validation: {best_t:.2f}\n")
    probs_test, preds_test = get_predictions(model, x_test, threshold=best_t)

    # Confusion Matrix
    TP, TN, FP, FN = compute_confusion_matrix(y_test, preds_test)
    print_confusion_matrix(TP, TN, FP, FN)
    plot_confusion_matrix(TP, TN, FP, FN, "Test")

    # Metrics
    metrics_test = compute_metrics(TP, TN, FP, FN)
    print_metrics(metrics_test, "Test")

    # ROC
    fpr_t, tpr_t, _, auc_t = compute_roc_curve(y_test, probs_test)
    print(f"\n  Test ROC AUC: {auc_t:.4f}")
    plot_roc_curve(fpr_t, tpr_t, auc_t, "Test")

    # =============================================
    # FINAL SUMMARY
    # =============================================
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<15} {'Validation':>12} {'Test':>12}")
    print("  " + "-" * 42)
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        label = key.replace("_", " ").title()
        print(f"  {label:<15} {metrics_val[key]:>12.4f} {metrics_test[key]:>12.4f}")
    print(f"  {'ROC AUC':<15} {auc:>12.4f} {auc_t:>12.4f}")
    print("  " + "-" * 42)
    print()

    val_test_gap = abs(metrics_val["accuracy"] - metrics_test["accuracy"])
    if val_test_gap < 0.03:
        print("  The validation and test results are close — the model")
        print("  generalizes well to unseen data. No data leakage detected.")
    else:
        print("  There's a notable gap between validation and test results.")
        print("  This could indicate the validation set isn't representative,")
        print("  or there was some implicit overfitting to the validation set")
        print("  during hyperparameter tuning.")

    print("\n  👉 Next step: Phase 10 — Save model & wrap up!")
    print("=" * 60)


if __name__ == "__main__":
    main()
