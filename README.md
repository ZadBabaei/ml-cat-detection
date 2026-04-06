# Cat Detector — ML Learning Project

A binary image classification project that detects whether a picture contains a cat or not. Built as a hands-on learning exercise covering the full machine learning lifecycle.

## Project Structure

```
cat_detector/
├── README.md                          ← you are here
├── plan.md                            ← full 10-phase project plan
├── sample_images.png                  ← sample data visualization
├── augmentation_demo.png              ← data augmentation examples
├── training_curves_default.png        ← loss/accuracy curves (default HP)
├── training_curves_best_tuned.png     ← loss/accuracy curves (tuned HP)
├── confusion_matrix_test.png          ← final test confusion matrix
├── roc_curve_test.png                 ← final test ROC curve
├── data/
│   ├── processed_splits.npz           ← raw train/val/test splits
│   └── normalized_splits.npz          ← normalized (0-1) splits
├── models/
│   └── final_best_model.keras         ← best trained model
└── src/
    ├── phase1_data_setup.py           ← data loading, EDA, splitting
    ├── phase3_preprocessing.py        ← normalization & augmentation
    ├── phase4_model.py                ← model architectures (baseline, CNN, transfer)
    ├── phase5_6_7_train_tune.py       ← training, overfitting diagnosis, HP tuning
    └── phase8_9_evaluate.py           ← metrics, confusion matrix, ROC, final test
```

## How to Run

Each script is self-contained and runs in order:

```bash
# 1. Set up data (downloads CIFAR-10 or generates synthetic data)
python src/phase1_data_setup.py

# 2. Preprocess & build augmentation pipeline
python src/phase3_preprocessing.py

# 3. View model architectures
python src/phase4_model.py

# 4. Train, diagnose overfitting, and tune hyperparameters
python src/phase5_6_7_train_tune.py

# 5. Evaluate with all metrics and run final test
python src/phase8_9_evaluate.py
```

## Requirements

```
tensorflow>=2.15
numpy
matplotlib
```

Install with: `pip install tensorflow numpy matplotlib`

## Switching to Real Data

The project uses synthetic data by default (for environments without internet). To use real CIFAR-10:

1. Open `src/phase1_data_setup.py`
2. Change `USE_SYNTHETIC = True` to `USE_SYNTHETIC = False`
3. Re-run all scripts

For even better results, increase `DEFAULT_EPOCHS` to 30-50 in `phase5_6_7_train_tune.py` and set `MAX_TRAIN_SAMPLES = None` to use all training data.

## Key Concepts Covered

| Phase | Concept | What You Learn |
|-------|---------|----------------|
| 1 | Data setup & EDA | Exploring data, checking class balance, cleaning |
| 2 | Train/Val/Test split | Why three splits, preventing data leakage |
| 3 | Preprocessing | Normalization (why 0-1), data augmentation (why & how) |
| 4 | Model architecture | CNNs (Conv2D, MaxPool, Dense), transfer learning |
| 5 | Training & Backpropagation | Forward pass, loss, chain rule, gradient descent |
| 6 | Overfitting vs Underfitting | Reading training curves, diagnosis, regularization |
| 7 | Hyperparameter tuning | Grid search, random search, validation-based selection |
| 8 | Evaluation metrics | Accuracy (and why it's misleading!), precision, recall, F1, ROC/AUC |
| 9 | Final testing | One-time test evaluation, threshold tuning |

## Final Results (Synthetic Data Demo)

| Metric | Value |
|--------|-------|
| Accuracy | 99.4% |
| Precision | 94.3% |
| Recall | 100% |
| F1 Score | 97.1% |
| ROC AUC | 0.997 |

Key insight: with the default threshold of 0.5, the model appeared to have 90% accuracy but **0% recall** — it never actually predicted "cat." Tuning the threshold to 0.07 unlocked the model's true performance. This demonstrates why accuracy alone is dangerous with imbalanced data.

- <!-- add-to-portfolio -->
