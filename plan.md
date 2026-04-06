# Cat Detection — Classical ML & Deep Learning Project Plan

A simple binary image classification project: **given a picture, detect whether there is a cat in it or not.**

---

## Phase 1: Problem Definition & Data

**Goal:** Understand the problem and prepare the dataset.

- **Problem type:** Binary image classification (cat vs. not-cat).
- **Dataset:** Use a public dataset such as Kaggle's "Dogs vs. Cats" (filter for cat / not-cat) or the CIFAR-10 subset (cat class vs. everything else).
- **Data collection:** Download images, organize into two folders — `cat/` and `not_cat/`.
- **Exploratory Data Analysis (EDA):**
  - Count images per class — check for class imbalance.
  - Visualize a few samples from each class.
  - Check image sizes, formats, and quality.
- **Data cleaning:** Remove corrupt files, duplicates, or mislabeled images.

---

## Phase 2: Data Splitting — Train / Validation / Test

**Goal:** Split data properly so we can train, tune, and evaluate fairly.

- **Training set (~70%):** The model learns from this data.
- **Validation set (~15%):** Used during training to tune hyperparameters and monitor overfitting. The model never learns from this set directly.
- **Test set (~15%):** Touched only once at the very end to report final performance. Keeps our evaluation honest.
- **Why three splits?** Training on all data and testing on the same data gives a misleadingly high score. The validation set lets us make decisions (e.g. when to stop training) without contaminating the test set.

---

## Phase 3: Preprocessing & Data Augmentation

**Goal:** Get images into a consistent format and artificially expand the training data.

- **Resize** all images to a fixed size (e.g. 128×128 or 224×224).
- **Normalize** pixel values to [0, 1] or standardize to zero-mean, unit-variance.
- **Data augmentation** (training set only):
  - Random horizontal flip
  - Random rotation (±15°)
  - Random crop / zoom
  - Brightness & contrast jitter
- Augmentation helps the model generalize and reduces overfitting.

---

## Phase 4: Model Selection & Architecture

**Goal:** Choose a model architecture.

### Option A — Classical ML (baseline)
1. Flatten each image into a 1-D vector of pixel values (or extract hand-crafted features with HOG / color histograms).
2. Train a simple classifier: Logistic Regression, SVM, or Random Forest.
3. Fast to train, useful as a sanity-check baseline.

### Option B — Convolutional Neural Network (CNN) from scratch
1. Build a small CNN: a few Conv → ReLU → MaxPool blocks, then Flatten → Dense → Sigmoid.
2. Example architecture:
   - Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
   - Conv2D(64, 3×3) → ReLU → MaxPool(2×2)
   - Flatten → Dense(128) → ReLU → Dropout(0.5) → Dense(1) → Sigmoid
3. Loss function: Binary Cross-Entropy.
4. Optimizer: Adam (good default).

### Option C — Transfer Learning (recommended for best results)
1. Load a pretrained model (e.g. MobileNetV2 or ResNet18) without the top classification layer.
2. Freeze the pretrained layers.
3. Add your own Dense → Sigmoid head.
4. Fine-tune the last few layers later if needed.

---

## Phase 5: Training & Backpropagation

**Goal:** Train the model and understand what happens under the hood.

- **Forward pass:** Input image → model → predicted probability (cat or not).
- **Loss calculation:** Compare prediction to the true label using Binary Cross-Entropy.
- **Backpropagation:** Compute the gradient of the loss with respect to every weight in the network by applying the chain rule layer by layer, from output back to input.
- **Weight update:** The optimizer (e.g. Adam) uses those gradients to adjust the weights so the loss decreases.
- **Epoch:** One full pass through the entire training set. Typically we train for many epochs.
- **Batch size:** We don't feed all images at once — we split the training set into mini-batches (e.g. 32 images). Each batch triggers one forward + backward pass and one weight update.

### Key things to monitor during training
- Training loss per epoch (should go down).
- Validation loss per epoch (should also go down — if it starts going up, we may be overfitting).
- Training accuracy vs. validation accuracy.

---

## Phase 6: Overfitting vs. Underfitting

**Goal:** Diagnose and fix common training problems.

### Underfitting (high bias)
- **Symptom:** Both training and validation accuracy are low.
- **Cause:** Model is too simple to capture patterns.
- **Fixes:** Use a bigger / deeper model, train longer, reduce regularization, add more features.

### Overfitting (high variance)
- **Symptom:** Training accuracy is high but validation accuracy is much lower.
- **Cause:** Model memorizes the training data instead of learning general patterns.
- **Fixes:**
  - Add more training data or use data augmentation.
  - Add regularization: Dropout, L2 weight decay.
  - Use early stopping (stop training when validation loss stops improving).
  - Simplify the model (fewer layers / filters).

### The sweet spot
- We want a model that is complex enough to learn the real patterns but not so complex that it memorizes noise. Plotting training vs. validation loss curves is the best way to diagnose this.

---

## Phase 7: Hyperparameter Tuning

**Goal:** Find the best configuration for the model.

### What are hyperparameters?
Settings we choose *before* training — they are not learned from data. Examples:
- Learning rate (e.g. 0.001, 0.0001)
- Batch size (e.g. 16, 32, 64)
- Number of epochs
- Number of layers / filters in the CNN
- Dropout rate
- Optimizer choice (Adam, SGD, RMSprop)
- Weight decay (L2 regularization strength)

### Tuning strategies
1. **Manual tuning:** Change one thing at a time, observe validation performance.
2. **Grid Search:** Try every combination of a predefined set of values. Thorough but slow.
3. **Random Search:** Sample random combinations. Often more efficient than grid search.
4. **Bayesian Optimization:** Smart search that uses past results to choose the next set of hyperparameters (libraries: Optuna, Ray Tune).

### Important rules
- Always evaluate hyperparameter choices on the **validation set**, never on the test set.
- Keep the test set locked away until the very final evaluation.

---

## Phase 8: Evaluation Metrics

**Goal:** Measure how good the model really is.

| Metric | What it tells you |
|---|---|
| **Accuracy** | % of images classified correctly. Can be misleading with imbalanced classes. |
| **Precision** | Of all images the model called "cat," how many truly are cats? |
| **Recall (Sensitivity)** | Of all actual cat images, how many did the model catch? |
| **F1 Score** | Harmonic mean of precision and recall — a single balanced number. |
| **Confusion Matrix** | A 2×2 table showing True Positives, True Negatives, False Positives, False Negatives. |
| **ROC Curve & AUC** | Plots True Positive Rate vs. False Positive Rate at different thresholds. AUC closer to 1.0 = better. |

### How to use them
1. After training, run the model on the **test set**.
2. Compute all metrics above.
3. Look at the confusion matrix to understand *what kinds* of mistakes the model makes.
4. If precision matters more (e.g. you don't want false alarms), tune the decision threshold up. If recall matters more (you don't want to miss any cat), tune it down.

---

## Phase 9: Final Testing & Results

**Goal:** Get the definitive performance number.

1. Take the best model (chosen via validation performance in Phase 7).
2. Run it on the held-out **test set** — one time only.
3. Report accuracy, precision, recall, F1, and show the confusion matrix.
4. Compare against the classical ML baseline from Phase 4A.

---

## Phase 10: Wrap-Up & Next Steps

- Save the trained model (e.g. `model.h5` or `model.pth`).
- Write a short summary of results and lessons learned.
- Optional extensions:
  - Deploy the model as a simple web app (Flask / FastAPI + HTML upload form).
  - Try multi-class classification (cat, dog, bird, …).
  - Experiment with Grad-CAM to visualize what the model "looks at" in each image.

---

## Concept Glossary

| Term | Short explanation |
|---|---|
| **Backpropagation** | Algorithm that computes gradients of the loss w.r.t. each weight by applying the chain rule backward through the network. |
| **Epoch** | One complete pass through the entire training dataset. |
| **Batch size** | Number of samples processed before the model's weights are updated. |
| **Learning rate** | Step size for each weight update. Too high = unstable, too low = slow. |
| **Loss function** | Measures how far predictions are from the true labels (e.g. Binary Cross-Entropy). |
| **Optimizer** | Algorithm that updates weights using gradients (e.g. SGD, Adam). |
| **Regularization** | Techniques to prevent overfitting (Dropout, L2 penalty, data augmentation). |
| **Early stopping** | Stop training when validation loss hasn't improved for N epochs. |
| **Transfer learning** | Reusing a model pretrained on a large dataset (like ImageNet) for a new task. |
| **Confusion matrix** | Table showing TP, TN, FP, FN counts for a classifier. |
