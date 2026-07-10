# Cat Detector ML Learning Project

A binary image-classification project that walks through the full machine-learning lifecycle for detecting whether an image contains a cat. The project is designed as a hands-on learning repo, with scripts for data setup, preprocessing, model design, training, hyperparameter tuning, evaluation, and prediction.

## Important note on results

The checked-in plots and final metrics are from the synthetic-data demo path. Synthetic mode is enabled by default so the pipeline can run in offline or sandboxed environments. For real image classification work, switch to CIFAR-10 in `src/phase1_data_setup.py` and rerun the full pipeline.

## What it demonstrates

- **Data setup:** Creates a cat-vs-not-cat binary classification target from synthetic data or CIFAR-10.
- **Preprocessing:** Saves train/validation/test splits, normalizes images, and demonstrates augmentation.
- **Modeling:** Compares a logistic-regression baseline, a CNN from scratch, and a MobileNetV2 transfer-learning scaffold.
- **Training and tuning:** Tracks overfitting, class imbalance, regularization, and threshold selection.
- **Evaluation:** Produces confusion matrices, ROC curves, precision, recall, F1, and AUC.
- **Inference:** Includes `src/predict.py` for running a trained `.keras` model against a local image.

## Repository layout

```text
ml-cat-detection/
  src/
    phase1_data_setup.py          Data loading, EDA, train/validation/test split
    phase3_preprocessing.py       Normalization and augmentation
    phase4_model.py               Baseline, CNN, and transfer-learning model definitions
    phase5_6_7_train_tune.py      Training, overfitting diagnosis, hyperparameter tuning
    phase8_9_evaluate.py          Validation/test metrics, threshold tuning, plots
    predict.py                    Local image inference with a trained model
  *.png                           Generated learning artifacts and metric plots
  plan.md                         Ten-phase learning plan
  requirements.txt                Python dependencies
```

## Setup

Use Python 3.10 or 3.11 for best TensorFlow compatibility.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
python src/phase1_data_setup.py
python src/phase3_preprocessing.py
python src/phase4_model.py
python src/phase5_6_7_train_tune.py
python src/phase8_9_evaluate.py
```

Run prediction after training:

```bash
python src/predict.py path/to/image.jpg
```

## Switching from synthetic data to CIFAR-10

1. Open `src/phase1_data_setup.py`.
2. Change `USE_SYNTHETIC = True` to `USE_SYNTHETIC = False`.
3. Rerun the full pipeline.

For a more realistic training run, increase `DEFAULT_EPOCHS` in `src/phase5_6_7_train_tune.py` and set sample limits to use more of CIFAR-10.

## Synthetic demo results

| Metric | Value |
| --- | --- |
| Accuracy | 99.4% |
| Precision | 94.3% |
| Recall | 100% |
| F1 score | 97.1% |
| ROC AUC | 0.997 |

The key lesson is threshold selection: with the default threshold of `0.5`, the model can look acceptable on accuracy while failing recall. Tuning the threshold exposed why precision, recall, F1, ROC, and confusion matrices matter for imbalanced classification.

## Validation

CI intentionally avoids training models. The automated check compiles every Python source file:

```bash
python -m compileall -q src
```

Full validation requires running the pipeline locally because it creates data files, model artifacts, and plots.

<!-- add-to-portfolio -->

## License

MIT
