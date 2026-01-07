# Detecting toxic comments in online discussions

## Problem statement
The goal of this project is to build a system that automatically detects toxic comments in online discussions.
Toxic content includes insults, threats, obscene language, and hate speech.

This is a **multi-label text classification** task: each comment may belong to several toxicity categories
simultaneously.

**Labels (6):** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

## Data
Dataset: Kaggle competition **Jigsaw Toxic Comment Classification Challenge**.

The code expects the file:

- `data/raw/train.csv.zip` (the Kaggle `train.csv` zipped)

Data is managed via **DVC** (do not commit data to Git).

## Metrics
Main metric: mean ROC-AUC across 6 labels.
We also log per-class ROC-AUC and macro F1.

---

## Setup (macOS)

### 1) Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Restart terminal, then:

```bash
poetry --version
```

### 2) Install dependencies

```bash
poetry install
```

### 3) Enable pre-commit

```bash
poetry run pre-commit install
poetry run pre-commit run -a
```

---

## Data (DVC)

### 1) Init DVC

```bash
poetry run dvc init
```

### 2) Put dataset file

Place Kaggle `train.csv.zip` here:

- `data/raw/train.csv.zip`

### 3) Track with DVC

```bash
poetry run dvc add data/raw/train.csv.zip
```

### 4) Add remote (Google Drive example)

```bash
poetry run dvc remote add --default gdrive_remote gdrive://<FOLDER_ID>/dvcstore
poetry run dvc remote modify gdrive_remote gdrive_acknowledge_abuse true
poetry run dvc push
```

---

## Train

### Baseline (fast)

```bash
poetry run python -m toxic_comments.train model=baseline
```

### DistilBERT (main model)

```bash
poetry run python -m toxic_comments.train model=distilbert trainer.max_epochs=1
```

Outputs:
- `outputs/model/` contains saved model artifacts
- `plots/` contains at least 3 metric plots

---

## Logging (MLflow)

Start MLflow server locally (optional for testing):

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080
```

Training logs metrics, params and git commit id to MLflow URI from config.

---

## Production preparation

### Export to ONNX

```bash
poetry run python -m toxic_comments.export_onnx
```

File:
- `outputs/model/model.onnx`

### TensorRT (optional)

On machines with NVIDIA + TensorRT installed:

```bash
bash scripts/convert_tensorrt.sh outputs/model/model.onnx outputs/model/model_trt.engine
```

---

## Infer

### Local CLI

```bash
poetry run python -m toxic_comments.infer text="You are stupid"
```

### MLflow Serving (optional)

If you logged a `pyfunc_model` in MLflow, you can serve it:

```bash
mlflow models serve -m runs:/<RUN_ID>/pyfunc_model -p 5000
```

Request example:

```bash
curl -X POST http://127.0.0.1:5000/invocations \
  -H "Content-Type: application/json" \
  --data '{"dataframe_split": {"columns": ["text"], "data": [["You are stupid"]]}}'
```
