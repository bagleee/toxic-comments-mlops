# Toxic Comments Classification

This project is about detecting toxic comments in text.
The task is multi-label classification: one comment can belong to several toxicity categories at the same time.

Labels:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

---

## Data

Dataset: **Jigsaw Toxic Comment Classification Challenge (Kaggle)**.

The dataset should be placed manually in:

```
data/raw/train.csv
```

Required columns:

- `comment_text` — text of the comment
- label columns listed above

DVC is **not** used in this project.

---

## Environment

The project uses **Poetry**.

Install dependencies:

```bash
poetry install
```

---

## Training (Transformer)

Training is done using Hydra configs.

Main training config:

```
configs/train.yaml
```

Run training:

```bash
poetry run python -m toxic_comments.commands
```

Trained model will be saved to:

```
outputs/model/hf
```

---

## Baseline Model

Baseline model: **TF-IDF + Logistic Regression**.

Baseline config:

```
configs/model/baseline.yaml
```

Run baseline training:

```bash
poetry run python -m toxic_comments.commands baseline
```

Baseline artifacts are saved to:

```
outputs/baseline/tfidf_logreg
```

---

## Inference

Inference config:

```
configs/infer.yaml
```

Run inference:

```bash
poetry run python -m toxic_comments.commands infer
```

You can override text from command line:

```bash
poetry run python -m toxic_comments.commands infer infer.text="this is a test comment"
```

---

## MLflow

MLflow is used to track experiments.

Start MLflow UI:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## Model Export

The transformer model can be exported to ONNX format.

Export script:

```
toxic_comments/export_onnx.py
```

---

## Metrics

Main metric:

- mean ROC-AUC over 6 labels

Also logged:

- per-class ROC-AUC
- macro F1-score
