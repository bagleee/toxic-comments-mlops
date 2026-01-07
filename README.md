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

Data is managed via **DVC**

## Metrics

Main metric: mean ROC-AUC across 6 labels.
We also log per-class ROC-AUC and macro F1.
