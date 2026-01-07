from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MultilabelAUROC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from toxic_comments import LABELS


class ToxicTextDataset(Dataset):
    def __init__(self, texts: np.ndarray, labels: np.ndarray, tokenizer: Any, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


@dataclass
class DistilBertArtifacts:
    model: nn.Module
    tokenizer: Any


def build_distilbert(pretrained_model_name: str) -> DistilBertArtifacts:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )
    return DistilBertArtifacts(model=model, tokenizer=tokenizer)


class DistilBertLightning(LightningModule):
    def __init__(self, pretrained_model_name: str, learning_rate: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()

        artifacts = build_distilbert(pretrained_model_name)
        self.model = artifacts.model
        self.tokenizer = artifacts.tokenizer

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.auroc = MultilabelAUROC(num_labels=len(LABELS))

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        out = self.model(**batch)
        return out.logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        labels = batch.pop("labels")
        logits = self(**batch)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        labels = batch.pop("labels")
        logits = self(**batch)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        self.auroc.update(probs, labels.int())
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        auc = self.auroc.compute()
        self.log("val_auroc", auc, prog_bar=True)
        self.auroc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )


def make_loaders(
    artifacts: DistilBertArtifacts,
    train_texts: np.ndarray,
    train_labels: np.ndarray,
    val_texts: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    max_length: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = ToxicTextDataset(train_texts, train_labels, artifacts.tokenizer, max_length)
    val_ds = ToxicTextDataset(val_texts, val_labels, artifacts.tokenizer, max_length)

    train_dl = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )
    return train_dl, val_dl
