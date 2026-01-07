from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main(cfg: DictConfig) -> None:
    model_dir = Path(cfg.infer.model_dir)
    text = str(cfg.infer.text)
    max_length = int(cfg.infer.max_length)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.infer.use_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    with torch.no_grad():
        enc = tokenizer(
            [text], truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).detach().cpu().tolist()[0]

    print(probs)
