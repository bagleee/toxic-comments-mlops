from __future__ import annotations

from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import onnxruntime as ort
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from toxic_comments import LABELS
from toxic_comments.utils import ensure_data


def _predict_pytorch(model_dir: Path, text: str) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    enc = tokenizer(text, truncation=True, padding=True, max_length=192, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    return {name: float(probs[i]) for i, name in enumerate(LABELS)}


def _predict_onnx(onnx_path: Path, model_dir: Path, text: str) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    enc = tokenizer(text, truncation=True, padding="max_length", max_length=192, return_tensors="np")
    session = ort.InferenceSession(str(onnx_path))

    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    outputs = session.run(None, inputs)
    logits = outputs[0]
    probs = 1 / (1 + np.exp(-logits))
    probs = probs.squeeze(0)

    return {name: float(probs[i]) for i, name in enumerate(LABELS)}


@hydra.main(config_path="../configs/infer", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model_dir = Path(cfg.model_dir)

    # If using DVC, ensure model dir exists via training artifacts.
    text = cfg.text
    if text is None:
        text = "This is a sample comment"

    if bool(cfg.use_onnx):
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}. Run export_onnx first.")
        out = _predict_onnx(onnx_path, model_dir / "hf", str(text))
    else:
        out = _predict_pytorch(model_dir / "hf", str(text))

    print(out)


if __name__ == "__main__":
    main()
