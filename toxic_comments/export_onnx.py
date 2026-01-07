from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@hydra.main(config_path="../configs/infer", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model_dir = Path(cfg.model_dir) / "hf"
    out_path = Path(cfg.model_dir) / "model.onnx"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dummy = tokenizer(
        "export onnx",
        truncation=True,
        padding="max_length",
        max_length=192,
        return_tensors="pt",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
