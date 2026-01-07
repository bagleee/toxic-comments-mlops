from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    from toxic_comments.train import main as train_main

    train_main(cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def infer(cfg: DictConfig) -> None:
    from toxic_comments.infer import main as infer_main

    infer_main(cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def baseline(cfg: DictConfig) -> None:
    from toxic_comments.baseline import main as baseline_main

    baseline_main(cfg)


if __name__ == "__main__":
    train()
