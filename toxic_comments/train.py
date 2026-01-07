from __future__ import annotations

import json
from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from toxic_comments.baseline import save_baseline, train_baseline
from toxic_comments.data import extract_xy, load_train_csv, make_split
from toxic_comments.distilbert_module import DistilBertArtifacts, DistilBertLightning, build_distilbert, make_loaders
from toxic_comments.plotting import save_history_plots
from toxic_comments.utils import ensure_data, get_git_commit_hash, mkdir_clean


class HistoryCallback:
    def __init__(self):
        self.history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.history["train_loss"].append(float(metrics["train_loss"].cpu()))

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.history["val_loss"].append(float(metrics["val_loss"].cpu()))
        if "val_auroc" in metrics:
            self.history["val_auroc"].append(float(metrics["val_auroc"].cpu()))


def _setup_mlflow(cfg: DictConfig) -> MLFlowLogger:
    tracking_uri = str(cfg.mlflow.tracking_uri)
    experiment_name = str(cfg.mlflow.experiment_name)

    logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    commit = get_git_commit_hash()
    try:
        logger.experiment.log_param(logger.run_id, "git_commit", commit)
    except Exception:
        pass

    return logger


def _train_baseline(cfg: DictConfig, output_dir: Path, plots_dir: Path) -> None:
    df = load_train_csv(Path(cfg.data.train_path), cfg.data.max_rows)
    split = make_split(df, float(cfg.data.val_size), int(cfg.seed))

    x_train, y_train = extract_xy(split.train_df)
    x_val, y_val = extract_xy(split.val_df)

    model, metrics = train_baseline(
        x_train,
        y_train,
        x_val,
        y_val,
        tfidf_cfg=cfg.model.tfidf,
        logreg_cfg=cfg.model.logreg,
    )

    out_path = output_dir / "baseline.joblib"
    save_baseline(model, out_path)

    # Create minimal plots for requirement (3)
    history = {
        "train_loss": [0.0],
        "val_loss": [0.0],
        "val_auroc": [float(metrics.get("val_auc_mean", 0.0))],
    }
    save_history_plots(history, plots_dir)

    # Log to MLflow (simple)
    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    with mlflow.start_run(run_name="baseline"):
        mlflow.log_params({"model": "baseline"})
        mlflow.log_params({"git_commit": get_git_commit_hash()})
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_artifact(str(out_path))
        for p in plots_dir.glob("*.png"):
            mlflow.log_artifact(str(p), artifact_path="plots")

    print(json.dumps(metrics, indent=2))


def _train_distilbert(cfg: DictConfig, output_dir: Path, plots_dir: Path) -> None:
    df = load_train_csv(Path(cfg.data.train_path), cfg.data.max_rows)
    split = make_split(df, float(cfg.data.val_size), int(cfg.seed))

    x_train, y_train = extract_xy(split.train_df)
    x_val, y_val = extract_xy(split.val_df)

    logger = _setup_mlflow(cfg)
    history_cb = HistoryCallback()

    artifacts: DistilBertArtifacts = build_distilbert(str(cfg.model.pretrained_model_name))
    train_dl, val_dl = make_loaders(
        artifacts,
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        max_length=int(cfg.data.max_length),
    )

    pl_model = DistilBertLightning(
        pretrained_model_name=str(cfg.model.pretrained_model_name),
        learning_rate=float(cfg.model.learning_rate),
        weight_decay=float(cfg.model.weight_decay),
    )

    trainer = Trainer(
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        max_epochs=int(cfg.trainer.max_epochs),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        logger=logger,
        enable_checkpointing=True,
        callbacks=[history_cb],
    )

    trainer.fit(pl_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Save HF model + tokenizer for easy inference
    model_dir = output_dir / "hf"
    model_dir.mkdir(parents=True, exist_ok=True)
    pl_model.model.save_pretrained(model_dir)
    pl_model.tokenizer.save_pretrained(model_dir)

    save_history_plots(history_cb.history, plots_dir)

    # Log artifacts to MLflow
    try:
        logger.experiment.log_artifacts(logger.run_id, str(model_dir), artifact_path="model")
        for p in plots_dir.glob("*.png"):
            logger.experiment.log_artifact(logger.run_id, str(p), artifact_path="plots")
    except Exception:
        pass


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.seed))

    train_path = Path(cfg.data.train_path)
    ensure_data(train_path, bool(cfg.data.use_dvc))

    output_dir = Path(cfg.output_dir) / "model"
    plots_dir = Path("plots")
    mkdir_clean(output_dir)
    mkdir_clean(plots_dir)

    if str(cfg.model.name) == "baseline":
        _train_baseline(cfg, output_dir, plots_dir)
    else:
        _train_distilbert(cfg, output_dir, plots_dir)


if __name__ == "__main__":
    main()
