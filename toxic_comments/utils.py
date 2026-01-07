from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Optional


def get_git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.strip().decode("utf-8")
    except Exception:
        return "unknown"


def run_dvc_pull(path: Path) -> None:
    subprocess.run(["dvc", "pull", str(path)], check=True)


def ensure_data(path: str | Path, use_dvc: bool) -> None:
    path = Path(path)

    if path.suffix == ".csv" and path.exists():
        return

    if path.suffix == ".zip" and path.exists():
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(path.parent)
        return

    if path.suffix == ".zip":
        csv_path = path.with_suffix("")  # train.csv.zip -> train.csv
        if csv_path.exists():
            return

    if path.suffix == ".csv":
        zip_path = path.with_suffix(path.suffix + ".zip")  # train.csv -> train.csv.zip
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(path.parent)
            return

    raise FileNotFoundError(
        f"Data file not found: {path}. Put train.csv into {path.parent} or enable DVC in config."
    )


def mkdir_clean(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_int(value: Optional[int]) -> Optional[int]:
    return value if value is not None else None
