from __future__ import annotations

import subprocess
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


def ensure_data(path: Path, use_dvc: bool) -> None:
    if path.exists():
        return

    if not use_dvc:
        raise FileNotFoundError(
            f"Data file not found: {path}. Put it there or enable DVC in config."
        )

    # Try pull via DVC
    run_dvc_pull(path)

    if not path.exists():
        raise FileNotFoundError(
            f"After DVC pull, data file still not found: {path}. Check DVC remote setup."
        )


def mkdir_clean(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_int(value: Optional[int]) -> Optional[int]:
    return value if value is not None else None
