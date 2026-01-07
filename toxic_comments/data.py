from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from toxic_comments import LABELS


@dataclass
class DatasetSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame


def load_train_csv(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)

    if path.suffix == ".zip":
        df = pd.read_csv(path, compression="zip", nrows=max_rows)
    else:
        df = pd.read_csv(path, nrows=max_rows)

    return df


def make_split(df: pd.DataFrame, val_size: float, seed: int) -> DatasetSplit:
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=seed, shuffle=True)
    return DatasetSplit(
        train_df=train_df.reset_index(drop=True), val_df=val_df.reset_index(drop=True)
    )


def extract_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    texts = df["comment_text"].astype(str).to_numpy()
    labels = df[LABELS].astype(np.float32).to_numpy()
    return texts, labels
