from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def save_history_plots(history: Dict[str, List[float]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    for key, values in history.items():
        if not values:
            continue
        plt.figure()
        plt.plot(values)
        plt.title(key)
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{key}.png")
        plt.close()
