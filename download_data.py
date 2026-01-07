from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    urllib.request.urlretrieve(url, dst)


def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/raw")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = os.environ.get("DATA_URL", "").strip()
    local_zip = out_dir / "dataset.zip"

    if url:
        download(url, local_zip)
        unzip(local_zip, out_dir)
        return

    raise SystemExit()


if __name__ == "__main__":
    main()
