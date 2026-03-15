"""
Bootstrap the canonical raw EC-DarkPattern dataset path from the upstream clone.

Usage:
    python -m src.data.bootstrap_raw
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import typer

from src.utils.config import load_config


app = typer.Typer()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.command()
def main(force: bool = typer.Option(False, help="Overwrite the canonical raw dataset if it exists.")) -> None:
    config = load_config()
    source_path = config.paths.upstream_dataset_repo / "dataset" / "dataset.tsv"
    destination_path = config.paths.raw_data

    if not source_path.exists():
        raise FileNotFoundError(
            "Upstream dataset source not found.\n"
            f"Expected: {source_path}\n"
            "Keep data/dataset_repo as the upstream reference clone and ensure dataset/dataset.tsv exists."
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        source_hash = file_sha256(source_path)
        destination_hash = file_sha256(destination_path)
        if source_hash == destination_hash and not force:
            print(f"Canonical raw dataset already up to date: {destination_path}")
            print(f"sha256={destination_hash}")
            return
        if not force:
            raise FileExistsError(
                "Canonical raw dataset already exists with different contents.\n"
                f"Destination: {destination_path}\n"
                "Re-run with --force to overwrite it from the upstream reference clone."
            )

    shutil.copy2(source_path, destination_path)
    print(f"Copied raw dataset -> {destination_path}")
    print(f"sha256={file_sha256(destination_path)}")


if __name__ == "__main__":
    app()
