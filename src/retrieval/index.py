"""
Build and persist a FAISS flat index over the training split.

Usage:
    python -m src.retrieval.index --encoder sbert
    python -m src.retrieval.index --encoder bert
    python -m src.retrieval.index --encoder roberta

Outputs (one set per encoder):
    outputs/indices/{encoder}_embeddings.npy   — raw embedding matrix
    outputs/indices/{encoder}_index.faiss      — FAISS IVF / flat index
    outputs/indices/{encoder}_metadata.json    — per-row text / category / id
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import typer

from src.retrieval.embeddings import encode_texts

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "outputs" / "indices"

app = typer.Typer()


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (inner product = cosine similarity when
    embeddings are L2-normalised, which encode_texts guarantees).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    metadata: list[dict],
    encoder: str,
    out_dir: Path = INDEX_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / f"{encoder}_index.faiss"))
    np.save(out_dir / f"{encoder}_embeddings.npy", embeddings)
    with open(out_dir / f"{encoder}_metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"Saved FAISS index ({encoder}) → {out_dir}/")


def load_index(
    encoder: str, out_dir: Path = INDEX_DIR
) -> tuple[faiss.IndexFlatIP, np.ndarray, list[dict]]:
    """Load a previously built index for a given encoder."""
    index = faiss.read_index(str(out_dir / f"{encoder}_index.faiss"))
    embeddings = np.load(out_dir / f"{encoder}_embeddings.npy")
    with open(out_dir / f"{encoder}_metadata.json") as f:
        metadata = json.load(f)
    return index, embeddings, metadata


@app.command()
def main(
    encoder: str = typer.Option("sbert", help="Encoder: sbert | bert | roberta | multilingual | multilingual_de"),
    split: str = typer.Option("train", help="Which split to index (usually 'train')"),
    batch_size: int = typer.Option(64, help="Encoding batch size"),
) -> None:
    csv_path = PROCESSED_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m src.data.preprocess` first."
        )

    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()

    print(f"Encoding {len(texts)} texts with encoder='{encoder}' ...")
    embeddings = encode_texts(texts, encoder=encoder, batch_size=batch_size)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build metadata list (one dict per training example)
    metadata = df[["text", "category"]].to_dict(orient="records")
    if "page_id" in df.columns:
        for i, row in enumerate(metadata):
            row["page_id"] = int(df.iloc[i]["page_id"]) if not pd.isna(df.iloc[i]["page_id"]) else i

    index = build_index(embeddings)
    print(f"FAISS index size: {index.ntotal} vectors, dim={embeddings.shape[1]}")

    save_index(index, embeddings, metadata, encoder)


if __name__ == "__main__":
    app()
