"""
Translate the English training set to German and build a bilingual FAISS index.

Steps:
  1. Load data/processed/train_v2.csv  (English training examples)
  2. Translate each text using Helsinki-NLP/opus-mt-en-de
  3. Save translated CSV to data/processed/train_v2_de.csv
  4. Encode translated texts with the multilingual encoder
  5. Build FAISS index and save as outputs/indices/multilingual_de_*

The resulting index contains German-language versions of English training
examples, so German queries retrieve semantically aligned German neighbours
rather than mismatched English ones.

Usage:
    python -m src.retrieval.translate_index
    python -m src.retrieval.translate_index --batch-size 32 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "outputs" / "indices"


def translate_texts(texts: list[str], batch_size: int = 32, device: str = "cpu") -> list[str]:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    model_name = "Helsinki-NLP/opus-mt-en-de"
    print(f"Loading {model_name} (device={device}) ...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    translations = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(torch_device) for k, v in encoded.items()}
        with torch.no_grad():
            translated = model.generate(**encoded)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translations.extend(decoded)
        print(f"  Translated {min(start + batch_size, n)}/{n}", end="\r", flush=True)
    print()
    return translations


def build_translated_index(batch_size: int = 32, device: str = "cpu") -> None:
    from src.retrieval.embeddings import encode_texts
    from src.retrieval.index import build_index, save_index
    import faiss

    # Load training data
    csv_path = PROCESSED_DIR / "train_v2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} English training examples")

    # Translate
    de_out = PROCESSED_DIR / "train_v2_de.csv"
    if de_out.exists():
        print(f"Found existing translation: {de_out} — skipping translation step")
        df_de = pd.read_csv(de_out)
        de_texts = df_de["text_de"].tolist()
    else:
        de_texts = translate_texts(texts, batch_size=batch_size, device=device)
        df_de = df.copy()
        df_de["text_de"] = de_texts
        df_de.to_csv(de_out, index=False)
        print(f"Saved translated texts → {de_out}")

    # Encode with multilingual encoder
    print("Encoding translated texts with multilingual encoder ...")
    embeddings = encode_texts(
        de_texts,
        encoder="multilingual",
        batch_size=64,
        show_progress=True,
        device=device,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Build metadata using translated text
    metadata = []
    for i, row in df.iterrows():
        metadata.append({
            "text": de_texts[i],
            "text_en": row["text"],
            "category": row["category"],
        })
        if "page_id" in df.columns and not pd.isna(row.get("page_id")):
            metadata[-1]["page_id"] = int(row["page_id"])

    # Save index under encoder name "multilingual_de"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    encoder_name = "multilingual_de"
    index = build_index(embeddings)
    save_index(index, embeddings, metadata, encoder_name, INDEX_DIR)
    print(f"Bilingual index saved as '{encoder_name}' → {INDEX_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English training index to German")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    build_translated_index(batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
