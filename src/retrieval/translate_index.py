"""
Translate the English training set to a target language and build a bilingual
FAISS index for cross-lingual retrieval.

Steps:
  1. Load data/processed/train_v2.csv  (English training examples)
  2. Translate each text using a Helsinki-NLP/opus-mt-en-<lang> model
  3. Save translated CSV to data/processed/train_v2_<lang>.csv
  4. Encode translated texts with the multilingual encoder
  5. Build FAISS index and save as outputs/indices/multilingual_<lang>_*

The resulting index contains target-language versions of English training
examples, so target-language queries retrieve semantically aligned neighbours
rather than mismatched English ones.

Supported target languages: any pair available as Helsinki-NLP/opus-mt-en-<lang>.
This module defaults to German for back-compat with prior runs.

Usage:
    python -m src.retrieval.translate_index                  # German (default)
    python -m src.retrieval.translate_index --lang it        # Italian
    python -m src.retrieval.translate_index --lang de --batch-size 32 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "outputs" / "indices"


def translate_texts(
    texts: list[str],
    tgt_lang: str = "de",
    batch_size: int = 32,
    device: str = "cpu",
) -> list[str]:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    model_name = f"Helsinki-NLP/opus-mt-en-{tgt_lang}"
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


def build_translated_index(
    tgt_lang: str = "de",
    batch_size: int = 32,
    device: str = "cpu",
) -> None:
    from src.retrieval.embeddings import encode_texts
    from src.retrieval.index import build_index, save_index

    # Load training data
    csv_path = PROCESSED_DIR / "train_v2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} English training examples")

    # Translate (reuse a cached translation only if it matches the current train_v2)
    text_col = f"text_{tgt_lang}"
    out_csv  = PROCESSED_DIR / f"train_v2_{tgt_lang}.csv"

    def _cache_is_valid(path: Path) -> bool:
        """A cached translation is reusable only if it has the required column,
        the same row count, and the same English source texts as the current
        train_v2.csv. Guards against a stale cache from an older split."""
        try:
            cached = pd.read_csv(path)
        except Exception:
            return False
        if text_col not in cached.columns or "text" not in cached.columns:
            print(f"  cache {path.name} missing required columns — re-translating")
            return False
        if len(cached) != len(df):
            print(f"  cache {path.name} has {len(cached)} rows but train_v2 has "
                  f"{len(df)} — stale, re-translating")
            return False
        if cached["text"].tolist() != texts:
            print(f"  cache {path.name} English source texts differ from train_v2 "
                  f"— stale, re-translating")
            return False
        return True

    if out_csv.exists() and _cache_is_valid(out_csv):
        print(f"Found valid translation cache: {out_csv} — skipping translation step")
        df_tgt = pd.read_csv(out_csv)
        tgt_texts = df_tgt[text_col].tolist()
    else:
        tgt_texts = translate_texts(texts, tgt_lang=tgt_lang, batch_size=batch_size, device=device)
        df_tgt = df.copy()
        df_tgt[text_col] = tgt_texts
        df_tgt.to_csv(out_csv, index=False)
        print(f"Saved translated texts → {out_csv}")

    # Encode with multilingual encoder
    print("Encoding translated texts with multilingual encoder ...")
    embeddings = encode_texts(
        tgt_texts,
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
            "text": tgt_texts[i],
            "text_en": row["text"],
            "category": row["category"],
        })
        if "page_id" in df.columns and not pd.isna(row.get("page_id")):
            metadata[-1]["page_id"] = int(row["page_id"])

    # Save index under encoder name "multilingual_<lang>"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    encoder_name = f"multilingual_{tgt_lang}"
    index = build_index(embeddings)
    save_index(index, embeddings, metadata, encoder_name, INDEX_DIR)
    print(f"Bilingual index saved as '{encoder_name}' → {INDEX_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English training index to a target language")
    parser.add_argument("--lang",       type=str, default="de", help="Target language code (e.g. de, it). Default: de")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",     type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    build_translated_index(tgt_lang=args.lang, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
