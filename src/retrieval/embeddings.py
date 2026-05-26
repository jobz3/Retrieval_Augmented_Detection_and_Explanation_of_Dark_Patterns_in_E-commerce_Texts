"""
Encode texts into fixed-size embeddings using sentence-transformers, BERT, or RoBERTa.

Four encoder options (set in config.yaml → retrieval.encoder):
  "sbert"          — sentence-transformers/all-mpnet-base-v2  (fast, good quality, English-only)
  "bert"           — bert-base-uncased CLS pooling
  "roberta"        — roberta-large CLS pooling
  "multilingual"   — paraphrase-multilingual-MiniLM-L12-v2 (50+ languages, same dim as sbert)
"""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import torch
from tqdm import tqdm


_SBERT_CACHE: dict[str, object] = {}


def get_sbert_encoder(
    model_name: str = os.environ.get("SBERT_MODEL_PATH", "sentence-transformers/all-mpnet-base-v2"),
    device: str | None = None,
):
    from sentence_transformers import SentenceTransformer
    key = f"{model_name}::{device}"
    if key not in _SBERT_CACHE:
        _SBERT_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _SBERT_CACHE[key]


def encode_with_sbert(
    texts: list[str],
    model_name: str = os.environ.get("SBERT_MODEL_PATH", "sentence-transformers/all-mpnet-base-v2"),
    batch_size: int = 64,
    show_progress: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """Return (N, D) float32 numpy array of L2-normalised embeddings."""
    model = get_sbert_encoder(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def encode_with_hf(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
    max_length: int = 128,
    device: str | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    CLS-token pooling over a HuggingFace encoder (BERT, RoBERTa, etc.).
    Returns (N, D) float32 L2-normalised numpy array.
    """
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Encoding [{model_name}]")

    with torch.no_grad():
        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**enc)
            # CLS token embedding
            cls = outputs.last_hidden_state[:, 0, :]  # (B, D)
            # L2 normalise
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
            all_embeddings.append(cls.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


MULTILINGUAL_MODEL = os.environ.get(
    "MULTILINGUAL_MODEL_PATH",
    "paraphrase-multilingual-MiniLM-L12-v2",
)


def encode_texts(
    texts: list[str],
    encoder: str = "sbert",
    sbert_model: str = os.environ.get("SBERT_MODEL_PATH", "sentence-transformers/all-mpnet-base-v2"),
    bert_model: str = os.environ.get("BERT_MODEL_PATH", "bert-base-uncased"),
    roberta_model: str = "roberta-large",
    batch_size: int = 64,
    show_progress: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """
    Unified entry point. Dispatches to the appropriate encoder.

    Args:
        encoder: "sbert", "bert", "roberta", or "multilingual"
        device:  Force a specific device (e.g. "cpu") — useful when GPU memory
                 is shared with an LLM inference process (Ollama).
                 Defaults to CUDA if available.
    """
    if encoder == "sbert":
        return encode_with_sbert(texts, sbert_model, batch_size, show_progress, device=device)
    elif encoder == "bert":
        return encode_with_hf(texts, bert_model, batch_size, show_progress=show_progress, device=device)
    elif encoder == "roberta":
        return encode_with_hf(texts, roberta_model, batch_size, show_progress=show_progress, device=device)
    elif encoder == "multilingual" or encoder.startswith("multilingual_"):
        # multilingual_de, multilingual_it, etc. all share the same encoder;
        # only the underlying index differs (translated to that target language).
        return encode_with_sbert(texts, MULTILINGUAL_MODEL, batch_size, show_progress, device=device)
    else:
        raise ValueError(
            f"Unknown encoder '{encoder}'. Choose: sbert, bert, roberta, multilingual, multilingual_<lang>"
        )
