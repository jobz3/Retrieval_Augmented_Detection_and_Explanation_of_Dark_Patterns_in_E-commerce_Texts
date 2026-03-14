"""
Encode texts into fixed-size embeddings using sentence-transformers, BERT, or RoBERTa.

Three encoder options (set in config.yaml → retrieval.encoder):
  "sbert"   — sentence-transformers/all-mpnet-base-v2  (fast, good quality)
  "bert"    — bert-base-uncased CLS pooling
  "roberta" — roberta-large CLS pooling
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def get_sbert_encoder(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def encode_with_sbert(
    texts: list[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Return (N, D) float32 numpy array of L2-normalised embeddings."""
    model = get_sbert_encoder(model_name)
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


def encode_texts(
    texts: list[str],
    encoder: str = "sbert",
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2",
    bert_model: str = "bert-base-uncased",
    roberta_model: str = "roberta-large",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Unified entry point. Dispatches to the appropriate encoder.

    Args:
        encoder: "sbert", "bert", or "roberta"
    """
    if encoder == "sbert":
        return encode_with_sbert(texts, sbert_model, batch_size, show_progress)
    elif encoder == "bert":
        return encode_with_hf(texts, bert_model, batch_size, show_progress=show_progress)
    elif encoder == "roberta":
        return encode_with_hf(texts, roberta_model, batch_size, show_progress=show_progress)
    else:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose: sbert, bert, roberta")
