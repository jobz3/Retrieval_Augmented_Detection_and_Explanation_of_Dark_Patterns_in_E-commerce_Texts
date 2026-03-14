"""
Retrieval interface — given a query text, return the top-k training examples.

Three strategies:
  "knn"       — naive top-k by cosine similarity
  "prototype" — prefer examples closest to their class centroid
  "diversity" — MMR: balance similarity and diversity (penalise near-duplicates)

Usage example:
    from src.retrieval.retrieve import Retriever
    r = Retriever(encoder="sbert", strategy="knn", k=5)
    examples = r.retrieve("Only 2 left in stock!")
    # returns list of {"text": ..., "category": ..., "score": ...}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.retrieval.embeddings import encode_texts
from src.retrieval.index import load_index, INDEX_DIR

ROOT = Path(__file__).resolve().parents[2]


class Retriever:
    """
    Loads a pre-built FAISS index and retrieves examples for a query text.

    Args:
        encoder:   "sbert", "bert", or "roberta" — must match a built index
        strategy:  "knn", "prototype", or "diversity"
        k:         number of examples to return
        index_dir: directory containing .faiss / .npy / .json files
    """

    def __init__(
        self,
        encoder: str = "sbert",
        strategy: str = "knn",
        k: int = 5,
        diversity_lambda: float = 0.5,
        index_dir: Path = INDEX_DIR,
    ):
        self.encoder = encoder
        self.strategy = strategy
        self.k = k
        self.diversity_lambda = diversity_lambda

        print(f"Loading FAISS index [{encoder}] ...")
        self.index, self.embeddings, self.metadata = load_index(encoder, index_dir)

        if strategy == "prototype":
            self._centroids = self._compute_centroids()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict]:
        """
        Return k dicts: {"text": str, "category": str, "score": float}.
        """
        query_emb = encode_texts([query], encoder=self.encoder, show_progress=False)  # (1, D)

        if self.strategy == "knn":
            return self._knn(query_emb)
        elif self.strategy == "prototype":
            return self._prototype(query_emb)
        elif self.strategy == "diversity":
            return self._diversity(query_emb)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

    def retrieve_batch(self, queries: list[str]) -> list[list[dict]]:
        """Retrieve for a list of queries."""
        return [self.retrieve(q) for q in queries]

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _knn(self, query_emb: np.ndarray) -> list[dict]:
        # Retrieve slightly more to allow dedup of exact matches
        fetch_k = min(self.k + 5, self.index.ntotal)
        scores, indices = self.index.search(query_emb, fetch_k)
        scores, indices = scores[0], indices[0]

        results = []
        seen_texts: set[str] = set()
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            item = self.metadata[idx]
            if item["text"] in seen_texts:
                continue
            seen_texts.add(item["text"])
            results.append({**item, "score": float(score)})
            if len(results) == self.k:
                break
        return results

    def _prototype(self, query_emb: np.ndarray) -> list[dict]:
        """
        Prototype-preferring retrieval:
        Score each candidate as a combination of query similarity and
        proximity to its category centroid (prototypicality).
        """
        # First fetch a candidate pool (5x k)
        fetch_k = min(self.k * 5, self.index.ntotal)
        scores, indices = self.index.search(query_emb, fetch_k)
        scores, indices = scores[0], indices[0]

        candidates = []
        for sim, idx in zip(scores, indices):
            if idx < 0:
                continue
            item = self.metadata[idx]
            cat = item["category"]
            if cat not in self._centroids:
                proto_score = 0.0
            else:
                centroid = self._centroids[cat]
                emb = self.embeddings[idx]
                # cosine similarity to centroid (both L2-normalised)
                proto_score = float(np.dot(emb, centroid))
            # Combined score: equal weight on query similarity and prototypicality
            combined = 0.5 * float(sim) + 0.5 * proto_score
            candidates.append({**item, "score": combined, "_idx": idx})

        # Sort by combined score, deduplicate
        candidates.sort(key=lambda x: x["score"], reverse=True)
        seen_texts: set[str] = set()
        results = []
        for c in candidates:
            if c["text"] in seen_texts:
                continue
            seen_texts.add(c["text"])
            results.append({k: v for k, v in c.items() if k != "_idx"})
            if len(results) == self.k:
                break
        return results

    def _diversity(self, query_emb: np.ndarray) -> list[dict]:
        """
        Maximal Marginal Relevance (MMR):
        Iteratively pick the candidate that maximises:
            λ * sim(c, query) - (1-λ) * max_sim(c, already_selected)
        """
        lam = self.diversity_lambda
        fetch_k = min(self.k * 10, self.index.ntotal)
        scores, indices = self.index.search(query_emb, fetch_k)
        scores, indices = scores[0], indices[0]

        # Build candidate pool
        pool = []
        seen_texts: set[str] = set()
        for sim, idx in zip(scores, indices):
            if idx < 0:
                continue
            item = self.metadata[idx]
            if item["text"] in seen_texts:
                continue
            seen_texts.add(item["text"])
            pool.append({"item": item, "sim": float(sim), "emb": self.embeddings[idx], "idx": idx})

        selected = []
        selected_embs = []

        while len(selected) < self.k and pool:
            best_score = -1e9
            best_i = 0
            for i, cand in enumerate(pool):
                if not selected_embs:
                    redundancy = 0.0
                else:
                    redundancy = max(float(np.dot(cand["emb"], s)) for s in selected_embs)
                mmr = lam * cand["sim"] - (1 - lam) * redundancy
                if mmr > best_score:
                    best_score = mmr
                    best_i = i

            chosen = pool.pop(best_i)
            selected.append({**chosen["item"], "score": best_score})
            selected_embs.append(chosen["emb"])

        return selected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_centroids(self) -> dict[str, np.ndarray]:
        """Compute mean L2-normalised embedding per category."""
        from collections import defaultdict

        groups: dict[str, list[np.ndarray]] = defaultdict(list)
        for i, item in enumerate(self.metadata):
            groups[item["category"]].append(self.embeddings[i])

        centroids = {}
        for cat, embs in groups.items():
            mat = np.stack(embs)
            mean_emb = mat.mean(axis=0)
            # L2 normalise centroid
            norm = np.linalg.norm(mean_emb)
            centroids[cat] = mean_emb / (norm + 1e-8)
        return centroids
