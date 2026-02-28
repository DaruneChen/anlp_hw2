"""
retriever.py — Hybrid retrieval (dense + sparse) with fusion.

Dense:
- SentenceTransformer embeddings
- FAISS IndexFlatIP over normalized embeddings (cosine similarity)

Sparse:
- BM25 via bm25s with a better tokenizer

Fusion:
- Reciprocal Rank Fusion (RRF) with weights

No high-level RAG frameworks used (assignment-compliant).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import faiss
import bm25s
from sentence_transformers import SentenceTransformer


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source: str
    text: str
    score: float


class DenseRetriever:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.chunks: list[dict] = []

    def build_index(self, chunks: list[dict], batch_size: int = 64) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        print(f"[Dense] Encoding {len(texts)} chunks with {self.model_name}...")
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)
        print(f"[Dense] FAISS built. dim={dim} n={self.index.ntotal}")

    def retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self.index is None:
            raise RuntimeError("Dense index not built/loaded.")
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q, top_k)

        out: list[tuple[int, float]] = []
        for idx, sc in zip(indices[0].tolist(), scores[0].tolist()):
            if idx == -1:
                continue
            out.append((int(idx), float(sc)))
        return out

    def save(self, path: str) -> None:
        if self.index is None:
            raise RuntimeError("No dense index to save.")
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)


class SparseRetriever:
    def __init__(self):
        self.bm25 = bm25s.BM25()
        self._built = False
        self.chunks: list[dict] = []

    def build_index(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        corpus = [tokenize(c["text"]) for c in chunks]
        print(f"[Sparse] Building BM25 over {len(corpus)} chunks...")
        self.bm25.index(corpus)
        self._built = True
        print("[Sparse] Done.")

    def retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self._built:
            raise RuntimeError("Sparse index not built.")
        q = tokenize(query)
        results, scores = self.bm25.retrieve([q], k=top_k)

        idxs = results[0].tolist()
        scs = scores[0].tolist()
        return [(int(i), float(s)) for i, s in zip(idxs, scs)]


class HybridRetriever:
    def __init__(
        self,
        chunks: list[dict],
        dense_model: str = "BAAI/bge-large-en-v1.5",
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.dense = DenseRetriever(dense_model)
        self.sparse = SparseRetriever()

    def build_index(self) -> None:
        self.dense.build_index(self.chunks)
        self.sparse.build_index(self.chunks)

    def save_dense_index(self, path: str) -> None:
        self.dense.save(path)

    def load_dense_index(self, path: str) -> None:
        self.dense.load(path)
        self.dense.chunks = self.chunks  # keep metadata consistent

    def _rrf_scores(self, results: list[tuple[int, float]], weight: float) -> dict[int, float]:
        scores: dict[int, float] = {}
        for rank, (idx, _raw) in enumerate(results):
            scores[idx] = scores.get(idx, 0.0) + weight * (1.0 / (self.rrf_k + rank + 1))
        return scores
    def expand_query(query):
        expansions = [
            query,
            query.replace("CMU", "Carnegie Mellon University"),
            query.replace("Pittsburgh", "City of Pittsburgh"),
            query + " history",
            query + " event",
        ]
        return list(set(expansions))
    def retrieve(self, query: str, top_k: int, candidate_k: int) -> list[RetrievedChunk]:
        if candidate_k < top_k:
            candidate_k = top_k

        dense_res = self.dense.retrieve(query, candidate_k)
        sparse_res = self.sparse.retrieve(query, candidate_k)

        fused: dict[int, float] = {}
        for idx, sc in self._rrf_scores(dense_res, self.dense_weight).items():
            fused[idx] = fused.get(idx, 0.0) + sc
        for idx, sc in self._rrf_scores(sparse_res, self.sparse_weight).items():
            fused[idx] = fused.get(idx, 0.0) + sc

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        out: list[RetrievedChunk] = []
        for idx, sc in ranked:
            c = self.chunks[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    source=c["source"],
                    text=c["text"],
                    score=float(sc),
                )
            )
        return out