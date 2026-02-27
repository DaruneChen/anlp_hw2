"""
retriever.py — Hybrid retrieval combining dense (FAISS) and sparse (BM25).
Fusion via Reciprocal Rank Fusion (RRF).

Usage:
    retriever = HybridRetriever(chunks)
    results = retriever.retrieve("Who founded CMU?", top_k=5)
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import faiss
import bm25s
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    text: str
    score: float


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        print(f"[Dense] Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine sim (with normalized vecs)
        self.index.add(embeddings.astype(np.float32))
        print(f"[Dense] FAISS index built. dim={dim}")

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    def save(self, path: str = "data/dense_index"):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_chunks.json", "w") as f:
            json.dump(self.chunks, f)

    def load(self, path: str = "data/dense_index"):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_chunks.json") as f:
            self.chunks = json.load(f)


class SparseRetriever:
    def __init__(self):
        self.retriever = None
        self.chunks = []

    def build_index(self, chunks: list[dict]):
        self.chunks = chunks
        corpus = [c["text"].split() for c in chunks]  # tokenized
        print(f"[Sparse] Building BM25 index over {len(corpus)} chunks...")
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus)
        print("[Sparse] Done.")

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        tokenized_query = query.split()
        results, scores = self.retriever.retrieve([tokenized_query], k=top_k)
        # results shape: (1, top_k), scores shape: (1, top_k)
        indices = results[0].tolist()
        score_list = scores[0].tolist()
        return list(zip(indices, score_list))


class HybridRetriever:
    """
    Combines dense and sparse retrieval using Reciprocal Rank Fusion (RRF).
    RRF score = sum(1 / (k + rank_i)) for each retriever i.
    """

    def __init__(self, chunks: list[dict], dense_model: str = "all-MiniLM-L6-v2",
                 rrf_k: int = 60, dense_weight: float = 0.7, sparse_weight: float = 0.3):
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.dense = DenseRetriever(model_name=dense_model)
        self.sparse = SparseRetriever()

    def build_index(self):
        self.dense.build_index(self.chunks)
        self.sparse.build_index(self.chunks)

    def _rrf_fuse(self, dense_results: list[tuple[int, float]],
                  sparse_results: list[tuple[int, float]]) -> dict[int, float]:
        scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0) + self.dense_weight * (1.0 / (self.rrf_k + rank + 1))

        for rank, (idx, _) in enumerate(sparse_results):
            scores[idx] = scores.get(idx, 0) + self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))

        return scores

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20) -> list[RetrievedChunk]:
        dense_results = self.dense.retrieve(query, top_k=candidate_k)
        sparse_results = self.sparse.retrieve(query, top_k=candidate_k)

        fused = self._rrf_fuse(dense_results, sparse_results)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievedChunk(
                chunk_id=self.chunks[idx]["chunk_id"],
                source=self.chunks[idx]["source"],
                text=self.chunks[idx]["text"],
                score=score,
            )
            for idx, score in ranked
        ]

    def save_dense_index(self, path: str = "data/dense_index"):
        self.dense.save(path)

    def load_dense_index(self, path: str = "data/dense_index"):
        self.dense.load(path)
        self.dense.chunks = self.chunks


if __name__ == "__main__":
    # Quick smoke test
    chunks_path = Path("data/chunks/chunks_sentence.json")
    with open(chunks_path) as f:
        chunks = json.load(f)

    retriever = HybridRetriever(chunks)
    retriever.build_index()

    query = "Who is Pittsburgh named after?"
    results = retriever.retrieve(query, top_k=5)
    print(f"\nTop results for: '{query}'")
    for r in results:
        print(f"  [{r.score:.4f}] ({r.source}) {r.text[:120]}...")
