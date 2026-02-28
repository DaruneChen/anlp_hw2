"""
retriever.py — Hybrid retrieval (dense + sparse) with fusion.

Features:
- SentenceTransformer dense retrieval (FAISS cosine similarity)
- BM25 sparse retrieval (bm25s)
- Query expansion
- Reciprocal Rank Fusion
- Correct embedding prefixes for E5 and BGE models
- Assignment compliant (no LangChain / LlamaIndex)
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


# ============================
# Dense Retriever
# ============================

class DenseRetriever:

    def __init__(self, model_name: str):

        self.model_name = model_name

        print(f"[Dense] Loading model: {model_name}")

        self.model = SentenceTransformer(model_name)

        self.index: Optional[faiss.Index] = None

        self.chunks: list[dict] = []

        self.use_e5 = "e5" in model_name.lower()

        self.use_bge = "bge" in model_name.lower()


    def _format_passage(self, text: str) -> str:

        if self.use_e5:
            return "passage: " + text

        if self.use_bge:
            return text

        return text


    def _format_query(self, query: str) -> str:

        if self.use_e5:
            return "query: " + query

        if self.use_bge:
            return (
                "Represent this sentence for searching relevant passages: "
                + query
            )

        return query


    def build_index(
        self,
        chunks: list[dict],
        batch_size: int = 64,
    ) -> None:

        self.chunks = chunks

        texts = [
            self._format_passage(c["text"])
            for c in chunks
        ]

        print(
            f"[Dense] Encoding {len(texts)} chunks..."
        )

        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        dim = emb.shape[1]

        self.index = faiss.IndexFlatIP(dim)

        self.index.add(emb)

        print(
            f"[Dense] FAISS built. dim={dim} n={self.index.ntotal}"
        )


    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:

        if self.index is None:
            raise RuntimeError("Dense index not built")

        q = self._format_query(query)

        q_emb = self.model.encode(
            [q],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(
            q_emb,
            top_k,
        )

        out = []

        for idx, sc in zip(
            indices[0],
            scores[0],
        ):

            if idx == -1:
                continue

            out.append(
                (int(idx), float(sc))
            )

        return out


    def save(self, path: str) -> None:

        if self.index is None:
            raise RuntimeError("No index")

        faiss.write_index(
            self.index,
            f"{path}.faiss",
        )

        with open(
            f"{path}_chunks.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.chunks,
                f,
                ensure_ascii=False,
            )


    def load(self, path: str) -> None:

        self.index = faiss.read_index(
            f"{path}.faiss"
        )

        with open(
            f"{path}_chunks.json",
            encoding="utf-8",
        ) as f:

            self.chunks = json.load(f)


# ============================
# Sparse Retriever (BM25)
# ============================

class SparseRetriever:

    def __init__(self):

        self.bm25 = bm25s.BM25()

        self._built = False

        self.chunks: list[dict] = []


    def build_index(
        self,
        chunks: list[dict],
    ) -> None:

        self.chunks = chunks

        corpus = [
            tokenize(c["text"])
            for c in chunks
        ]

        print(
            f"[Sparse] Building BM25 over {len(corpus)} chunks..."
        )

        self.bm25.index(corpus)

        self._built = True

        print("[Sparse] Done.")


    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:

        if not self._built:
            raise RuntimeError("Sparse index not built")

        q = tokenize(query)

        results, scores = self.bm25.retrieve(
            [q],
            k=top_k,
        )

        return [
            (int(i), float(s))
            for i, s in zip(
                results[0],
                scores[0],
            )
        ]


# ============================
# Hybrid Retriever
# ============================

class HybridRetriever:

    def __init__(
        self,
        chunks: list[dict],
        dense_model: str,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):

        self.chunks = chunks

        self.rrf_k = rrf_k

        self.dense_weight = dense_weight

        self.sparse_weight = sparse_weight

        self.dense = DenseRetriever(
            dense_model
        )

        self.sparse = SparseRetriever()


    def build_index(self) -> None:

        self.dense.build_index(
            self.chunks
        )

        self.sparse.build_index(
            self.chunks
        )


    def save_dense_index(
        self,
        path: str,
    ) -> None:

        self.dense.save(path)


    def load_dense_index(
        self,
        path: str,
    ) -> None:

        self.dense.load(path)

        self.dense.chunks = self.chunks


    def _rrf_scores(
        self,
        results,
        weight,
    ):

        scores = {}

        for rank, (idx, _) in enumerate(results):

            scores[idx] = (
                scores.get(idx, 0)
                + weight
                * (1 / (self.rrf_k + rank + 1))
            )

        return scores


    def _expand_query(
        self,
        query,
    ):

        expansions = [

            query,

            query.replace(
                "CMU",
                "Carnegie Mellon University",
            ),

            query.replace(
                "Carnegie Mellon University",
                "CMU",
            ),

            query.replace(
                "Pittsburgh",
                "City of Pittsburgh",
            ),

            query + " history",

            query + " event",

        ]

        return list(set(expansions))


    def retrieve(
        self,
        query,
        top_k,
        candidate_k,
    ):

        expanded = self._expand_query(
            query
        )

        dense_res = []

        sparse_res = []

        for q in expanded:

            dense_res.extend(
                self.dense.retrieve(
                    q,
                    candidate_k,
                )
            )

            sparse_res.extend(
                self.sparse.retrieve(
                    q,
                    candidate_k,
                )
            )

        fused = {}

        for idx, sc in self._rrf_scores(
            dense_res,
            self.dense_weight,
        ).items():

            fused[idx] = (
                fused.get(idx, 0)
                + sc
            )

        for idx, sc in self._rrf_scores(
            sparse_res,
            self.sparse_weight,
        ).items():

            fused[idx] = (
                fused.get(idx, 0)
                + sc
            )

        ranked = sorted(
            fused.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        out = []

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