"""
pipeline.py — End-to-end RAG runner.

Key improvements:
- FIXES syntax issues in your current pipeline.py
- deeper retrieval (candidate_k) + reranking
- better embedding model default (BGE large)
- BM25 tokenization improvements (in retriever.py)

Example:
python src/pipeline.py \
  --queries leaderboard_queries.json \
  --output system_outputs/system_output_1.json \
  --top_k 8 \
  --candidate_k 30
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from chunker import chunk_all
from retriever import HybridRetriever
from reader import FewShotReader


def load_queries(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}

    if isinstance(data, list):
        out: dict[str, str] = {}
        for i, item in enumerate(data):
            if isinstance(item, dict):
                qid = str(item.get("id", item.get("qid", i + 1)))
                q = item.get("question", item.get("query", ""))
                out[qid] = str(q)
            else:
                out[str(i + 1)] = str(item)
        return out

    raise ValueError("Unrecognized query JSON format.")


def load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder
    print(f"[Reranker] Loading {model_name}...")
    r = CrossEncoder(model_name)
    print("[Reranker] Loaded.")
    return r


def rerank_chunks(reranker, query: str, chunks: list[str], top_k: int) -> list[str]:
    if not chunks:
        return []
    pairs = [(query, c) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="leaderboard_queries.json")
    parser.add_argument("--output", default="system_outputs/system_output_1.json")

    parser.add_argument("--chunks", default="data/chunks/chunks_sentence.json")
    parser.add_argument("--chunk_strategy", choices=["fixed", "paragraph", "sentence"], default="sentence")
    parser.add_argument("--chunk_max_words", type=int, default=350)
    parser.add_argument("--chunk_overlap_words", type=int, default=100)
    parser.add_argument("--chunk_overlap_sents", type=int, default=2)

    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--candidate_k", type=int, default=30)

    parser.add_argument("--dense_model", default="intfloat/e5-large-v2")
    parser.add_argument("--dense_index", default="data/dense_index")
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--dense_weight", type=float, default=0.7)
    parser.add_argument("--sparse_weight", type=float, default=0.3)

    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--no_reranker", action="store_true")
    parser.add_argument("--reranker_model", default="BAAI/bge-reranker-base")

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"[Pipeline] Missing chunks at {chunks_path}. Running chunker...")
        chunk_all(
            strategy=args.chunk_strategy,
            max_words=args.chunk_max_words,
            overlap_words=args.chunk_overlap_words,
            overlap_sents=args.chunk_overlap_sents,
        )

    print(f"[Pipeline] Loading chunks from {chunks_path}...")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[Pipeline] Loaded {len(chunks)} chunks.")

    retriever = HybridRetriever(
        chunks,
        dense_model=args.dense_model,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )

    dense_faiss = Path(f"{args.dense_index}.faiss")
    dense_chunks = Path(f"{args.dense_index}_chunks.json")

    if dense_faiss.exists() and dense_chunks.exists():
        print("[Pipeline] Loading existing dense index...")
        retriever.load_dense_index(args.dense_index)
        print("[Pipeline] Building BM25 index...")
        retriever.sparse.build_index(chunks)
    else:
        print("[Pipeline] Building indexes (dense + sparse)...")
        retriever.build_index()
        retriever.save_dense_index(args.dense_index)

    reader = FewShotReader(model_name=args.model)

    reranker = None
    if not args.no_reranker:
        reranker = load_reranker(args.reranker_model)

    queries = load_queries(args.queries)
    outputs: dict[str, str] = {}

    for qid, question in queries.items():
        question = str(question).strip()
        if not question:
            outputs[qid] = "unknown"
            continue

        print(f"\n[Q{qid}] {question}")

        retrieved = retriever.retrieve(
            question,
            top_k=args.candidate_k,
            candidate_k=args.candidate_k,
        )
        candidate_texts = [r.text for r in retrieved]

        if reranker is not None:
            context_chunks = rerank_chunks(reranker, question, candidate_texts, top_k=args.top_k)
        else:
            context_chunks = candidate_texts[:args.top_k]

        ans = reader.answer(question, context_chunks)
        print(f"  → {ans}")
        outputs[qid] = ans

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"\n[Pipeline] Done. Saved to {args.output}")


if __name__ == "__main__":
    main()