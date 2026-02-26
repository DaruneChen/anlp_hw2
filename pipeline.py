"""
pipeline.py — End-to-end RAG pipeline.

Steps:
  1. Load chunks
  2. Build hybrid retriever index
  3. Load reader (LLM)
  4. Run over query file
  5. Save output JSON

Usage:
    python src/pipeline.py --queries leaderboard_queries.json --output system_outputs/system_output_1.json
"""

import json
import argparse
from pathlib import Path

from chunker import chunk_all
from retriever import HybridRetriever
from reader import FewShotReader


def load_queries(path: str) -> dict[str, str]:
    """Load queries JSON. Expected format: {"1": "question...", "2": "..."}"""
    with open(path) as f:
        return json.load(f)


def run_pipeline(
    queries_path: str,
    chunks_path: str = "data/chunks/chunks_sentence.json",
    output_path: str = "system_outputs/system_output_1.json",
    top_k: int = 5,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    dense_model: str = "all-MiniLM-L6-v2",
    dense_index_path: str = "data/dense_index",
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load chunks ──────────────────────────────────────────────────────
    print(f"[Pipeline] Loading chunks from {chunks_path}...")
    if not Path(chunks_path).exists():
        print("[Pipeline] No chunks found, running chunker first...")
        chunk_all()
    with open(chunks_path) as f:
        chunks = json.load(f)
    print(f"[Pipeline] {len(chunks)} chunks loaded.")

    # ── 2. Build / load retriever ────────────────────────────────────────────
    retriever = HybridRetriever(chunks, dense_model=dense_model)
    dense_faiss = Path(f"{dense_index_path}.faiss")
    if dense_faiss.exists():
        print("[Pipeline] Loading existing dense index...")
        retriever.load_dense_index(dense_index_path)
        retriever.sparse.build_index(chunks)  # BM25 is fast, rebuild each time
    else:
        print("[Pipeline] Building retriever indexes...")
        retriever.build_index()
        retriever.save_dense_index(dense_index_path)

    # ── 3. Load reader ───────────────────────────────────────────────────────
    reader = FewShotReader(model_name=model_name)

    # ── 4. Run queries ───────────────────────────────────────────────────────
    queries = load_queries(queries_path)
    outputs = {}

    for qid, question in queries.items():
        print(f"\n[Q{qid}] {question}")
        retrieved = retriever.retrieve(question, top_k=top_k)
        context_chunks = [r.text for r in retrieved]
        answer = reader.answer(question, context_chunks)
        print(f"  → {answer}")
        outputs[qid] = answer

    # ── 5. Save output ───────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"\n[Pipeline] Done. Outputs saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="leaderboard_queries.json")
    parser.add_argument("--chunks", default="data/chunks/chunks_sentence.json")
    parser.add_argument("--output", default="system_outputs/system_output_1.json")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--dense_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--dense_index", default="data/dense_index")
    args = parser.parse_args()

    run_pipeline(
        queries_path=args.queries,
        chunks_path=args.chunks,
        output_path=args.output,
        top_k=args.top_k,
        model_name=args.model,
        dense_model=args.dense_model,
        dense_index_path=args.dense_index,
    )
