"""
pipeline.py — End-to-end RAG pipeline.

Steps:
  1. Load chunks
  2. Build hybrid retriever index
  3. Load reader (LLM)
  4. Rerank retrieved chunks with cross-encoder
  5. Generate answers with improved prompt (best guess mode)
  6. Save output JSON

Usage:
    python pipeline.py --queries data/leaderboard_queries.json --output data/system_outputs/system_output_1.json
"""

import json
import argparse
from pathlib import Path

from chunker import chunk_all
from retriever import HybridRetriever
from reader import FewShotReader


def load_queries(path: str) -> dict[str, str]:
    """Load queries JSON. Handles both dict {"1": "question"} and list [{"id":1, "question":"..."}] formats."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        result = {}
        for i, item in enumerate(data):
            if isinstance(item, dict):
                qid = str(item.get("id", item.get("qid", i + 1)))
                question = item.get("question", item.get("query", str(item)))
            else:
                qid = str(i + 1)
                question = str(item)
            result[qid] = question
        return result
    return data


# ── Step 4: Cross-encoder reranker ────────────────────────────────────────────

def load_reranker():
    from sentence_transformers import CrossEncoder
    print("[Reranker] Loading cross-encoder/ms-marco-MiniLM-L-6-v2...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("[Reranker] Loaded.")
    return reranker

def rerank(reranker, query: str, chunks: list[str], top_k: int) -> list[str]:
    scores = reranker.predict([(query, chunk) for chunk in chunks])
    ranked = sorted(zip(scores, chunks), reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]


# ── Step 5: Improved prompt (best guess mode) ─────────────────────────────────

BEST_GUESS_SYSTEM_PROMPT = """You are a factual QA assistant specializing in Pittsburgh and Carnegie Mellon University.
Answer in as few words as possible — ideally just a name, date, number, or short phrase.
Use the provided context to answer. If the answer is not explicitly stated, make your best guess based on the context.
Never say "I don't know" — always provide the most likely answer."""


def run_pipeline(
    queries_path: str,
    chunks_path: str = "data/chunks/chunks_sentence.json",
    output_path: str = "system_outputs/system_output_1.json",
    top_k: int = 10,
    candidate_k: int = 20,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    dense_model: str = "BAAI/bge-large-en-v1.5",
    dense_index_path: str = "data/dense_index",
    use_reranker: bool = True,
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
        retriever.sparse.build_index(chunks)
    else:
        print("[Pipeline] Building retriever indexes...")
        retriever.build_index()
        retriever.save_dense_index(dense_index_path)

    # ── 3. Load reader ───────────────────────────────────────────────────────
    reader = FewShotReader(model_name=model_name)
    # Override system prompt with best-guess version
    import reader as reader_module
    reader_module.SYSTEM_PROMPT = BEST_GUESS_SYSTEM_PROMPT

    # ── 4. Load reranker ─────────────────────────────────────────────────────
    reranker = load_reranker() if use_reranker else None

    # ── 5. Run queries ───────────────────────────────────────────────────────
    queries = load_queries(queries_path)
    outputs = {}

    for qid, question in queries.items():
        print(f"\n[Q{qid}] {question}")

        # Retrieve more candidates than needed so reranker has room to work
        retrieved = retriever.retrieve(question, top_k=candidate_k)
        context_chunks = [r.text for r in retrieved]

        # Rerank and keep top_k
        if reranker:
            context_chunks = rerank(reranker, question, context_chunks, top_k=top_k)

        answer = reader.answer(question, context_chunks)
        print(f"  → {answer}")
        outputs[qid] = answer

    # ── 6. Save output ───────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"\n[Pipeline] Done. Outputs saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/leaderboard_queries.json")
    parser.add_argument("--chunks", default="data/chunks/chunks_sentence.json")
    parser.add_argument("--output", default="data/system_outputs/system_output_1.json")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--candidate_k", type=int, default=20)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--dense_model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--dense_index", default="data/dense_index")
    parser.add_argument("--no_reranker", action="store_true", help="Disable cross-encoder reranker")
    args = parser.parse_args()

    run_pipeline(
        queries_path=args.queries,
        chunks_path=args.chunks,
        output_path=args.output,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        model_name=args.model,
        dense_model=args.dense_model,
        dense_index_path=args.dense_index,
        use_reranker=not args.no_reranker,
    )
