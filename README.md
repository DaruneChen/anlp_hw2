# ANLP Spring 2026 HW2 — RAG Pipeline

## Project Structure

```
.
├── leaderboard_queries.json
├── requirements.txt
├── data/
│   ├── raw/                  # scraped .txt files
│   └── chunks/               # chunked JSONs
├── src/
│   ├── scraper.py            # Step 1: collect raw data
│   ├── chunker.py            # Step 2: chunk documents
│   ├── retriever.py          # Step 3: dense + sparse hybrid retrieval
│   ├── reader.py             # Step 4: LLM-based answer generation
│   ├── pipeline.py           # End-to-end runner
│   └── evaluate.py           # Score predictions
└── system_outputs/
    └── system_output_1.json  # Submit this to leaderboard / Canvas
```

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

### Step 1 — Scrape data
```bash
python src/scraper.py
```

### Step 2 — Chunk documents
```bash
python src/chunker.py
# Edit STRATEGY in chunker.py to try "fixed", "paragraph", or "sentence"
```

### Step 3 — Run full pipeline
```bash
python src/pipeline.py \
    --queries leaderboard_queries.json \
    --output system_outputs/system_output_1.json \
    --top_k 5 \
    --model mistralai/Mistral-7B-Instruct-v0.2
```

### Step 4 — Evaluate (on training data)
```bash
python src/evaluate.py \
    --pred system_outputs/system_output_1.json \
    --ref data/train/reference_answers.json
```

## Key Design Decisions

| Component | Default | Alternatives to try |
|-----------|---------|---------------------|
| Chunking | sentence-aware | fixed-size, paragraph |
| Dense model | `all-MiniLM-L6-v2` | `all-mpnet-base-v2`, MTEB top models |
| Sparse | BM25 (bm25s) | rank-bm25 |
| Fusion | RRF (k=60) | score normalization + weighted avg |
| Reader | Mistral-7B-Instruct | Llama-3-8B-Instruct, Phi-3 |

## Submission Format

```json
{
    "1": "Answer 1",
    "2": "Answer 2",
    ...
}
```
