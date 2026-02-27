"""
chunker.py — Document chunking utilities.

Supports:
- fixed-size word chunking with overlap
- paragraph chunking
- sentence-aware chunking with sentence overlap

Outputs:
data/chunks/chunks_<strategy>.json

Usage:
    python src/chunker.py --strategy sentence --max_words 350 --overlap_words 100 --overlap_sents 2
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict


RAW_DIR = Path("data/raw")
CHUNKS_DIR = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str       # "<source_stem>_<idx>"
    source: str         # original filename
    text: str
    start_char: int
    strategy: str


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_fixed(text: str, size_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if size_words <= 0:
        return []
    if overlap_words < 0:
        overlap_words = 0
    if overlap_words >= size_words:
        overlap_words = max(0, size_words - 1)

    step = size_words - overlap_words
    out: list[str] = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i + size_words]))
        i += step
    return out


def chunk_paragraphs(text: str, max_words: int, overlap_words: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    out: list[str] = []
    current_words: list[str] = []
    count = 0

    for para in paragraphs:
        w = para.split()
        if count + len(w) > max_words and current_words:
            out.append(" ".join(current_words))
            if overlap_words > 0:
                tail = current_words[-overlap_words:] if overlap_words < len(current_words) else current_words[:]
                current_words = tail[:]
                count = len(current_words)
            else:
                current_words = []
                count = 0

        current_words.extend(w)
        count += len(w)

    if current_words:
        out.append(" ".join(current_words))

    return out


def chunk_sentences(text: str, max_words: int, overlap_sents: int) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sents:
        return []

    if overlap_sents < 0:
        overlap_sents = 0

    out: list[str] = []
    current: list[str] = []
    count = 0

    for sent in sents:
        wc = len(sent.split())

        # Very long single sentence: hard split by words
        if wc > max_words and max_words > 0:
            if current:
                out.append(" ".join(current))
                current = current[-overlap_sents:] if overlap_sents else []
                count = sum(len(x.split()) for x in current)

            words = sent.split()
            i = 0
            while i < len(words):
                out.append(" ".join(words[i:i + max_words]))
                i += max_words
            continue

        if count + wc > max_words and current:
            out.append(" ".join(current))
            current = current[-overlap_sents:] if overlap_sents else []
            count = sum(len(x.split()) for x in current)

        current.append(sent)
        count += wc

    if current:
        out.append(" ".join(current))

    return out


def chunk_file(
    path: Path,
    strategy: str,
    max_words: int,
    overlap_words: int,
    overlap_sents: int,
    min_chars: int = 60,
) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _clean_text(raw)

    if strategy == "fixed":
        raw_chunks = chunk_fixed(text, max_words, overlap_words)
    elif strategy == "paragraph":
        raw_chunks = chunk_paragraphs(text, max_words, overlap_words)
    elif strategy == "sentence":
        raw_chunks = chunk_sentences(text, max_words, overlap_sents)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    chunks: list[Chunk] = []
    char_offset = 0
    for i, c in enumerate(raw_chunks):
        c = c.strip()
        if len(c) < min_chars:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{path.stem}_{i:04d}",
                source=path.name,
                text=c,
                start_char=char_offset,
                strategy=strategy,
            )
        )
        char_offset += len(c) + 1

    return chunks


def chunk_all(strategy: str, max_words: int, overlap_words: int, overlap_sents: int) -> list[Chunk]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"{RAW_DIR} not found. Run scraper first.")

    all_chunks: list[Chunk] = []
    for raw_file in sorted(RAW_DIR.glob("*.txt")):
        chunks = chunk_file(
            raw_file,
            strategy=strategy,
            max_words=max_words,
            overlap_words=overlap_words,
            overlap_sents=overlap_sents,
        )
        all_chunks.extend(chunks)
        print(f"[Chunker] {raw_file.name} → {len(chunks)} chunks")

    out_path = CHUNKS_DIR / f"chunks_{strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2, ensure_ascii=False)

    print(f"[Chunker] Total chunks: {len(all_chunks)} → saved to {out_path}")
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["fixed", "paragraph", "sentence"], default="sentence")
    parser.add_argument("--max_words", type=int, default=350)
    parser.add_argument("--overlap_words", type=int, default=100)
    parser.add_argument("--overlap_sents", type=int, default=2)
    args = parser.parse_args()

    chunk_all(
        strategy=args.strategy,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
        overlap_sents=args.overlap_sents,
    )


if __name__ == "__main__":
    main()