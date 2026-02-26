
import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict

RAW_DIR = Path("data/raw")
CHUNKS_DIR = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Chunk:
    chunk_id: str       # unique id: "<source_file>_<idx>"
    source: str         # original filename
    text: str
    start_char: int
    strategy: str



def chunk_fixed(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks



def chunk_paragraphs(text: str, max_words: int = 400) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current, count = [], [], 0
    for para in paragraphs:
        words = para.split()
        if count + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.extend(words)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_sentences(text: str, max_words: int = 300, overlap_sents: int = 1) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, count = [], [], 0
    for i, sent in enumerate(sentences):
        wc = len(sent.split())
        if count + wc > max_words and current:
            chunks.append(" ".join(current))
            # overlap: keep last N sentences
            current = current[-overlap_sents:] if overlap_sents else []
            count = sum(len(s.split()) for s in current)
        current.append(sent)
        count += wc
    if current:
        chunks.append(" ".join(current))
    return chunks



STRATEGY = "sentence"  # Change to "fixed" or "paragraph" to experiment

def chunk_file(path: Path, strategy: str = STRATEGY) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if strategy == "fixed":
        raw_chunks = chunk_fixed(text)
    elif strategy == "paragraph":
        raw_chunks = chunk_paragraphs(text)
    else:
        raw_chunks = chunk_sentences(text)

    chunks = []
    char_offset = 0
    for i, c in enumerate(raw_chunks):
        if len(c.strip()) < 30:  
            continue
        chunks.append(Chunk(
            chunk_id=f"{path.stem}_{i:04d}",
            source=path.name,
            text=c.strip(),
            start_char=char_offset,
            strategy=strategy,
        ))
        char_offset += len(c)
    return chunks


def chunk_all(strategy: str = STRATEGY):
    all_chunks = []
    for raw_file in RAW_DIR.glob("*.txt"):
        chunks = chunk_file(raw_file, strategy)
        all_chunks.extend(chunks)
        print(f"[OK] {raw_file.name} → {len(chunks)} chunks")

    out_path = CHUNKS_DIR / f"chunks_{strategy}.json"
    with open(out_path, "w") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)} → saved to {out_path}")
    return all_chunks


if __name__ == "__main__":
    chunk_all()
