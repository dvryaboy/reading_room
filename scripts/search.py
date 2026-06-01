#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "openai>=1.0",
# ]
# ///
"""
search.py

Semantic search over the dblect paper corpus.

Embeds the query via OpenAI, does cosine similarity against corpus.db,
and looks up chunk text from the JSONL files in the chunks directory.

Usage (run from the repo root):
    ./scripts/search.py "incremental view maintenance self-maintainability"
    ./scripts/search.py --n 20 "property-based testing generator shrinking"
    ./scripts/search.py --chunks data/paper-chunks --db data/corpus.db "refinement types"

Options:
    --db FILE        Path to corpus.db (default: <repo>/data/corpus.db)
    --chunks DIR     Directory of per-paper JSONL chunk files (default: <repo>/data/paper-chunks)
    --n N            Number of results to return (default: 10)
    --model TEXT     OpenAI embedding model (default: text-embedding-3-small)
"""

import argparse
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# Chunk text lookup
# ---------------------------------------------------------------------------

def load_chunk_index(chunks_dir: Path) -> dict[str, str]:
    """
    Load all chunk text from JSONL files into a dict keyed by chunk ID.
    Lazy-loads on first call; call once and pass around.
    """
    index: dict[str, str] = {}
    jsonl_files = list(chunks_dir.glob("*.jsonl"))
    if not jsonl_files:
        return index
    for f in jsonl_files:
        try:
            for line in f.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                index[rec["id"]] = rec["content"]
        except Exception:
            continue
    return index


# ---------------------------------------------------------------------------
# Vector similarity
# ---------------------------------------------------------------------------

def decode_vector(blob: bytes) -> np.ndarray:
    """Decode a stored float32 vector from llm's sqlite blob format."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def embed_query(query: str, model: str) -> np.ndarray:
    client = OpenAI()
    resp = client.embeddings.create(input=query, model=model)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def search(
    query: str,
    db_path: Path,
    chunks_dir: Path,
    n: int,
    model: str,
) -> list[dict]:
    if not db_path.exists():
        print(f"Error: corpus.db not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    # Embed the query
    query_vec = embed_query(query, model)

    # Load all vectors from SQLite
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT id, embedding FROM embeddings WHERE collection_id = "
        "(SELECT id FROM collections WHERE name = 'corpus' LIMIT 1)"
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No embeddings found in corpus. Run embed-multi first.", file=sys.stderr)
        sys.exit(1)

    # Score all chunks
    scored = []
    for chunk_id, blob in rows:
        vec = decode_vector(blob)
        score = cosine_similarity(query_vec, vec)
        scored.append((score, chunk_id))

    scored.sort(reverse=True)
    top = scored[:n]

    # Load chunk text index (JSONL files)
    chunk_index = load_chunk_index(chunks_dir)

    results = []
    for score, chunk_id in top:
        # ID format: "filename.pdf::N"
        parts = chunk_id.rsplit("::", 1)
        source = parts[0] if parts else chunk_id
        text = chunk_index.get(chunk_id, "(text not available — re-embed with --store)")
        results.append({
            "score": score,
            "id": chunk_id,
            "source": source,
            "text": text,
        })

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results: list[dict], query: str):
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        bar = "█" * int(r["score"] * 20)
        print(f"[{i:2d}] {r['score']:.3f} {bar}")
        print(f"     {r['source']}")
        # Show up to 400 chars of the chunk, clean whitespace
        snippet = " ".join(r["text"].split())[:400]
        if len(" ".join(r["text"].split())) > 400:
            snippet += "…"
        print(f"     {snippet}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semantic search over the dblect paper corpus."
    )
    data_dir = Path(__file__).resolve().parent.parent / "data"
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--db", default=str(data_dir / "corpus.db"),
                        help="Path to corpus.db")
    parser.add_argument("--chunks", default=str(data_dir / "paper-chunks"),
                        help="Directory of per-paper JSONL chunk files")
    parser.add_argument("--n", type=int, default=10, help="Number of results")
    parser.add_argument("--model", default="text-embedding-3-small",
                        help="OpenAI embedding model")
    args = parser.parse_args()

    query = " ".join(args.query)
    db_path = Path(args.db)
    chunks_dir = Path(args.chunks)

    if not chunks_dir.exists():
        print(f"Warning: chunks directory not found at {chunks_dir} — text lookup disabled",
              file=sys.stderr)
        chunks_dir = Path("/dev/null")  # graceful degradation

    results = search(query, db_path, chunks_dir, args.n, args.model)
    print_results(results, query)


if __name__ == "__main__":
    main()
