#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "openai>=1.0",
# ]
# ///
"""
corpus.py — dblect paper corpus manager

Orchestrates extraction, incremental embedding, and search over the paper corpus.
Delegates extraction to extract_chunks.py and embedding to `llm`.

Commands:
    extract     Extract text chunks from PDFs into data/paper-chunks/
    embed       Incrementally embed any chunks not yet in data/corpus.db
    search      Semantic search over the embedded corpus
    show        Print a chunk (by id from search) plus its neighbors
    status      Show corpus stats (papers, chunks, embedded, not yet embedded)

Usage (run from the repo root):
    # Full build from scratch
    ./scripts/corpus.py extract
    ./scripts/corpus.py embed

    # Search (snippets), then expand a hit by its chunk id
    ./scripts/corpus.py search "incremental view maintenance"
    ./scripts/corpus.py search --n 20 "property-based testing generator shrinking"
    ./scripts/corpus.py search --full "refinement types"   # whole chunks, not snippets
    ./scripts/corpus.py show "Paper.pdf::7" --context 2     # a chunk + its neighbors

    # Add new papers: drop PDFs into data/papers/, then:
    ./scripts/corpus.py extract   # --resume is on by default; only new papers extracted
    ./scripts/corpus.py embed     # only new chunks embedded

    # Status
    ./scripts/corpus.py status

Environment:
    OPENAI_API_KEY    Required for embed and search
    CORPUS_DIR        Base data directory (default: <repo>/data)
"""

import argparse
import json
import os
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — override via env or CLI flags
# ---------------------------------------------------------------------------

DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
BASE_DIR       = Path(os.environ.get("CORPUS_DIR", str(DATA_DIR)))
PAPERS_DIR     = BASE_DIR / "papers"
CHUNKS_DIR     = BASE_DIR / "paper-chunks"
DB_PATH        = BASE_DIR / "corpus.db"
EMBED_LOG      = BASE_DIR / ".embedded_files"   # tracks which jsonl files are embedded
EXTRACT_SCRIPT = Path(__file__).parent / "extract_chunks.py"

CHUNK_SIZE     = 400   # tokens — conservative to stay under OpenAI's 8192 limit
CHUNK_OVERLAP  = 80
EMBED_MODEL    = "text-embedding-3-small"
SEARCH_N       = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embedded_set() -> set[str]:
    """Return the set of JSONL filenames already embedded."""
    if not EMBED_LOG.exists():
        return set()
    return set(EMBED_LOG.read_text().splitlines())


def mark_embedded(filename: str):
    with EMBED_LOG.open("a") as f:
        f.write(filename + "\n")


def run(cmd: list[str], **kwargs) -> int:
    return subprocess.run(cmd, **kwargs).returncode


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_extract(args):
    if not EXTRACT_SCRIPT.exists():
        print(f"Error: extract_chunks.py not found at {EXTRACT_SCRIPT}", file=sys.stderr)
        print("Make sure extract_chunks.py is in the same directory as corpus.py.", file=sys.stderr)
        sys.exit(1)

    if not PAPERS_DIR.exists():
        print(f"Error: papers directory not found at {PAPERS_DIR}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        "uv", "run", "--script", str(EXTRACT_SCRIPT),
        "--papers", str(PAPERS_DIR),
        "--out", str(CHUNKS_DIR),
        "--chunk-size", str(CHUNK_SIZE),
        "--overlap", str(CHUNK_OVERLAP),
    ]
    if not args.no_resume:
        cmd.append("--resume")

    rc = run(cmd)
    sys.exit(rc)


def cmd_embed(args):
    if not CHUNKS_DIR.exists() or not any(CHUNKS_DIR.glob("*.jsonl")):
        print("No chunk files found. Run `./corpus.py extract` first.", file=sys.stderr)
        sys.exit(1)

    # Find llm in PATH or venv
    llm = _find_llm()
    if not llm:
        print("Error: `llm` not found. Install with: pip install llm", file=sys.stderr)
        sys.exit(1)

    already_embedded = load_embedded_set()
    all_jsonl = sorted(CHUNKS_DIR.glob("*.jsonl"))
    pending = [f for f in all_jsonl if f.name not in already_embedded]

    if not pending:
        print(f"Nothing to embed — all {len(all_jsonl)} chunk files already embedded.")
        return

    print(f"Embedding {len(pending)} / {len(all_jsonl)} chunk files "
          f"({len(already_embedded)} already done)...")

    failed = []
    for i, jsonl_file in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] {jsonl_file.name}")
        rc = run([
            llm, "embed-multi", "corpus",
            str(jsonl_file),
            "-d", str(DB_PATH),
            "--format", "nl",
            "-m", EMBED_MODEL,
        ])
        if rc == 0:
            mark_embedded(jsonl_file.name)
        else:
            print(f"    Warning: embedding failed for {jsonl_file.name} (exit {rc})")
            failed.append(jsonl_file.name)

    print(f"\nDone. Embedded: {len(pending) - len(failed)} | Failed: {len(failed)}")
    if failed:
        print("Failed files (will retry on next embed run):")
        for f in failed:
            print(f"  {f}")


def _find_llm() -> str | None:
    import shutil
    # Check venv first, then PATH
    venv_llm = Path(sys.executable).parent / "llm"
    if venv_llm.exists():
        return str(venv_llm)
    return shutil.which("llm")


def cmd_search(args):
    query = " ".join(args.query)

    if not DB_PATH.exists():
        print(f"Error: corpus.db not found at {DB_PATH}. Run embed first.", file=sys.stderr)
        sys.exit(1)

    # Embed the query
    client = OpenAI()
    resp = client.embeddings.create(input=query, model=EMBED_MODEL)
    query_vec = np.array(resp.data[0].embedding, dtype=np.float32)

    # Load all vectors
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT id, embedding FROM embeddings WHERE collection_id = "
        "(SELECT id FROM collections WHERE name = 'corpus' LIMIT 1)"
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No embeddings in corpus.db. Run `./corpus.py embed` first.")
        return

    # Score
    scored = []
    for chunk_id, blob in rows:
        n = len(blob) // 4
        vec = np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)
        denom = np.linalg.norm(query_vec) * np.linalg.norm(vec)
        score = float(np.dot(query_vec, vec) / denom) if denom > 0 else 0.0
        scored.append((score, chunk_id))

    scored.sort(reverse=True)
    top = scored[:args.n]

    # Look up chunk text from JSONL files
    chunk_index = _load_chunk_index()

    print(f"\nQuery: {query}")
    print(f"Top {len(top)} results:\n")
    for i, (score, chunk_id) in enumerate(top, 1):
        text = chunk_index.get(chunk_id, "(text not in chunks dir)")
        bar = "█" * int(score * 20)
        print(f"[{i:2d}] {score:.3f} {bar}")
        print(f"     {chunk_id}")
        if args.full:
            for line in text.splitlines():
                print(f"     {line}")
        else:
            cleaned = " ".join(text.split())
            snippet = cleaned[:400] + ("…" if len(cleaned) > 400 else "")
            print(f"     {snippet}")
        print()


def _load_chunk_index() -> dict[str, str]:
    index: dict[str, str] = {}
    if not CHUNKS_DIR.exists():
        return index
    for f in CHUNKS_DIR.glob("*.jsonl"):
        try:
            for line in f.read_text(errors="replace").splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    index[rec["id"]] = rec["content"]
        except Exception:
            continue
    return index


def cmd_status(args):
    all_pdfs = list(PAPERS_DIR.glob("*.pdf")) + list(PAPERS_DIR.glob("*.PDF")) \
        if PAPERS_DIR.exists() else []
    all_jsonl = list(CHUNKS_DIR.glob("*.jsonl")) if CHUNKS_DIR.exists() else []
    already_embedded = load_embedded_set()
    pending = [f for f in all_jsonl if f.name not in already_embedded]

    total_chunks = 0
    for f in all_jsonl:
        try:
            total_chunks += sum(1 for l in f.read_text().splitlines() if l.strip())
        except Exception:
            pass

    db_count = 0
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE collection_id = "
                "(SELECT id FROM collections WHERE name = 'corpus' LIMIT 1)"
            ).fetchone()
            db_count = row[0] if row else 0
            conn.close()
        except Exception:
            pass

    print(f"\n=== Corpus Status ===")
    print(f"Papers dir:       {PAPERS_DIR}")
    print(f"PDFs:             {len(all_pdfs)}")
    print(f"Chunk files:      {len(all_jsonl)} / {len(all_pdfs)} papers extracted")
    print(f"Total chunks:     {total_chunks}")
    print(f"Embedded files:   {len(already_embedded)} / {len(all_jsonl)}")
    print(f"Pending embed:    {len(pending)}")
    print(f"Vectors in DB:    {db_count}")
    print(f"DB path:          {DB_PATH}")
    print()


def cmd_show(args):
    """Print a chunk and its neighbors, identified by chunk id (<paper>.pdf::N)."""
    chunk_id = args.id
    source, sep, num = chunk_id.rpartition("::")
    if not sep or not num.isdigit():
        print(f"Error: id must look like '<paper>.pdf::N', got {chunk_id!r}", file=sys.stderr)
        sys.exit(1)
    target = int(num)

    stem = source[:-4] if source.endswith(".pdf") else source
    jsonl = CHUNKS_DIR / f"{stem}.jsonl"
    if not jsonl.exists():
        print(f"Error: chunk file not found: {jsonl}", file=sys.stderr)
        sys.exit(1)

    # Index this paper's chunks by their numeric suffix
    chunks: dict[int, str] = {}
    for line in jsonl.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        chunks[int(rec["id"].rpartition("::")[2])] = rec["content"]

    if target not in chunks:
        print(f"Error: chunk {target} not in {jsonl.name} "
              f"(paper has chunks {min(chunks)}–{max(chunks)})", file=sys.stderr)
        sys.exit(1)

    lo = max(min(chunks), target - args.context)
    hi = min(max(chunks), target + args.context)
    print(f"\nPaper: {source}")
    print(f"Chunks {lo}–{hi} (match: {target})\n")
    for n in range(lo, hi + 1):
        if n not in chunks:
            continue
        marker = "  ◀── match" if n == target else ""
        print(f"── {source}::{n}{marker} " + "─" * 20)
        print(chunks[n])
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="dblect corpus manager — extract, embed, search."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract chunks from PDFs")
    p_extract.add_argument("--no-resume", action="store_true",
                           help="Re-extract all PDFs even if chunks already exist")

    sub.add_parser("embed", help="Incrementally embed new chunk files into corpus.db")

    p_search = sub.add_parser("search", help="Semantic search over the corpus")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("--n", type=int, default=SEARCH_N, help="Number of results")
    p_search.add_argument("--full", action="store_true",
                          help="Print whole chunks instead of 400-char snippets")

    p_show = sub.add_parser("show", help="Print a chunk and its neighbors by id")
    p_show.add_argument("id", help="Chunk id from search output, e.g. 'Paper.pdf::3'")
    p_show.add_argument("--context", type=int, default=2,
                        help="Neighbor chunks to show on each side (default: 2)")

    sub.add_parser("status", help="Show corpus statistics")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "embed":
        cmd_embed(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
