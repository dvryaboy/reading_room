#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pymupdf>=1.24",
#   "pdfplumber>=0.11",
# ]
# ///
"""
extract_chunks.py

Extract and chunk text from a directory of academic PDFs into per-paper JSONL
files, ready for `llm embed-multi`.

Usually invoked via `corpus.py extract`; can also be run directly.

Usage (run from the repo root):
    # Extract from data/papers/ into data/paper-chunks/ (the defaults)
    ./scripts/extract_chunks.py

    # Or point at explicit directories
    ./scripts/extract_chunks.py --papers data/papers --out data/paper-chunks

    # Then embed with llm (point at the directory):
    llm embed-multi corpus -d data/corpus.db --format nl --input <(cat data/paper-chunks/*.jsonl) -m text-embedding-3-small

    # Or embed incrementally, one file at a time:
    for f in data/paper-chunks/*.jsonl; do
        llm embed-multi corpus -d data/corpus.db --format nl --input "$f" -m text-embedding-3-small
    done

    # Search:
    llm similar corpus -d data/corpus.db -c "incremental view maintenance"

    # Remove a paper from the corpus: delete its .jsonl and re-embed from scratch,
    # or use llm to delete by ID prefix (all chunks share the same filename stem).

Options:
    --papers DIR      Directory of PDF files (default: <repo>/data/papers)
    --out DIR         Output directory for per-paper JSONL files (default: <repo>/data/paper-chunks)
    --chunk-size N    Target chunk size in tokens (default: 512)
    --overlap N       Overlap between chunks in tokens (default: 100)
    --resume          Skip PDFs that already have a .jsonl in the output directory
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Rough token estimate: 1 token ≈ 3 chars for dense technical text.
# Conservative estimate avoids exceeding API limits.
CHARS_PER_TOKEN = 3

# Hard cap: OpenAI's limit is 8192 tokens; stay well under it.
CHUNK_HARD_CAP_CHARS = 6000  # ~2000 tokens of safety margin


def token_estimate(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def chunk_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Split text into overlapping chunks of approximately chunk_tokens each.
    Splits on sentence boundaries where possible.
    Enforces a hard character cap on every output chunk regardless of sentence length.
    """
    chunk_chars = min(chunk_tokens * CHARS_PER_TOKEN, CHUNK_HARD_CAP_CHARS)
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    # Split into sentences — handles "et al.", "Fig.", "i.e." reasonably well
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Hard-truncate any single sentence that exceeds the chunk cap on its own
    sentences = [s[:CHUNK_HARD_CAP_CHARS] if len(s) > CHUNK_HARD_CAP_CHARS else s
                 for s in sentences]

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        slen = len(sentence)
        if current_len + slen > chunk_chars and current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk[:CHUNK_HARD_CAP_CHARS])  # enforce cap on output
            # Backtrack for overlap: keep sentences from the end that fit
            overlap_buf = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap_chars:
                    break
                overlap_buf.insert(0, s)
                overlap_len += len(s)
            current = overlap_buf
            current_len = overlap_len

        current.append(sentence)
        current_len += slen

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk[:CHUNK_HARD_CAP_CHARS])

    return chunks


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

# Patterns that indicate noise lines to discard
NOISE_PATTERNS = [
    re.compile(r'^\s*\d+\s*$'),                          # bare page numbers
    re.compile(r'^\s*-\s*\d+\s*-\s*$'),                  # - N -
    re.compile(r'^(proceedings|vldb|sigmod|acm|ieee)', re.I),  # running headers
    re.compile(r'^\s*©\s*\d{4}'),                         # copyright lines
    re.compile(r'^\s*doi:\s*10\.\S+', re.I),              # DOI lines
    re.compile(r'^\s*https?://\S+\s*$'),                  # bare URLs
    re.compile(r'^\s*_{3,}\s*$'),                         # horizontal rules
]

# References section markers — discard everything after these
REFERENCES_MARKERS = re.compile(
    r'^\s*(references|bibliography|works cited)\s*$', re.I
)

# Hyphenated line-break rejoining: "con-\ntrols" → "controls"
HYPHEN_BREAK = re.compile(r'(\w)-\s*\n\s*(\w)')


def is_noise(line: str) -> bool:
    return any(p.match(line) for p in NOISE_PATTERNS)


def clean_lines(raw: str) -> str:
    """
    Rejoin hyphenated line breaks, strip noise lines, and stop at references.
    Works on the full text string from either extractor.
    """
    # Rejoin hyphenated breaks before splitting into lines
    raw = HYPHEN_BREAK.sub(r'\1\2', raw)

    lines = []
    hit_references = False
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if REFERENCES_MARKERS.match(line):
            hit_references = True
        if hit_references:
            continue
        if is_noise(line):
            continue
        lines.append(line)

    return " ".join(lines)


def extract_via_pdfplumber(pdf_path: Path) -> str | None:
    """
    Primary extractor using pdfplumber with explicit column detection.
    For two-column pages: splits at the vertical midpoint, extracts left
    column then right column top-to-bottom. Falls back to full-page
    extraction for single-column pages (title, abstract, etc).
    """
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_chars = 0
            page_texts = []

            for page in pdf.pages:
                width = page.width
                height = page.height
                mid = width / 2

                # Crop left and right halves
                left  = page.crop((0,    0, mid,   height))
                right = page.crop((mid,  0, width, height))

                left_text  = left.extract_text()  or ""
                right_text = right.extract_text() or ""
                full_text  = page.extract_text()  or ""

                # Heuristic: if both halves have substantial text and the
                # combined half-text is close to the full-page text in length,
                # it's two-column. Otherwise treat as single-column.
                half_len = len(left_text) + len(right_text)
                full_len = len(full_text)

                if (len(left_text) > 100 and len(right_text) > 100
                        and half_len > full_len * 0.7):
                    # Two-column: left then right
                    raw = left_text + "\n" + right_text
                else:
                    # Single-column or mixed: use full page
                    raw = full_text

                total_chars += len(raw)
                cleaned = clean_lines(raw)
                if cleaned:
                    page_texts.append(cleaned)

            if total_chars < 500:
                return None

            return " ".join(page_texts)

    except Exception as e:
        log.debug("  pdfplumber failed for %s: %s", pdf_path.name, e)
        return None


def extract_via_pymupdf(pdf_path: Path) -> str | None:
    """
    Fallback extractor using PyMuPDF. Less accurate for two-column layouts
    but handles PDFs that pdfplumber cannot open.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        log.warning("  Could not open %s: %s", pdf_path.name, e)
        return None

    total_chars = 0
    page_texts = []

    for page in doc:
        raw = page.get_text("text", sort=True)
        total_chars += len(raw)
        cleaned = clean_lines(raw)
        if cleaned:
            page_texts.append(cleaned)

    doc.close()

    if total_chars < 500:
        return None

    return " ".join(page_texts)


def extract_text(pdf_path: Path) -> str | None:
    """
    Extract clean text from a PDF.
    Uses pdfplumber (layout-aware, handles two-column papers) with PyMuPDF
    as fallback for PDFs pdfplumber cannot handle.
    Returns None if the PDF appears scanned or unreadable.
    """
    text = extract_via_pdfplumber(pdf_path)
    if text:
        return text

    log.debug("  pdfplumber yielded nothing for %s — trying PyMuPDF", pdf_path.name)
    text = extract_via_pymupdf(pdf_path)
    if text is None:
        log.warning("  %s appears scanned or empty — skipping", pdf_path.name)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract and chunk PDFs into per-paper JSONL files for llm embed-multi."
    )
    data_dir = Path(__file__).resolve().parent.parent / "data"
    parser.add_argument("--papers", default=str(data_dir / "papers"),
                        help="PDF directory")
    parser.add_argument("--out", default=str(data_dir / "paper-chunks"),
                        help="Output directory for JSONL files")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Target chunk size in tokens (default: 512)")
    parser.add_argument("--overlap", type=int, default=100,
                        help="Overlap between chunks in tokens (default: 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip PDFs that already have a .jsonl in the output directory")
    args = parser.parse_args()

    papers_dir = Path(args.papers)
    out_dir = Path(args.out)

    if not papers_dir.exists():
        log.error("Papers directory not found: %s", papers_dir)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(papers_dir.glob("*.pdf")) + sorted(papers_dir.glob("*.PDF"))
    if not pdfs:
        log.error("No PDFs found in %s", papers_dir)
        sys.exit(1)

    log.info("Found %d PDFs in %s", len(pdfs), papers_dir)
    log.info("Writing per-paper JSONL files to %s", out_dir)

    processed = 0
    skipped_resume = 0
    skipped_scanned = 0
    total_chunks = 0

    for i, pdf_path in enumerate(pdfs):
        out_file = out_dir / (pdf_path.stem + ".jsonl")

        if args.resume and out_file.exists():
            log.debug("Skipping already-processed: %s", pdf_path.name)
            skipped_resume += 1
            continue

        log.info("[%d/%d] %s", i + 1, len(pdfs), pdf_path.name)

        text = extract_text(pdf_path)
        if text is None:
            skipped_scanned += 1
            continue

        chunks = chunk_text(text, args.chunk_size, args.overlap)
        if not chunks:
            log.warning("  No chunks produced for %s", pdf_path.name)
            continue

        with out_file.open("w") as f:
            for j, chunk in enumerate(chunks):
                record = {
                    "id": f"{pdf_path.name}::{j}",
                    "content": chunk,
                }
                f.write(json.dumps(record) + "\n")

        total_chunks += len(chunks)
        processed += 1
        log.info("  → %d chunks (avg %d tokens)",
                 len(chunks), token_estimate(text) // max(len(chunks), 1))

    log.info("Done. Processed: %d | Skipped (resume): %d | Skipped (scanned/empty): %d | Total chunks: %d",
             processed, skipped_resume, skipped_scanned, total_chunks)
    log.info("Output directory: %s", out_dir)
    log.info("")
    log.info("Next steps:")
    log.info("  # Embed all at once:")
    log.info("  cat %s/*.jsonl | llm embed-multi corpus -d corpus.db --format nl -m text-embedding-3-small", out_dir)
    log.info("  # Search:")
    log.info("  llm similar corpus -d corpus.db -c 'your query here'")


if __name__ == "__main__":
    main()
