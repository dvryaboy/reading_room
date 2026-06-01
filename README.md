# reading_room

A semantic-search corpus over the database & programming-languages research
literature. It discovers high-impact papers via Semantic Scholar, downloads the
open-access PDFs, chunks and embeds them, and lets you search the corpus by
meaning rather than keywords.

## Pipeline

```
fetch_db_corpus.py        extract_chunks.py        corpus.py embed         corpus.py / search.py
   discover ──► manifest ──► download ──► PDFs ──► JSONL chunks ──► embeddings ──► semantic search
                (S2 API)      (arxiv/OA)           (PyMuPDF)        (OpenAI + llm)
```

Each stage writes into `data/` (git-ignored):

| Artifact              | Path                     | Produced by                  |
| --------------------- | ------------------------ | ---------------------------- |
| Paper manifest        | `data/manifest.jsonl`    | `fetch_db_corpus.py discover`|
| Downloaded PDFs        | `data/papers/`           | `fetch_db_corpus.py download`|
| Per-paper text chunks | `data/paper-chunks/`     | `extract_chunks.py`          |
| Embedding store       | `data/corpus.db`         | `corpus.py embed` (`llm`)    |
| Embed tracking log    | `data/.embedded_files`   | `corpus.py embed`            |

## Layout

```
reading_room/
├── scripts/              # CLI tools (self-contained uv scripts, PEP 723 inline deps)
│   ├── fetch_db_corpus.py   # discover + download papers from Semantic Scholar / arxiv
│   ├── extract_chunks.py    # PDFs -> per-paper JSONL chunks
│   ├── corpus.py            # orchestrator: extract / embed / search / status
│   └── search.py            # standalone semantic search over the embeddings
└── data/                # all large/generated artifacts (git-ignored)
```

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) — runs the scripts and manages their
  dependencies automatically; no manual virtualenv or `pip install` needed.
- [`llm`](https://llm.datasette.com/) — used by `corpus.py embed` to write
  embeddings into `corpus.db` (`uv tool install llm` or `pipx install llm`).
- `OPENAI_API_KEY` — required for `embed` and `search` (embeddings).
- `S2_API_KEY` *(optional)* — higher Semantic Scholar rate limits during `discover`.

## Quickstart

Run everything from the repo root. Each script resolves `data/` relative to its
own location, so paths work regardless of your current directory.

```sh
export OPENAI_API_KEY=sk-...

# 1. Discover papers and build the manifest
./scripts/fetch_db_corpus.py discover

# 2. Download the open-access PDFs
./scripts/fetch_db_corpus.py download

# 3. Extract text chunks from the PDFs
./scripts/corpus.py extract

# 4. Embed any chunks not yet in corpus.db (incremental)
./scripts/corpus.py embed

# 5. Search
./scripts/corpus.py search "incremental view maintenance self-maintainability"
./scripts/corpus.py search --n 20 "property-based testing generator shrinking"

# Corpus stats at any time
./scripts/corpus.py status
```

Adding more papers later: drop PDFs into `data/papers/` (or run `fetch` again),
then re-run `extract` and `embed` — both are incremental and only process new
files.

## Configuration

- `CORPUS_DIR` — point `corpus.py` at a different data directory (defaults to
  `data/`). `search.py` and `extract_chunks.py` accept `--db`, `--chunks`,
  `--papers`, and `--out` flags to override individual paths.
- All scripts are standalone; you can also invoke them explicitly, e.g.
  `uv run --script scripts/corpus.py status`.
