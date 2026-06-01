#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.32",
# ]
# ///
"""
fetch_db_corpus.py

Two-phase DB/PL literature corpus builder.

Phase 1 (discover): Query Semantic Scholar for high-quality CS.DB papers,
                    write results to a JSONL manifest.
Phase 2 (download): Read the manifest, download PDFs from arxiv where
                    available, skip the rest gracefully.

No setup required. uv handles the virtualenv and dependencies automatically.

Paths default to <repo>/data/ (manifest.jsonl, papers/). Run from the repo root.

Usage:
    # Phase 1: discover papers, write manifest (-> data/manifest.jsonl)
    ./scripts/fetch_db_corpus.py discover

    # Phase 2: download PDFs listed in manifest (-> data/papers/)
    ./scripts/fetch_db_corpus.py download

    # Inspect what you have before downloading
    ./scripts/fetch_db_corpus.py stats

    # Or invoke explicitly via uv
    uv run --script scripts/fetch_db_corpus.py discover

    # Optional: provide S2 API key for higher rate limits (1 req/s vs shared pool)
    S2_API_KEY=your_key ./scripts/fetch_db_corpus.py discover
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

S2_BASE = "https://api.semanticscholar.org/graph/v1"

# Semantic Scholar field-of-study tags we care about
S2_FIELDS_OF_STUDY = ["Computer Science"]

# S2 paper fields to retrieve — keep minimal per their guidance
S2_PAPER_FIELDS = "paperId,title,year,citationCount,externalIds,abstract,authors,venue"

# Citation floor — papers below this are likely low-impact noise.
# Adjust down if you want more recent / less-cited work.
MIN_CITATIONS = 50

# Year floor — foundational DB work starts in the early 1970s (Codd, System R, INGRES)
MIN_YEAR = 1970

# ---------------------------------------------------------------------------
# Venue filtering
# ---------------------------------------------------------------------------
# S2 has no server-side venue filter, so we do it client-side.
# Strategy:
#   1. If venue matches the allowlist -> accept unconditionally.
#   2. If venue matches the blocklist -> reject.
#   3. If venue is empty/unknown     -> accept only if title+abstract contain
#                                       at least one DB/PL signal term.
#      (Handles legitimate preprints and workshop papers with no venue.)

# Canonical DB, PL, and systems venues. Substring match, case-insensitive.
VENUE_ALLOWLIST = [
    # VLDB family
    "vldb", "very large data base",
    # SIGMOD / PODS (including S2 data quality variant "sgmd")
    "sigmod", "sgmd", "pods", "principles of database",
    # ICDE
    "icde", "international conference on data engineering",
    # CIDR
    "cidr", "conference on innovative data systems",
    # EDBT
    "edbt", "extending database technology",
    # PVLDB / VLDBJ
    "proceedings of the vldb", "vldb journal",
    # SIGKDD (data mining overlap)
    "kdd", "knowledge discovery",
    # ICDT
    "icdt", "international conference on database theory",
    # WebDB / GRADES / aiDM workshop family
    "webdb", "grades", "aidm",
    # PL venues
    "popl", "pldi", "oopsla", "ecoop", "icfp", "esop", "lics",
    "principles of programming",
    "programming language design",
    "functional programming",
    # Systems venues
    "osdi", "sosp", "eurosys", "atc", "usenix annual",
    "nsdi", "fast", "systor",
    # Theoretical CS
    "stoc", "focs", "soda", "icalp",
    # arXiv itself
    "arxiv",
    # IEEE Data Engineering
    "ieee transactions on knowledge",
    "tkde",
    # ACM Transactions
    "tods", "transactions on database",
    "jacm", "journal of the acm",
    # Distributed systems
    "distributed computing", "disc ", "podc",
    # Logic and formal methods for databases
    "logics for databases", "information systems",
    # Computer architecture (relevant for storage, memory hierarchies)
    "international symposium on computer architecture", "isca",
    # Machine learning (relevant for learned components)
    "neural information processing", "neurips", "nips",
    # CS department technical reports (old foundational papers)
    "department of computer science", "computer science technical report",
    "technical report",
    # Springer LNCS (covers many DB/PL/theory workshops and proceedings)
    "lecture notes in computer science",
    # Knowledge engineering / expert systems overlap
    "knowledge-based systems",
    # Statistics and data analysis (relevant for data quality, approximate query)
    "computational statistics", "data analysis",
    # IEEE Fuzzy Systems (relevant for approximate/uncertain query processing)
    "ieee transactions on fuzzy",
]

# Venues that are definitively not relevant. Substring match, case-insensitive.
VENUE_BLOCKLIST = [
    "bioinformatics", "bmc", "genomics", "proteomics", "medical", "clinical",
    "chemistry", "chemical", "physics", "journal of computational physics",
    "sensors", "sensor network", "iot ", "internet of things",
    "environmental", "ecology", "ecological",
    "materials science", "metallurgy",
    "economics", "finance", "accounting",
    "education", "pedagogy", "teaching",
    "geography", "geospatial", "remote sensing",
    "astronomy", "astrophysics",
    "psychology", "neuroscience", "cognitive",
    "civil engineering", "mechanical engineering", "electrical engineering",
    "italian national",
    "journal of chemical",
    "plos ", "nature communications", "scientific reports",
]

# If venue is empty/unknown, require at least one of these in title or abstract.
DB_SIGNAL_TERMS = [
    "database", "query", "sql", "relational", "transaction", "datalog",
    "data warehouse", "olap", "oltp", "schema", "tuple", "relation",
    "index", "b-tree", "lsm", "storage engine", "buffer pool",
    "join", "aggregat", "view maintenance", "incremental",
    "type system", "refinement type", "property-based test",
    "data pipeline", "data quality", "data lineage", "provenance",
    "formal verification", "program analysis", "dataflow",
]


def venue_ok(venue: str, title: str, abstract: str) -> bool:
    """Return True if the paper should be kept based on venue heuristics."""
    v = venue.lower()
    t = (title + " " + abstract).lower()

    # Empty venue: fall back to content signal
    if not v:
        return any(term in t for term in DB_SIGNAL_TERMS)

    # Blocklist check first (fast rejection)
    if any(bad in v for bad in VENUE_BLOCKLIST):
        return False

    # Allowlist check
    if any(good in v for good in VENUE_ALLOWLIST):
        return True

    # Venue present but not on either list: require content signal
    return any(term in t for term in DB_SIGNAL_TERMS)

# arxiv API: 3 seconds between requests per their terms of use
ARXIV_DELAY = 3.5

# S2 API: 1 req/s with a key, ~1 req/3s without. We stay conservative.
S2_DELAY_UNAUTH = 3.5
S2_DELAY_AUTH = 1.1

# Bulk search returns up to 1000 results per query; paginate via token
S2_BULK_LIMIT = 500  # results per page, max 500

# Search queries — the S2 bulk search endpoint uses keyword matching.
# We fan out over several focused queries rather than one broad one,
# because S2 relevance ranking degrades on very generic queries.
SEARCH_QUERIES = [
    # Core DB systems
    "query optimization database",
    "query processing relational database",
    "transaction processing concurrency control",
    "database storage engine buffer management",
    "columnar storage analytical query",
    "database indexing B-tree LSM",
    "distributed database consensus",
    "OLAP OLTP hybrid database",
    "database join algorithms",
    "cost-based query optimizer",
    # Modern systems
    "vectorized query execution",
    "learned index structure database",
    "learned query optimizer",
    "in-memory database",
    "NewSQL distributed transaction",
    "database cracking adaptive indexing",
    "approximate query processing",
    "streaming query processing",
    "graph database query language",
    "time series database",
    # PL / type systems relevant to query languages
    "query language type system",
    "relational algebra formal semantics",
    "datalog recursive query",
    "SQL semantics formal verification",
    "language integrated query",
    # Storage / systems substrate
    "log structured merge tree",
    "write ahead log recovery",
    "MVCC multiversion concurrency",
    "disaggregated storage database",
    "database replication consistency",

    # Property-based testing
    "property-based testing",
    "property-based testing shrinking generator",
    "random testing program specification",
    "QuickCheck specification testing",
    # Refinement types and type-level constraints
    "refinement types dependent types",
    "liquid types type inference",
    "dependent type database query",
    "semantic types program verification",
    # Program analysis and dataflow
    "monotone dataflow analysis lattice",
    "abstract interpretation program analysis",
    "static analysis type system dataflow",
    "interprocedural dataflow analysis",
    # SQL / relational semantics -- null handling, bag semantics, aggregation
    "SQL null semantics three-valued logic",
    "bag semantics multiset relational algebra",
    "aggregation semantics SQL correctness",
    "SQL formal semantics provably correct",
    # Incremental computation and view maintenance
    "incremental view maintenance database",
    "incremental computation dataflow",
    "self-maintainable view materialized",
    "differential dataflow incremental",
    # Conservation, aggregation correctness, algebraic properties
    "aggregation correctness commutativity associativity",
    "double counting detection data pipeline",
    "cardinality estimation join fanout",
    "data lineage provenance query",
    # Random and metamorphic testing of database systems
    "SQLancer random testing database",
    "metamorphic testing database SQL",
    "fuzzing SQL query generator",
    "differential testing database system",
    # Data quality and pipeline testing
    "data quality validation pipeline",
    "data pipeline testing correctness",
    "data contract schema validation",
    "analytics pipeline data quality",
    # Change impact and schema evolution
    "schema evolution database migration",
    "breaking change impact analysis SQL",
    "semantic versioning data schema",
    # dbt and analytics engineering (sparse academically, but some exists)
    "dbt data transformation testing",
    "analytics engineering data modeling",
    "ELT pipeline correctness",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def s2_session(api_key: str | None) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "dblect-corpus-builder/0.1 (research tool)"})
    if api_key:
        sess.headers["x-api-key"] = api_key
    return sess


def s2_bulk_search(sess: requests.Session, query: str, delay: float) -> list[dict]:
    """
    Paginate through S2 bulk paper search for a single query.
    Returns list of raw paper dicts.
    """
    url = f"{S2_BASE}/paper/search/bulk"
    params = {
        "query": query,
        "fields": S2_PAPER_FIELDS,
        "fieldsOfStudy": ",".join(S2_FIELDS_OF_STUDY),
        "minCitationCount": MIN_CITATIONS,
        "year": f"{MIN_YEAR}-",
        "limit": S2_BULK_LIMIT,
    }
    results = []
    token = None
    page = 0

    while True:
        if token:
            params["token"] = token
        elif page > 0:
            break  # no more pages

        time.sleep(delay)
        try:
            resp = sess.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            log.warning("S2 request failed for query %r: %s", query, e)
            break

        if resp.status_code == 429:
            retry = int(resp.headers.get("retry-after", 60))
            log.warning("Rate limited. Sleeping %ds.", retry)
            time.sleep(retry)
            continue

        if resp.status_code != 200:
            log.warning("S2 returned %d for query %r: %s", resp.status_code, query, resp.text[:200])
            break

        data = resp.json()
        batch = data.get("data", [])
        results.extend(batch)
        token = data.get("token")
        page += 1

        log.info("  query=%r page=%d fetched=%d total_so_far=%d", query, page, len(batch), len(results))

        if not token or not batch:
            break

    return results


def normalize_paper(raw: dict) -> dict | None:
    """
    Normalize a S2 paper dict into our manifest format.
    Returns None if the paper should be filtered out.
    """
    pid = raw.get("paperId", "")
    title = (raw.get("title") or "").strip()
    year = raw.get("year") or 0
    citations = raw.get("citationCount") or 0
    abstract = (raw.get("abstract") or "").strip()
    venue = (raw.get("venue") or "").strip()
    external = raw.get("externalIds") or {}
    authors = [a.get("name", "") for a in (raw.get("authors") or [])]

    if not title or year < MIN_YEAR:
        return None

    if not venue_ok(venue, title, abstract):
        return None

    arxiv_id = external.get("ArXiv")  # e.g. "1803.00144"
    doi = external.get("DOI")

    return {
        "s2id": pid,
        "title": title,
        "year": year,
        "citations": citations,
        "venue": venue,
        "abstract": abstract,
        "authors": authors,
        "arxiv_id": arxiv_id,
        "doi": doi,
        # download status, filled in Phase 2
        "downloaded": False,
        "pdf_path": None,
    }


# ---------------------------------------------------------------------------
# Phase 1: discover
# ---------------------------------------------------------------------------

def cmd_discover(args):
    api_key = os.environ.get("S2_API_KEY") or args.api_key
    delay = S2_DELAY_AUTH if api_key else S2_DELAY_UNAUTH
    if not api_key:
        log.info("No S2_API_KEY found — using unauthenticated rate limit (%.1fs delay). "
                 "Set S2_API_KEY env var for 3x faster discovery.", delay)

    sess = s2_session(api_key)
    out_path = Path(args.out)

    # Load existing manifest to allow resuming
    seen_ids: set[str] = set()
    existing: list[dict] = []
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    p = json.loads(line)
                    seen_ids.add(p["s2id"])
                    existing.append(p)
        log.info("Resuming — loaded %d existing papers from %s", len(existing), out_path)

    new_papers: list[dict] = []

    for i, query in enumerate(SEARCH_QUERIES):
        log.info("[%d/%d] Searching: %r", i + 1, len(SEARCH_QUERIES), query)
        raw_results = s2_bulk_search(sess, query, delay)

        added = 0
        for raw in raw_results:
            paper = normalize_paper(raw)
            if paper is None:
                continue
            if paper["s2id"] in seen_ids:
                continue
            seen_ids.add(paper["s2id"])
            new_papers.append(paper)
            added += 1

        log.info("  → %d new unique papers (total new this run: %d)", added, len(new_papers))

    # Append new papers to manifest
    with out_path.open("a") as f:
        for p in new_papers:
            f.write(json.dumps(p) + "\n")

    total = len(existing) + len(new_papers)
    log.info("Done. Manifest: %s (%d total papers, %d newly added)", out_path, total, len(new_papers))
    log.info("Run `./fetch_db_corpus.py stats --manifest %s` to inspect before downloading.", out_path)


# ---------------------------------------------------------------------------
# Phase 2: download
# ---------------------------------------------------------------------------

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"

# Domains that consistently 403 or require browser sessions.
# Papers from these hosts are skipped; no point burning time on them.
OA_BLOCKED_DOMAINS = [
    "dl.acm.org",           # ACM DL requires institutional access
    "ieeexplore.ieee.org",  # IEEE requires institutional access
    "academic.oup.com",     # Oxford UP blocks scrapers
    "ncbi.nlm.nih.gov",     # PubMed/PMC blocks direct PDF scraping
    "link.springer.com",    # Springer blocks scrapers
    "onlinelibrary.wiley.com",
    "tandfonline.com",
    "sciencedirect.com",
    "journals.sagepub.com",
    "jstor.org",
]


def oa_url_blocked(url: str) -> bool:
    from urllib.parse import urlparse
    host = urlparse(url).netloc.lower().lstrip("www.")
    return any(blocked in host for blocked in OA_BLOCKED_DOMAINS)


def safe_filename(title: str, key: str, year: int) -> str:
    """Build a filesystem-safe PDF filename."""
    clean = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    clean = clean.strip().replace(" ", "_")[:60]
    safe_key = key.replace('/', '_').replace(':', '_')[:40]
    return f"{year}_{safe_key}_{clean}.pdf"


def download_url(url: str, dest: Path, sess: requests.Session, delay: float) -> bool:
    """Download a PDF from an arbitrary URL. Returns True on success.
    Retries once with https:// if the http:// attempt fails — handles
    cases where the server has moved but S2's stored URL wasn't updated.
    """
    urls_to_try = [url]
    if url.startswith("http://"):
        urls_to_try.append("https://" + url[7:])

    time.sleep(delay)
    for attempt_url in urls_to_try:
        try:
            resp = sess.get(attempt_url, timeout=60, allow_redirects=True)
        except requests.RequestException as e:
            log.debug("  Download attempt failed %s: %s", attempt_url, e)
            continue

        if resp.status_code == 429:
            retry = int(resp.headers.get("retry-after", 120))
            log.warning("  Rate limited. Sleeping %ds.", retry)
            time.sleep(retry)
            return False

        if resp.status_code != 200:
            log.debug("  HTTP %d for %s", resp.status_code, attempt_url)
            continue

        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and len(resp.content) < 10_000:
            log.warning("  Suspicious response (content-type=%s, size=%d) — skipping",
                        content_type, len(resp.content))
            continue

        if attempt_url != url:
            log.info("  (followed http→https redirect to %s)", attempt_url)
        dest.write_bytes(resp.content)
        return True

    log.warning("  Failed all attempts for %s", url)
    return False


def cmd_download(args):
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    papers = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    # Include papers with any download URL, sorted by source priority first, then citations.
    # Tier 0 = vldb.org, tier 1 = arxiv, tier 2 = oa_url — ensures vldb/arxiv papers
    # are attempted before open-access URLs regardless of citation count.
    def download_sort_key(p):
        if p.get("vldb_url"):
            tier = 0
        elif p.get("unpaywall_url"):
            tier = 1
        elif p.get("arxiv_id"):
            tier = 2
        else:
            tier = 3
        return (tier, -p.get("citations", 0))

    downloadable = sorted(
        (p for p in papers if (p.get("arxiv_id") or p.get("vldb_url") or p.get("unpaywall_url") or p.get("oa_url")) and not p.get("downloaded")),
        key=download_sort_key,
    )
    skipped = sum(1 for p in papers if not p.get("arxiv_id") and not p.get("vldb_url") and not p.get("unpaywall_url") and not p.get("oa_url"))
    already_done = sum(1 for p in papers if p.get("downloaded"))
    via_vldb = sum(1 for p in downloadable if p.get("vldb_url"))
    via_unpaywall = sum(1 for p in downloadable if not p.get("vldb_url") and p.get("unpaywall_url"))
    via_arxiv = sum(1 for p in downloadable if not p.get("vldb_url") and not p.get("unpaywall_url") and p.get("arxiv_id"))
    via_oa = sum(1 for p in downloadable if not p.get("vldb_url") and not p.get("unpaywall_url") and not p.get("arxiv_id") and p.get("oa_url"))
    log.info("Manifest: %d total | %d already downloaded | %d no URL",
             len(papers), already_done, skipped)
    log.info("To download: %d (%d vldb.org, %d unpaywall, %d arxiv, %d oa)",
             len(downloadable), via_vldb, via_unpaywall, via_arxiv, via_oa)

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "dblect-corpus-builder/0.1 (research tool; contact: your@email.com)"
    })

    paper_by_id = {p["s2id"]: p for p in papers}

    ok = 0
    fail = 0
    blocked = 0

    for i, paper in enumerate(downloadable):
        arxiv_id = paper.get("arxiv_id")
        oa_url = paper.get("oa_url", "")

        # Priority: vldb.org > unpaywall > arxiv > openAccessPdf
        vldb_url = paper.get("vldb_url", "")
        unpaywall_url = paper.get("unpaywall_url", "")
        if vldb_url:
            url = vldb_url
            key = paper["s2id"]
            delay = ARXIV_DELAY
            source = f"vldb:{vldb_url[:60]}"
        elif unpaywall_url:
            url = unpaywall_url
            key = paper["s2id"]
            delay = S2_DELAY_UNAUTH
            source = f"unpaywall:{unpaywall_url[:60]}"
        elif arxiv_id:
            url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
            key = arxiv_id
            delay = ARXIV_DELAY
            source = f"arxiv:{arxiv_id}"
        else:
            url = oa_url
            if oa_url_blocked(url):
                log.info("[%d/%d] Skipping blocked domain: %s", i + 1, len(downloadable), url[:80])
                blocked += 1
                continue
            key = paper["s2id"]
            delay = S2_DELAY_UNAUTH
            source = f"oa:{oa_url[:60]}"

        fname = safe_filename(paper["title"], key, paper["year"])
        dest = out_dir / fname

        if dest.exists():
            log.info("[%d/%d] Already exists: %s", i + 1, len(downloadable), fname)
            paper_by_id[paper["s2id"]]["downloaded"] = True
            paper_by_id[paper["s2id"]]["pdf_path"] = str(dest)
            ok += 1
            continue

        log.info("[%d/%d] %s — %s (%d)", i + 1, len(downloadable), source, paper["title"][:60], paper["year"])

        success = download_url(url, dest, sess, delay)

        if success:
            paper_by_id[paper["s2id"]]["downloaded"] = True
            paper_by_id[paper["s2id"]]["pdf_path"] = str(dest)
            ok += 1
            log.info("  ✓ saved %s (%.1f KB)", fname, dest.stat().st_size / 1024)
        else:
            fail += 1
            log.info("  ✗ failed — will retry on next run")

        if (i + 1) % 10 == 0:
            _write_manifest(manifest_path, list(paper_by_id.values()))
            log.info("  (manifest flushed)")

    _write_manifest(manifest_path, list(paper_by_id.values()))
    log.info("Done. Downloaded: %d | Failed: %d | Blocked domains: %d | No URL: %d",
             ok, fail, blocked, skipped)
    if skipped:
        log.info("Run `enrich` to attempt to find openAccessPdf URLs for the remaining papers.")


def _write_manifest(path: Path, papers: list[dict]):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def cmd_stats(args):
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    papers = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    total = len(papers)
    with_arxiv = sum(1 for p in papers if p.get("arxiv_id"))
    with_vldb_only = sum(1 for p in papers if not p.get("arxiv_id") and p.get("vldb_url"))
    with_unpaywall_only = sum(1 for p in papers if not p.get("arxiv_id") and not p.get("vldb_url") and p.get("unpaywall_url"))
    with_oa_only = sum(1 for p in papers if not p.get("arxiv_id") and not p.get("vldb_url") and not p.get("unpaywall_url") and p.get("oa_url"))
    with_any_url = with_arxiv + with_vldb_only + with_unpaywall_only + with_oa_only
    downloaded = sum(1 for p in papers if p.get("downloaded"))
    no_url = total - with_any_url

    # Year distribution
    from collections import Counter
    decade_counts = Counter((p["year"] // 10) * 10 for p in papers)
    # Venue distribution (top 15)
    venue_counts = Counter(p.get("venue", "unknown") or "unknown" for p in papers)

    print(f"\n=== Corpus Stats: {manifest_path} ===")
    print(f"Total papers:       {total}")
    print(f"With arxiv ID:      {with_arxiv} ({with_arxiv/total*100:.0f}%)")
    print(f"With vldb.org URL:  {with_vldb_only} ({with_vldb_only/total*100:.0f}%)")
    print(f"With Unpaywall URL: {with_unpaywall_only} ({with_unpaywall_only/total*100:.0f}%)")
    print(f"With OA URL only:   {with_oa_only} ({with_oa_only/total*100:.0f}%)")
    print(f"Any download URL:   {with_any_url} ({with_any_url/total*100:.0f}%)")
    print(f"Downloaded:         {downloaded}")
    print(f"No URL:             {no_url} (run enrich/unpaywall, then manual retrieval)")
    print(f"\nBy decade:")
    for decade in sorted(decade_counts):
        print(f"  {decade}s: {decade_counts[decade]}")
    print(f"\nTop venues:")
    for venue, count in venue_counts.most_common(15):
        print(f"  {count:4d}  {venue}")

    # Citation distribution
    citations = sorted(p.get("citations", 0) for p in papers)
    if citations:
        import statistics
        print(f"\nCitation counts:")
        print(f"  median: {statistics.median(citations):.0f}")
        print(f"  p75:    {citations[int(len(citations)*0.75)]}")
        print(f"  p90:    {citations[int(len(citations)*0.90)]}")
        print(f"  max:    {citations[-1]}")

    print()


# ---------------------------------------------------------------------------
# Filter (apply venue filter to existing manifest without re-querying S2)
# ---------------------------------------------------------------------------

def cmd_filter(args):
    from collections import Counter

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    papers = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    kept, dropped = [], []
    for p in papers:
        if venue_ok(p.get("venue") or "", p.get("title") or "", p.get("abstract") or ""):
            kept.append(p)
        else:
            dropped.append(p)

    # Per-venue summary of dropped papers: count and total citations
    dropped_venue_counts = Counter(p.get("venue") or "(no venue)" for p in dropped)
    dropped_venue_citations = {}
    for p in dropped:
        v = p.get("venue") or "(no venue)"
        dropped_venue_citations[v] = dropped_venue_citations.get(v, 0) + p.get("citations", 0)

    print(f"\n=== Filter results: {manifest_path} ===")
    print(f"Total:   {len(papers)}")
    print(f"Kept:    {len(kept)}")
    print(f"Dropped: {len(dropped)}")
    if dropped:
        print(f"\nDropped by venue (count | total citations | venue):")
        for venue, count in dropped_venue_counts.most_common():
            total_cites = dropped_venue_citations[venue]
            print(f"  {count:5d}  {total_cites:8d}  {venue}")

    if args.dry_run:
        print("\nDry run — no changes written.")
        return

    _write_manifest(manifest_path, kept)
    log.info("Manifest rewritten: %s", manifest_path)


# ---------------------------------------------------------------------------
# Enrich (fetch openAccessPdf from S2 for papers that have no arxiv_id)
# ---------------------------------------------------------------------------

S2_BATCH_URL = f"{S2_BASE}/paper/batch"
S2_BATCH_SIZE = 500  # S2 batch endpoint max


def cmd_enrich(args):
    api_key = os.environ.get("S2_API_KEY") or args.api_key
    delay = S2_DELAY_AUTH if api_key else S2_DELAY_UNAUTH
    if not api_key:
        log.info("No S2_API_KEY — using unauthenticated rate limit (%.1fs delay).", delay)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    papers = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    # Only enrich papers that have no download path yet
    targets = [p for p in papers if not p.get("arxiv_id") and not p.get("oa_url") and not p.get("downloaded")]
    log.info("Papers to enrich: %d (of %d total)", len(targets), len(papers))

    if not targets:
        log.info("Nothing to enrich.")
        return

    sess = s2_session(api_key)
    paper_by_id = {p["s2id"]: p for p in papers}

    found = 0
    # S2 batch endpoint accepts up to 500 IDs per POST
    for batch_start in range(0, len(targets), S2_BATCH_SIZE):
        batch = targets[batch_start: batch_start + S2_BATCH_SIZE]
        ids = [p["s2id"] for p in batch]

        log.info("Batch %d-%d / %d ...",
                 batch_start + 1, batch_start + len(batch), len(targets))

        time.sleep(delay)
        try:
            resp = sess.post(
                S2_BATCH_URL,
                params={"fields": "paperId,openAccessPdf,externalIds"},
                json={"ids": ids},
                timeout=30,
            )
        except requests.RequestException as e:
            log.warning("Batch request failed: %s", e)
            continue

        if resp.status_code == 429:
            retry = int(resp.headers.get("retry-after", 60))
            log.warning("Rate limited. Sleeping %ds.", retry)
            time.sleep(retry)
            continue

        if resp.status_code != 200:
            log.warning("S2 returned %d: %s", resp.status_code, resp.text[:200])
            continue

        for item in resp.json():
            if not item:
                continue
            pid = item.get("paperId")
            if pid not in paper_by_id:
                continue

            # Prefer openAccessPdf from S2
            oa = item.get("openAccessPdf") or {}
            oa_url = oa.get("url") or ""

            # Also pick up any arxiv ID or DBLP key S2 now knows about
            external = item.get("externalIds") or {}
            arxiv_id = external.get("ArXiv")
            dblp_key = external.get("DBLP")  # e.g. "conf/vldb/SeshadriLR96"

            p = paper_by_id[pid]
            updated = False

            if arxiv_id and not p.get("arxiv_id"):
                p["arxiv_id"] = arxiv_id
                updated = True

            if dblp_key and not p.get("dblp_key"):
                p["dblp_key"] = dblp_key
                updated = True

            if oa_url and not p.get("oa_url"):
                p["oa_url"] = oa_url
                updated = True

            if updated:
                found += 1

        # Flush every batch
        _write_manifest(manifest_path, list(paper_by_id.values()))
        log.info("  Running total newly enriched: %d", found)

    _write_manifest(manifest_path, list(paper_by_id.values()))

    with_oa = sum(1 for p in paper_by_id.values() if p.get("oa_url") or p.get("arxiv_id"))
    log.info("Done. Newly enriched: %d | Total with a download URL: %d", found, with_oa)
    log.info("Run download to fetch — it will use oa_url as fallback when arxiv_id is absent.")


# ---------------------------------------------------------------------------
# VLDB (look up vldb.org PDF URLs via DBLP for VLDB-venue papers)
# ---------------------------------------------------------------------------

DBLP_SEARCH_URL = "https://dblp.org/search/publ/api"
DBLP_DELAY = 4.0       # DBLP explicitly asks for polite crawling; 4s is safe
DBLP_BACKOFF_BASE = 30  # seconds to wait after a rate-limit or server error

# Venue substrings that indicate a VLDB-family paper worth looking up
VLDB_VENUE_HINTS = [
    "vldb", "very large data base", "pvldb",
    "proceedings of the vldb",
]


def is_vldb_paper(paper: dict) -> bool:
    venue = (paper.get("venue") or "").lower()
    return any(h in venue for h in VLDB_VENUE_HINTS)


def _extract_vldb_url_from_ee(ee) -> str | None:
    """Extract a vldb.org PDF URL from a DBLP ee field (str, list, or dict)."""
    if not ee:
        return None
    if isinstance(ee, str):
        candidates = [ee]
    elif isinstance(ee, list):
        candidates = [e if isinstance(e, str) else e.get("#text", "") for e in ee]
    elif isinstance(ee, dict):
        candidates = [ee.get("#text", ""), ee.get("@href", ""), ee.get("url", "")]
    else:
        candidates = []
    for url in candidates:
        if url and "vldb.org" in url and url.lower().endswith(".pdf"):
            return url
    return None


def dblp_lookup_by_key(dblp_key: str, sess: requests.Session) -> str | None:
    """
    Fetch a DBLP XML record by key and extract a vldb.org PDF URL from the ee element.
    Much more reliable than the search API for older papers.
    e.g. dblp_key = "conf/vldb/SeshadriLR96"
    """
    import xml.etree.ElementTree as ET
    url = f"https://dblp.org/rec/{dblp_key}.xml"
    for attempt in range(3):
        try:
            resp = sess.get(url, timeout=20)
        except requests.RequestException as e:
            wait = DBLP_BACKOFF_BASE * (attempt + 1)
            log.warning("  DBLP XML fetch failed (%s) — backing off %ds", e, wait)
            time.sleep(wait)
            continue

        if resp.status_code in (429, 503):
            wait = int(resp.headers.get("retry-after", DBLP_BACKOFF_BASE * (attempt + 1)))
            log.warning("  DBLP returned %d — backing off %ds", resp.status_code, wait)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            log.debug("  DBLP XML returned %d for %s", resp.status_code, url)
            return None

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            return None

        for ee in root.iter("ee"):
            text = ee.text or ""
            href = ee.get("href", "") or ee.get("url", "")
            for candidate in [text, href]:
                if candidate and "vldb.org" in candidate and candidate.lower().endswith(".pdf"):
                    return candidate
        return None  # record found but no vldb.org PDF URL

    return None


def dblp_lookup_by_title(title: str, authors: list[str], sess: requests.Session) -> str | None:
    """
    Fallback: search DBLP by title+author and return a vldb.org PDF URL if found.
    Less reliable than key-based lookup for older papers where ee is an HTML link.
    """
    first_author_surname = ""
    if authors:
        parts = authors[0].split()
        if parts:
            first_author_surname = parts[-1]

    query = f"{title} {first_author_surname}".strip()

    for attempt in range(3):
        try:
            resp = sess.get(
                DBLP_SEARCH_URL,
                params={"q": query, "format": "json", "h": 5},
                timeout=20,
            )
        except requests.RequestException as e:
            wait = DBLP_BACKOFF_BASE * (attempt + 1)
            log.warning("  DBLP search failed (%s) — backing off %ds", e, wait)
            time.sleep(wait)
            continue

        if resp.status_code in (429, 503):
            wait = int(resp.headers.get("retry-after", DBLP_BACKOFF_BASE * (attempt + 1)))
            log.warning("  DBLP returned %d — backing off %ds", resp.status_code, wait)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            return None

        try:
            data = resp.json()
        except ValueError:
            return None

        hits = (data.get("result", {})
                    .get("hits", {})
                    .get("hit", []))

        title_lower = title.lower().rstrip(".")

        for hit in hits:
            info = hit.get("info", {})
            hit_title = (info.get("title") or "").lower().rstrip(".")

            # Require a close title match (case-insensitive, ignoring trailing punctuation)
            if title_lower[:50] not in hit_title and hit_title[:50] not in title_lower:
                continue

            url = _extract_vldb_url_from_ee(info.get("ee"))
            if url:
                return url

            # JSON search API sometimes omits the PDF ee; try fetching the XML record
            dblp_key = info.get("key", "")
            if dblp_key:
                time.sleep(DBLP_DELAY)
                url = dblp_lookup_by_key(dblp_key, sess)
                if url:
                    return url

        return None  # got a response, just no match

    return None


def dblp_lookup(paper: dict, sess: requests.Session) -> str | None:
    """
    Look up a vldb.org PDF URL for a paper.
    Uses DBLP key (from S2 enrich) for direct XML lookup when available,
    falls back to title search otherwise.
    """
    dblp_key = paper.get("dblp_key", "")
    if dblp_key:
        url = dblp_lookup_by_key(dblp_key, sess)
        if url:
            return url

    # Fall back to title search
    return dblp_lookup_by_title(
        paper.get("title", ""),
        paper.get("authors") or [],
        sess,
    )


def cmd_vldb(args):
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    papers = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    # Target: VLDB-venue papers without a vldb_url yet, not already downloaded
    targets = [
        p for p in papers
        if is_vldb_paper(p) and not p.get("vldb_url") and not p.get("downloaded")
    ]
    log.info("VLDB papers to look up: %d (of %d total)", len(targets), len(papers))

    if not targets:
        log.info("Nothing to look up.")
        return

    sess = requests.Session()
    sess.headers.update({"User-Agent": "dblect-corpus-builder/0.1 (research tool)"})

    paper_by_id = {p["s2id"]: p for p in papers}
    found = 0

    for i, paper in enumerate(targets):
        time.sleep(DBLP_DELAY)
        url = dblp_lookup(paper, sess)

        if url:
            paper_by_id[paper["s2id"]]["vldb_url"] = url
            found += 1
            log.info("[%d/%d] ✓ %s", i + 1, len(targets), paper["title"][:70])
            log.info("        %s", url)
        else:
            log.info("[%d/%d] — not found: %s (%d)", i + 1, len(targets), paper["title"][:60], paper.get("year", 0))

        if (i + 1) % 20 == 0:
            _write_manifest(manifest_path, list(paper_by_id.values()))
            log.info("  (manifest flushed, %d found so far)", found)

    _write_manifest(manifest_path, list(paper_by_id.values()))
    log.info("Done. vldb_url found: %d / %d", found, len(targets))


# ---------------------------------------------------------------------------
# Unpaywall (find legal open-access PDFs by DOI for papers without vldb/arxiv)
# ---------------------------------------------------------------------------

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
UNPAYWALL_DELAY = 1.0  # Unpaywall asks for <=100k req/day; 1s is comfortable

# Domains that Unpaywall commonly returns but which block scrapers anyway.
# Don't bother storing these as unpaywall_url.
UNPAYWALL_BLOCKED_DOMAINS = OA_BLOCKED_DOMAINS + [
    "researchgate.net",   # blocks automated downloads
    "academia.edu",       # blocks automated downloads
]


def unpaywall_url_ok(url: str) -> bool:
    """Return True if the URL is worth attempting to download."""
    from urllib.parse import urlparse
    host = urlparse(url).netloc.lower().lstrip("www.")
    return not any(blocked in host for blocked in UNPAYWALL_BLOCKED_DOMAINS)


def cmd_unpaywall(args):
    email = args.email or os.environ.get("UNPAYWALL_EMAIL", "")
    if not email:
        log.error("Unpaywall requires an email address. Pass --email or set UNPAYWALL_EMAIL env var.")
        sys.exit(1)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    papers = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    # Target papers that have a DOI but no high-quality URL yet
    targets = [
        p for p in papers
        if p.get("doi")
        and not p.get("arxiv_id")
        and not p.get("vldb_url")
        and not p.get("unpaywall_url")
        and not p.get("downloaded")
    ]
    log.info("Papers to query Unpaywall: %d (of %d total)", len(targets), len(papers))

    if not targets:
        log.info("Nothing to enrich via Unpaywall.")
        return

    sess = requests.Session()
    sess.headers.update({"User-Agent": "dblect-corpus-builder/0.1 (research tool)"})

    paper_by_id = {p["s2id"]: p for p in papers}
    found = 0

    for i, paper in enumerate(targets):
        doi = paper["doi"]
        time.sleep(UNPAYWALL_DELAY)

        try:
            resp = sess.get(
                f"{UNPAYWALL_BASE}/{doi}",
                params={"email": email},
                timeout=15,
            )
        except requests.RequestException as e:
            log.warning("  Unpaywall request failed for %s: %s", doi, e)
            continue

        if resp.status_code == 404:
            log.debug("  DOI not found: %s", doi)
            continue

        if resp.status_code == 429:
            wait = int(resp.headers.get("retry-after", 60))
            log.warning("  Unpaywall rate limited. Sleeping %ds.", wait)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            log.warning("  Unpaywall returned %d for %s", resp.status_code, doi)
            continue

        try:
            data = resp.json()
        except ValueError:
            continue

        # Prefer the best OA location Unpaywall knows about — direct PDF only
        best = data.get("best_oa_location") or {}
        url = best.get("url_for_pdf") or ""

        # Fall back to scanning all OA locations for a direct PDF URL
        if not url:
            for loc in data.get("oa_locations") or []:
                candidate = loc.get("url_for_pdf") or ""
                if candidate:
                    url = candidate
                    break

        if not url:
            log.debug("  No direct PDF URL found for %s (may have landing page only)", doi)
            continue

        if not unpaywall_url_ok(url):
            log.debug("  Blocked domain for %s: %s", doi, url[:60])
            continue

        paper_by_id[paper["s2id"]]["unpaywall_url"] = url
        found += 1
        log.info("[%d/%d] ✓ %s", i + 1, len(targets), paper["title"][:70])
        log.info("        %s", url)

        if (i + 1) % 50 == 0:
            _write_manifest(manifest_path, list(paper_by_id.values()))
            log.info("  (manifest flushed, %d found so far)", found)

    _write_manifest(manifest_path, list(paper_by_id.values()))
    log.info("Done. unpaywall_url found: %d / %d", found, len(targets))



# ---------------------------------------------------------------------------
# Add (register a manually obtained PDF into the manifest)
# ---------------------------------------------------------------------------

def cmd_add(args):
    manifest_path = Path(args.manifest)
    pdf_src = Path(args.pdf)
    papers_dir = Path(args.papers_dir)
    s2id = args.s2id.strip()

    if not pdf_src.exists():
        log.error("PDF not found: %s", pdf_src)
        sys.exit(1)

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    papers_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest and find the entry
    papers = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    paper_by_id = {p["s2id"]: p for p in papers}

    if s2id not in paper_by_id:
        log.error("S2 ID %r not found in manifest.", s2id)
        log.error("If this is a new paper not in the manifest, use --new to add it.")
        sys.exit(1)

    paper = paper_by_id[s2id]

    # Build destination filename using the same convention as download
    dest_fname = safe_filename(paper["title"], s2id[:8], paper.get("year", 0))
    dest = papers_dir / dest_fname

    if dest.exists() and not args.force:
        log.error("Destination already exists: %s", dest)
        log.error("Use --force to overwrite.")
        sys.exit(1)

    import shutil
    shutil.copy2(str(pdf_src), str(dest))
    log.info("Copied %s → %s", pdf_src.name, dest)

    paper["downloaded"] = True
    paper["pdf_path"] = str(dest)

    _write_manifest(manifest_path, papers)
    log.info("Manifest updated: %s", manifest_path)
    log.info("Title:  %s", paper["title"][:80])
    log.info("Run `./corpus.py extract && ./corpus.py embed` to add to search index.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DB/PL corpus builder: discover via Semantic Scholar, download from arxiv."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    data_dir = Path(__file__).resolve().parent.parent / "data"

    p_discover = sub.add_parser("discover", help="Query S2 and build manifest")
    p_discover.add_argument("--out", default=str(data_dir / "manifest.jsonl"), help="Output manifest path")
    p_discover.add_argument("--api-key", default=None, help="S2 API key (or set S2_API_KEY env)")

    p_download = sub.add_parser("download", help="Download PDFs from arxiv using manifest")
    p_download.add_argument("--manifest", default=str(data_dir / "manifest.jsonl"))
    p_download.add_argument("--out", default=str(data_dir / "papers"), help="Output directory for PDFs")

    p_stats = sub.add_parser("stats", help="Print manifest statistics")
    p_stats.add_argument("--manifest", default=str(data_dir / "manifest.jsonl"))

    p_filter = sub.add_parser("filter", help="Apply venue filter to existing manifest (no S2 calls)")
    p_filter.add_argument("--manifest", default=str(data_dir / "manifest.jsonl"))
    p_filter.add_argument("--dry-run", action="store_true", help="Show what would be dropped without writing")

    p_enrich = sub.add_parser("enrich", help="Fetch openAccessPdf URLs from S2 for non-arxiv papers")
    p_enrich.add_argument("--manifest", default=str(data_dir / "manifest.jsonl"))
    p_enrich.add_argument("--api-key", default=None, help="S2 API key (or set S2_API_KEY env)")

    p_vldb = sub.add_parser("vldb", help="Look up vldb.org PDF URLs via DBLP for VLDB-venue papers")
    p_vldb.add_argument("--manifest", default=str(data_dir / "manifest.jsonl"))

    p_unpaywall = sub.add_parser("unpaywall", help="Find legal OA PDFs via Unpaywall for papers with a DOI")
    p_unpaywall.add_argument("--manifest", default=str(data_dir / "manifest.jsonl"))
    p_unpaywall.add_argument("--email", default=None, help="Email for Unpaywall API (or set UNPAYWALL_EMAIL env)")

    p_add = sub.add_parser("add", help="Register a manually obtained PDF into the manifest")
    p_add.add_argument("--manifest", default=str(data_dir / "manifest.jsonl"))
    p_add.add_argument("--pdf", required=True, help="Path to the PDF file to add")
    p_add.add_argument("--s2id", required=True, help="Semantic Scholar paper ID from the manifest")
    p_add.add_argument("--papers-dir", default=str(data_dir / "papers"), help="Papers directory (default: <repo>/data/papers)")
    p_add.add_argument("--force", action="store_true", help="Overwrite if destination already exists")

    args = parser.parse_args()

    if args.command == "discover":
        cmd_discover(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "filter":
        cmd_filter(args)
    elif args.command == "enrich":
        cmd_enrich(args)
    elif args.command == "vldb":
        cmd_vldb(args)
    elif args.command == "unpaywall":
        cmd_unpaywall(args)
    elif args.command == "add":
        cmd_add(args)


if __name__ == "__main__":
    main()
