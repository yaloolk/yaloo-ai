"""
scripts/embed_docs.py

Fetches documents from URLs stored in doc_source table,
chunks them, embeds each chunk, and upserts to doc_chunk table.

Usage:
    cd yaloo_ai
    python -m scripts.embed_docs                  # embed all active docs
    python -m scripts.embed_docs --name faq       # re-embed one doc by name
    python -m scripts.embed_docs --force           # re-embed even if already embedded

Environment variables: same as .env
    SUPABASE_URL, SUPABASE_SERVICE_KEY, EMBEDDING_MODEL (optional)
"""
import argparse
import logging
import sys
import time
from typing import List, Optional

import httpx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from app.core.database import get_supabase
from app.services.vector_service import get_embedding_model, embed_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks to preserve context across boundaries


def _fetch_url(url: str) -> str:
    """Download text content from a URL. Supports .txt and .md files."""
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        log.error("Failed to fetch %s: %s", url, e)
        return ""


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    Tries to break at sentence boundaries ('. ') first,
    falls back to hard character split.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to find a sentence boundary near the end
        boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + size // 2:
            end = boundary + 1  # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap  # step back by overlap for context continuity

    return [c for c in chunks if len(c) > 50]  # discard tiny fragments


def _delete_existing_chunks(source_id: str) -> None:
    """Remove all existing chunks for a source before re-embedding."""
    get_supabase().table("doc_chunk").delete().eq("source_id", source_id).execute()
    log.info("  Deleted existing chunks for source %s", source_id)


def embed_source(source: dict, force: bool = False) -> bool:
    """
    Fetch, chunk, embed, and upsert one doc_source row.
    Returns True on success.
    """
    name = source["name"]
    url  = source["url"]
    sid  = source["id"]
    cat  = source["category"]

    log.info("Processing '%s' (%s) …", name, url)

    # Skip if already embedded and not forced
    if source.get("last_embedded_at") and not force:
        log.info("  Already embedded, skipping (use --force to re-embed)")
        return True

    # 1. Fetch document text
    text = _fetch_url(url)
    if not text:
        log.error("  Empty content from URL, skipping")
        return False

    # 2. Chunk
    chunks = _chunk_text(text)
    if not chunks:
        log.error("  No usable chunks produced, skipping")
        return False
    log.info("  %d chunks produced", len(chunks))

    # 3. Embed all chunks in one batch call (efficient)
    try:
        vectors = embed_batch(chunks)
    except Exception as e:
        log.error("  Embedding failed: %s", e)
        return False

    # 4. Delete old chunks for this source
    _delete_existing_chunks(sid)

    # 5. Upsert new chunks
    db = get_supabase()
    rows = [
        {
            "source_id": sid,
            "doc_name":  name,
            "category":  cat,
            "chunk_idx": i,
            "content":   chunk,
            "embedding": vec,
        }
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]

    # Insert in batches of 50 to avoid request size limits
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        db.table("doc_chunk").insert(rows[i:i + batch_size]).execute()
        log.info("  Inserted chunks %d–%d", i, min(i + batch_size, len(rows)) - 1)

    # 6. Update last_embedded_at timestamp
    db.table("doc_source").update(
        {"last_embedded_at": "now()"}
    ).eq("id", sid).execute()

    log.info("  '%s' done — %d chunks embedded", name, len(chunks))
    return True


def main():
    parser = argparse.ArgumentParser(description="Embed Yaloo docs from Supabase doc_source table")
    parser.add_argument("--name",  help="Embed only the doc with this name")
    parser.add_argument("--force", action="store_true", help="Re-embed even if already embedded")
    args = parser.parse_args()

    db = get_supabase()

    # Pre-warm embedding model
    log.info("Pre-loading embedding model …")
    get_embedding_model()
    log.info("Model ready.\n")

    # Fetch active sources
    query = db.table("doc_source").select("*").eq("is_active", True)
    if args.name:
        query = query.eq("name", args.name)
    sources = query.execute().data or []

    if not sources:
        log.warning("No active doc_source rows found. Add rows to doc_source table first.")
        return

    log.info("Found %d source(s) to process\n", len(sources))
    ok = fail = 0
    for source in sources:
        try:
            result = embed_source(source, force=args.force)
            if result:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            log.error("Unexpected error for '%s': %s", source.get("name"), e)
            fail += 1
        time.sleep(0.2)

    log.info("\nDone: %d ok, %d failed", ok, fail)


if __name__ == "__main__":
    main()
