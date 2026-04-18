"""
scripts/embed_all.py

Backfill embeddings for all existing Supabase rows.

Usage:
    # Embed everything (first run after migration)
    python -m scripts.embed_all

    # Only embed rows that are currently NULL (safe re-run after partial failure)
    python -m scripts.embed_all --only-nulls

    # Embed only specific entities (mix and match)
    python -m scripts.embed_all --guides
    python -m scripts.embed_all --stays
    python -m scripts.embed_all --activities
    python -m scripts.embed_all --tourists
    python -m scripts.embed_all --docs

    # Combine flags
    python -m scripts.embed_all --stays --tourists --only-nulls

Environment variables needed (same as .env):
    SUPABASE_URL, SUPABASE_SERVICE_KEY, EMBEDDING_MODEL (optional)
"""
import argparse
import logging
import sys
import time
from typing import Callable, List

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from app.core.database import get_supabase
from app.services.vector_service import (
    get_embedding_model,
    upsert_guide_embedding,
    upsert_stay_embedding,
    upsert_activity_embedding,
    upsert_tourist_embedding,
    upsert_doc_chunk_embedding,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Tuning ────────────────────────────────────────────────────────────────────
SLEEP_BETWEEN_ROWS = 0.15   # seconds between embed calls
MAX_RETRIES        = 3      # retry each failed row up to this many times
RETRY_BACKOFF      = 2.0    # sleep multiplier on each retry (0.15 → 0.30 → 0.60)


# ── Retry wrapper ─────────────────────────────────────────────────────────────

def _embed_with_retry(label: str, entity_id: str, upsert_fn: Callable) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = upsert_fn(entity_id)
            time.sleep(SLEEP_BETWEEN_ROWS)
            return bool(result)
        except Exception as e:
            wait = SLEEP_BETWEEN_ROWS * (RETRY_BACKOFF ** attempt)
            log.warning(
                "  %s %s — attempt %d/%d failed (%s). Retrying in %.1fs ...",
                label, entity_id[:8], attempt, MAX_RETRIES, e, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                log.error("  %s %s — all retries exhausted, skipping.", label, entity_id[:8])
                return False
    return False


# ── Backfill runner ───────────────────────────────────────────────────────────

def _backfill(label: str, ids: List[str], upsert_fn: Callable) -> None:
    if not ids:
        log.info("%s — no rows to embed, skipping.", label)
        return

    log.info("-- %s — embedding %d rows ...", label, len(ids))
    ok = fail = 0

    for i, entity_id in enumerate(ids, 1):
        success = _embed_with_retry(label, entity_id, upsert_fn)
        if success:
            ok += 1
        else:
            fail += 1
        if i % 10 == 0:
            log.info("   %d / %d done ...", i, len(ids))

    log.info("-- %s done — %d ok, %d failed.\n", label, ok, fail)


# ── Entity ID fetchers ────────────────────────────────────────────────────────

def _guide_ids(only_nulls: bool) -> List[str]:
    db = get_supabase()
    q = db.table("guide_profile").select("id")
    if only_nulls:
        q = q.is_("embedding", "null")
    return [r["id"] for r in q.execute().data or []]


def _stay_ids(only_nulls: bool) -> List[str]:
    db = get_supabase()
    q = db.table("stay").select("id")
    if only_nulls:
        q = q.is_("embedding", "null")
    return [r["id"] for r in q.execute().data or []]


def _activity_ids(only_nulls: bool) -> List[str]:
    db = get_supabase()
    q = db.table("activity").select("id")
    if only_nulls:
        q = q.is_("embedding", "null")
    return [r["id"] for r in q.execute().data or []]


def _tourist_ids(only_nulls: bool) -> List[str]:
    db = get_supabase()
    q = db.table("tourist_profile").select("id")
    if only_nulls:
        # t2g_embedding as proxy — if any of the 3 is NULL we recompute all
        q = q.is_("t2g_embedding", "null")
    return [r["id"] for r in q.execute().data or []]


def _doc_chunk_rows(only_nulls: bool) -> list:
    """Returns list of {id, content} dicts for doc embedding."""
    db = get_supabase()
    q = db.table("doc_chunk").select("id, content")
    if only_nulls:
        q = q.is_("embedding", "null")
    return q.execute().data or []


# ── Tourist wrapper ───────────────────────────────────────────────────────────

def _tourist_upsert(tourist_id: str) -> bool:
    """
    Writes all three tourist columns atomically:
      t2g_embedding (guide query vector)
      t2s_embedding (stay query vector)
      t2a_embedding (activity query vector)
    """
    try:
        upsert_tourist_embedding(tourist_id)
        return True
    except Exception as e:
        log.error("Tourist %s embed failed: %s", tourist_id[:8], e)
        return False


# ── Doc chunk embedder ────────────────────────────────────────────────────────

def _embed_docs(only_nulls: bool) -> None:
    """
    Embeds doc_chunk rows. Each chunk already has its content text
    stored in the DB — we read it and embed directly.
    This is separate from other entities because it takes content
    from the row itself rather than joining related tables.
    """
    chunks = _doc_chunk_rows(only_nulls)
    if not chunks:
        log.info("doc_chunks — no rows to embed, skipping.")
        return

    log.info("-- doc_chunks — embedding %d rows ...", len(chunks))
    ok = fail = 0

    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk["id"]
        content  = chunk.get("content", "").strip()

        if not content:
            log.warning("  doc_chunk %s has empty content, skipping.", chunk_id[:8])
            fail += 1
            continue

        # Build a closure so retry wrapper gets a callable with no args
        def _upsert(_id, _text=content):
            return upsert_doc_chunk_embedding(_id, _text)

        success = _embed_with_retry("doc_chunk", chunk_id, _upsert)
        if success:
            ok += 1
        else:
            fail += 1

        if i % 10 == 0:
            log.info("   %d / %d done ...", i, len(chunks))

    log.info("-- doc_chunks done — %d ok, %d failed.\n", ok, fail)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Yaloo AI — backfill embeddings")
    parser.add_argument("--only-nulls",  action="store_true",
                        help="Only embed rows where the embedding column is NULL. "
                             "Safe to re-run after partial failures.")
    parser.add_argument("--guides",      action="store_true", help="Embed guides only")
    parser.add_argument("--stays",       action="store_true", help="Embed stays only")
    parser.add_argument("--activities",  action="store_true", help="Embed activities only")
    parser.add_argument("--tourists",    action="store_true", help="Embed tourists only")
    parser.add_argument("--docs",        action="store_true",
                        help="Embed doc_chunks only — use this when you add new policy docs")
    args = parser.parse_args()

    # No specific flag → run all entities
    run_all = not any([args.guides, args.stays, args.activities, args.tourists, args.docs])

    log.info("Pre-loading embedding model ...")
    get_embedding_model()
    log.info("Model ready.\n")

    mode = "NULL rows only" if args.only_nulls else "all rows"
    log.info("Mode: %s\n", mode)

    if run_all or args.guides:
        _backfill("guides", _guide_ids(args.only_nulls), upsert_guide_embedding)

    if run_all or args.stays:
        _backfill("stays", _stay_ids(args.only_nulls), upsert_stay_embedding)

    if run_all or args.activities:
        _backfill("activities", _activity_ids(args.only_nulls), upsert_activity_embedding)

    if run_all or args.tourists:
        _backfill("tourists (t2g+t2s+t2a)", _tourist_ids(args.only_nulls), _tourist_upsert)

    if run_all or args.docs:
        _embed_docs(args.only_nulls)

    log.info("Backfill complete.")
    log.info("")
    log.info("Verify in Supabase SQL editor:")
    log.info("  SELECT 'guides',       COUNT(embedding)      FROM guide_profile")
    log.info("  UNION ALL SELECT 'stays',        COUNT(embedding)      FROM stay")
    log.info("  UNION ALL SELECT 'activities',   COUNT(embedding)      FROM activity")
    log.info("  UNION ALL SELECT 'tourist t2g',  COUNT(t2g_embedding)  FROM tourist_profile")
    log.info("  UNION ALL SELECT 'tourist t2s',  COUNT(t2s_embedding)  FROM tourist_profile")
    log.info("  UNION ALL SELECT 'tourist t2a',  COUNT(t2a_embedding)  FROM tourist_profile")
    log.info("  UNION ALL SELECT 'doc_chunks',   COUNT(embedding)      FROM doc_chunk;")

#uncomment below when u need to run embed_all.py

# def run_embed_all(only_nulls: bool = True) -> None:
#     """Callable entry point for programmatic use (e.g. startup hook)."""
#     log.info("Pre-loading embedding model ...")
#     get_embedding_model()
#     log.info("Model ready.\n")

#     mode = "NULL rows only" if only_nulls else "all rows"
#     log.info("Mode: %s\n", mode)

#     _backfill("guides",     _guide_ids(only_nulls),    upsert_guide_embedding)
#     _backfill("stays",      _stay_ids(only_nulls),     upsert_stay_embedding)
#     _backfill("activities", _activity_ids(only_nulls), upsert_activity_embedding)
#     _backfill("tourists (t2g+t2s+t2a)", _tourist_ids(only_nulls), _tourist_upsert)
#     _embed_docs(only_nulls)

#     log.info("Backfill complete.")

if __name__ == "__main__":
    main()
