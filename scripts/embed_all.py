"""
scripts/embed_all.py

Run this ONCE after the SQL migration to populate embeddings for all
existing rows in Supabase.

Usage:
    cd yaloo_ai
    python -m scripts.embed_all

Environment variables needed (same as .env):
    SUPABASE_URL, SUPABASE_SERVICE_KEY, EMBEDDING_MODEL (optional)
"""
import logging
import sys
import time
from typing import Callable, List

# Make sure app/ is on the path when running as script
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from app.core.database import get_supabase
from app.services.vector_service import (
    get_embedding_model,
    upsert_guide_embedding,
    upsert_stay_embedding,
    upsert_activity_embedding,
    upsert_tourist_embedding,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _backfill(
    label: str,
    ids: List[str],
    upsert_fn: Callable[[str], bool],
    delay: float = 0.1,
) -> None:
    log.info("Backfilling %d %s …", len(ids), label)
    ok = fail = 0
    for i, entity_id in enumerate(ids, 1):
        try:
            result = upsert_fn(entity_id)
            if result:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            log.error("  [%d/%d] %s FAILED: %s", i, len(ids), entity_id, e)
            fail += 1
        if i % 10 == 0:
            log.info("  %d / %d done …", i, len(ids))
        time.sleep(delay)   # be gentle on the embedding model + Supabase
    log.info("  %s done: %d ok, %d failed", label, ok, fail)


def main():
    db = get_supabase()

    # Pre-warm embedding model (downloads weights on first run)
    log.info("Pre-loading embedding model …")
    get_embedding_model()
    log.info("Model ready.\n")

    # --- Guides ---
    guide_ids = [r["id"] for r in db.table("guide_profile").select("id").execute().data or []]
    _backfill("guides", guide_ids, upsert_guide_embedding)

    # --- Stays ---
    stay_ids = [r["id"] for r in db.table("stay").select("id").execute().data or []]
    _backfill("stays", stay_ids, upsert_stay_embedding)

    # --- Global activities ---
    act_ids = [r["id"] for r in db.table("activity").select("id").execute().data or []]
    _backfill("activities", act_ids, upsert_activity_embedding)

    # --- Tourists (optional — they're computed lazily on first /recommend call) ---
    log.info("Tourist embeddings are computed lazily on first /recommend call.")
    log.info("To pre-warm all tourists, uncomment the block below.")
    # tourist_ids = [r["id"] for r in db.table("tourist_profile").select("id").execute().data or []]
    # def _tourist_upsert(tid):
    #     upsert_tourist_embedding(tid)
    #     return True
    # _backfill("tourists", tourist_ids, _tourist_upsert)

    log.info("\nBackfill complete.")


if __name__ == "__main__":
    main()
