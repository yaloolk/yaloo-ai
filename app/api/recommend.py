"""
app/api/recommend.py

All webhook endpoints + recommendation endpoint.

COMPLETE WEBHOOK LIST (17 registrations in Supabase Dashboard):

  GUIDE
    guide_profile        INSERT, UPDATE        → /embed/guide
    guide_specialization INSERT, DELETE        → /embed/guide/by-specialization
    user_interest        INSERT, DELETE        → /embed/guide/by-user
    user_language        INSERT, DELETE        → /embed/guide/by-user
    local_activity       INSERT, UPDATE, DELETE→ /embed/guide/by-local-activity
    user_profile         UPDATE               → /embed/user-profile/update   (also affects tourist)

  STAY
    stay                 INSERT, UPDATE        → /embed/stay
    stay_ambiance        INSERT, DELETE        → /embed/stay/by-ambiance
    stay_suitable_for    INSERT, DELETE        → /embed/stay/by-suitable-for
    local_activity       INSERT, UPDATE, DELETE→ /embed/stay/by-local-activity
    host_profile         UPDATE               → /embed/stay/by-host

  ACTIVITY
    activity             INSERT, UPDATE        → /embed/activity
    activity_suitable_for INSERT, DELETE       → /embed/activity/by-suitable-for

  TOURIST
    user_interest        INSERT, DELETE        → /embed/tourist/invalidate
    user_language        INSERT, DELETE        → /embed/tourist/invalidate
    tourist_profile      UPDATE               → /embed/tourist/by-profile
    user_profile         UPDATE               → /embed/user-profile/update   (also affects guide)

  DOCS
    doc_source           INSERT, UPDATE, DELETE→ /embed/doc

NOTE: user_interest, user_language, local_activity, and user_profile each need
TWO webhook registrations in Supabase (one per affected entity). Supabase
supports multiple webhooks on the same table — just add two rows.

BACKGROUND TASKS NOTE:
All embedding endpoints return 202 immediately and run the actual embedding
in a FastAPI BackgroundTask. This prevents Supabase webhook timeouts (max 10s)
from causing duplicate/retry re-embeds, since CPU embedding takes ~19 seconds.
Invalidation endpoints (tourist) are fast and also run in background for consistency.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, status

from app.core.config import get_settings
from app.core.database import get_supabase
from app.schemas.payloads import RecommendRequest, RecommendResponse, WebhookPayload
from app.services import rec_engine, vector_service

log = logging.getLogger(__name__)
router = APIRouter()


# ── Webhook secret verification ───────────────────────────────────────────────

def _verify(x_webhook_secret: Optional[str]) -> None:
    s = get_settings()
    if not s.supabase_webhook_secret:
        return  # dev mode — skip
    if x_webhook_secret != s.supabase_webhook_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook secret")


# ── Lookup helpers ────────────────────────────────────────────────────────────

def _guide_id_from_user(user_profile_id: str) -> Optional[str]:
    """Return guide_profile.id for a user, or None if not a guide."""
    try:
        res = (
            get_supabase()
            .table("guide_profile")
            .select("id")
            .eq("user_profile_id", user_profile_id)
            .maybe_single()
            .execute()
        )
        row = res.data if res else None
        return row["id"] if row else None
    except Exception as e:
        log.warning("_guide_id_from_user(%s) failed: %s", user_profile_id, e)
        return None


def _stay_ids_from_host(host_id: str) -> list:
    """Return all stay.id rows for a given host_id."""
    try:
        res = (
            get_supabase()
            .table("stay")
            .select("id")
            .eq("host_id", host_id)
            .execute()
        )
        rows = res.data if res else []
        return [r["id"] for r in (rows or [])]
    except Exception as e:
        log.warning("_stay_ids_from_host(%s) failed: %s", host_id, e)
        return []


def _tourist_id_from_user(user_profile_id: str) -> Optional[str]:
    """Return tourist_profile.id for a user, or None if not a tourist."""
    try:
        res = (
            get_supabase()
            .table("tourist_profile")
            .select("id")
            .eq("user_profile_id", user_profile_id)
            .maybe_single()
            .execute()
        )
        row = res.data if res else None
        return row["id"] if row else None
    except Exception as e:
        log.warning("_tourist_id_from_user(%s) failed: %s", user_profile_id, e)
        return None


def _host_id_from_user(user_profile_id: str) -> Optional[str]:
    """Return host_profile's user_profile_id (host_id) — same value, just confirms they're a host."""
    try:
        res = (
            get_supabase()
            .table("host_profile")
            .select("user_profile_id")
            .eq("user_profile_id", user_profile_id)
            .maybe_single()
            .execute()
        )
        row = res.data if res else None
        return row["user_profile_id"] if row else None
    except Exception as e:
        log.warning("_host_id_from_user(%s) failed: %s", user_profile_id, e)
        return None


# ── Background task helpers ───────────────────────────────────────────────────
# These wrap multi-step logic that needs to run in the background,
# since BackgroundTasks only accepts a single callable + args.

def _bg_embed_guide_by_user(user_profile_id: str) -> None:
    """Background: re-embed guide if this user is a guide."""
    guide_id = _guide_id_from_user(user_profile_id)
    if not guide_id:
        log.info("embed_guide_by_user: user %s is not a guide, skipping", user_profile_id)
        return
    vector_service.upsert_guide_embedding(guide_id)


def _bg_invalidate_tourist(user_profile_id: str) -> None:
    """Background: eagerly re-embed tourist vectors if this user is a tourist."""
    tourist_id = _tourist_id_from_user(user_profile_id)
    if not tourist_id:
        log.info("embed_tourist: user %s is not a tourist, skipping", user_profile_id)
        return
    vector_service.upsert_tourist_embedding(tourist_id)
    log.info("embed_tourist: re-embedded tourist %s (t2g, t2s, t2a)", tourist_id)


def _bg_embed_stays_by_host(host_id: str) -> None:
    """Background: re-embed all stays for a given host."""
    stay_ids = _stay_ids_from_host(host_id)
    if not stay_ids:
        log.info("embed_stays_by_host: no stays found for host %s", host_id)
        return
    for sid in stay_ids:
        vector_service.upsert_stay_embedding(sid)


def _bg_embed_user_profile_update(user_profile_id: str, rec: dict, old: dict) -> None:
    """
    Background: handle user_profile UPDATE for both guide and tourist sides.
    Only re-embeds/invalidates if the relevant fields actually changed.
    """
    guide_fields   = ("first_name", "last_name", "profile_bio", "gender")
    tourist_fields = ("profile_bio",)

    # Guide side
    if any(rec.get(f) != old.get(f) for f in guide_fields):
        guide_id = _guide_id_from_user(user_profile_id)
        if guide_id:
            vector_service.upsert_guide_embedding(guide_id)
            log.info("user_profile update: re-embedded guide %s", guide_id)
        else:
            log.info("user_profile update: user %s is not a guide", user_profile_id)

    # Tourist side
    if any(rec.get(f) != old.get(f) for f in tourist_fields):
        tourist_id = _tourist_id_from_user(user_profile_id)
        if tourist_id:
            vector_service.upsert_tourist_embedding(tourist_id)
            log.info("user_profile update: re-embedded tourist %s (t2g, t2s, t2a)", tourist_id)
        else:
            log.info("user_profile update: user %s is not a tourist", user_profile_id)


def _bg_embed_doc(source: dict) -> None:
    """Background: fetch, chunk, and embed a doc_source."""
    from scripts.embed_docs import embed_source
    embed_source(source, force=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GUIDE WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/guide", status_code=202)
async def embed_guide(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: guide_profile | Events: INSERT, UPDATE
    What changed: any embedding-relevant field on guide_profile itself
    (experience_years, rate_per_hour, avg_rating, active_level, city_id, is_available)

    For UPDATE events, skips re-embedding if none of the fields that actually
    contribute to the guide embedding changed. This prevents an infinite webhook
    loop when upsert_guide_embedding writes non-embedding fields (e.g. embedding
    columns) back to guide_profile and Supabase re-fires the UPDATE trigger.
    """
    _verify(x_webhook_secret)
    guide_id = payload.safe_record.get("id")
    if not guide_id:
        raise HTTPException(400, "record.id missing")

    # On UPDATE, only re-embed when an embedding-relevant field actually changed.
    # Columns that contribute to the guide embedding text (from guide_profile itself).
    # specializations / interests / languages / local_activities come via their own
    # junction-table webhooks (INSERT/DELETE only) so they don't need guarding here.
    if payload.type == "UPDATE":
        relevant = (
            "profile_bio",   # joined from user_profile via fetch_guide_row()
            "active_level",
        )
        old = payload.safe_old
        rec = payload.safe_record
        if not any(rec.get(f) != old.get(f) for f in relevant):
            log.info("embed_guide: no embedding-relevant fields changed for guide %s, skipping", guide_id)
            return {"status": "skipped", "reason": "no_embedding_fields_changed", "guide_id": guide_id}

    background_tasks.add_task(vector_service.upsert_guide_embedding, guide_id)
    return {"status": "accepted", "guide_id": guide_id}


@router.post("/embed/guide/by-specialization", status_code=202)
async def embed_guide_by_specialization(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: guide_specialization | Events: INSERT, DELETE
    What changed: guide added or removed a specialization
    """
    _verify(x_webhook_secret)
    rec = payload.safe_record
    guide_id = rec.get("guide_profile_id")
    if not guide_id:
        raise HTTPException(400, "record.guide_profile_id missing")
    background_tasks.add_task(vector_service.upsert_guide_embedding, guide_id)
    return {"status": "accepted", "guide_id": guide_id}


@router.post("/embed/guide/by-user", status_code=202)
async def embed_guide_by_user(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhooks (register BOTH separately in Dashboard):
      Table: user_interest | Events: INSERT, DELETE
      Table: user_language | Events: INSERT, DELETE
    What changed: guide added/removed an interest or language
    Note: these same tables also trigger /embed/tourist/invalidate — add that
    as a second webhook row for the same table in Supabase Dashboard.
    """
    _verify(x_webhook_secret)
    rec = payload.safe_record
    user_profile_id = rec.get("user_profile_id")
    if not user_profile_id:
        raise HTTPException(400, "record.user_profile_id missing")
    background_tasks.add_task(_bg_embed_guide_by_user, user_profile_id)
    return {"status": "accepted", "user_profile_id": user_profile_id}


@router.post("/embed/guide/by-local-activity", status_code=202)
async def embed_guide_by_local_activity(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: local_activity | Events: INSERT, UPDATE, DELETE
    What changed: a guide added/removed/changed a local activity they offer
    Note: local_activity ALSO affects stays (via host_id) — register a second
    webhook on the same table pointing to /embed/stay/by-local-activity.

    On UPDATE, skips re-embedding if none of the fields that contribute to the
    guide embedding text actually changed (e.g. only price or date updated).
    """
    _verify(x_webhook_secret)
    rec = payload.safe_record
    guide_id = rec.get("guide_id")
    if not guide_id:
        return {"status": "skipped", "reason": "no_guide_id_in_record"}

    if payload.type == "UPDATE":
        relevant = ("name", "category", "description")
        old = payload.safe_old
        if not any(rec.get(f) != old.get(f) for f in relevant):
            log.info("embed_guide/by-local-activity: no embedding-relevant fields changed, skipping guide %s", guide_id)
            return {"status": "skipped", "reason": "no_embedding_fields_changed", "guide_id": guide_id}

    background_tasks.add_task(vector_service.upsert_guide_embedding, guide_id)
    return {"status": "accepted", "guide_id": guide_id}


# ═══════════════════════════════════════════════════════════════════════════════
# STAY WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/stay", status_code=202)
async def embed_stay(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: stay | Events: INSERT, UPDATE
    What changed: stay description, budget, or type
    (suitable_for / ambiance / local_activities come via their own junction-table webhooks)

    On UPDATE, skips re-embedding if none of the fields that contribute to the
    stay embedding text actually changed, preventing loops when upsert_stay_embedding
    writes the embedding vector back to the stay table.
    """
    _verify(x_webhook_secret)
    stay_id = payload.safe_record.get("id")
    if not stay_id:
        raise HTTPException(400, "record.id missing")

    if payload.type == "UPDATE":
        relevant = (
            "description",
            "budget",
            "type",
        )
        old = payload.safe_old
        rec = payload.safe_record
        if not any(rec.get(f) != old.get(f) for f in relevant):
            log.info("embed_stay: no embedding-relevant fields changed for stay %s, skipping", stay_id)
            return {"status": "skipped", "reason": "no_embedding_fields_changed", "stay_id": stay_id}

    background_tasks.add_task(vector_service.upsert_stay_embedding, stay_id)
    return {"status": "accepted", "stay_id": stay_id}


@router.post("/embed/stay/by-ambiance", status_code=202)
async def embed_stay_by_ambiance(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: stay_ambiance | Events: INSERT, DELETE
    What changed: stay added or removed an ambiance tag
    """
    _verify(x_webhook_secret)
    rec = payload.safe_record
    stay_id = rec.get("stay_id")
    if not stay_id:
        raise HTTPException(400, "record.stay_id missing")
    background_tasks.add_task(vector_service.upsert_stay_embedding, stay_id)
    return {"status": "accepted", "stay_id": stay_id}


@router.post("/embed/stay/by-suitable-for", status_code=202)
async def embed_stay_by_suitable_for(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: stay_suitable_for | Events: INSERT, DELETE
    What changed: stay added or removed a suitable_for tag
    """
    _verify(x_webhook_secret)
    rec = payload.safe_record
    stay_id = rec.get("stay_id")
    if not stay_id:
        raise HTTPException(400, "record.stay_id missing")
    background_tasks.add_task(vector_service.upsert_stay_embedding, stay_id)
    return {"status": "accepted", "stay_id": stay_id}


@router.post("/embed/stay/by-local-activity", status_code=202)
async def embed_stay_by_local_activity(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook (second registration on local_activity):
      Table: local_activity | Events: INSERT, UPDATE, DELETE
    What changed: a host added/removed/changed an activity — affects all their stays
    Note: this is a SECOND webhook row on local_activity in Supabase Dashboard.
    The first points to /embed/guide/by-local-activity.

    On UPDATE, skips re-embedding if none of the fields that contribute to the
    stay embedding text actually changed (e.g. only price or date updated).
    """
    _verify(x_webhook_secret)
    rec = payload.safe_record
    host_id = rec.get("host_id")
    if not host_id:
        return {"status": "skipped", "reason": "no_host_id_in_record"}

    if payload.type == "UPDATE":
        relevant = ("name", "category", "description")
        old = payload.safe_old
        if not any(rec.get(f) != old.get(f) for f in relevant):
            log.info("embed_stay/by-local-activity: no embedding-relevant fields changed, skipping host %s", host_id)
            return {"status": "skipped", "reason": "no_embedding_fields_changed", "host_id": host_id}

    background_tasks.add_task(_bg_embed_stays_by_host, host_id)
    return {"status": "accepted", "host_id": host_id}


@router.post("/embed/stay/by-host", status_code=202)
async def embed_stay_by_host(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: host_profile | Events: UPDATE
    What changed: host avg_rating changed — re-embed all their stays
    avg_rating is joined from host_profile into stay text via fetch_stay_row()
    so a rating change must trigger stay re-embed.
    """
    _verify(x_webhook_secret)

    # Only re-embed if avg_rating actually changed
    old = payload.safe_old
    rec = payload.safe_record
    if rec.get("avg_rating") == old.get("avg_rating"):
        return {"status": "skipped", "reason": "avg_rating_unchanged"}

    host_id = rec.get("user_profile_id")
    if not host_id:
        raise HTTPException(400, "record.user_profile_id missing")

    background_tasks.add_task(_bg_embed_stays_by_host, host_id)
    return {"status": "accepted", "host_id": host_id}


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVITY WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/activity", status_code=202)
async def embed_activity(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: activity | Events: INSERT, UPDATE
    What changed: activity category, description, difficulty_level, or budget
    (suitable_for comes via its own junction-table webhook)

    On UPDATE, skips re-embedding if none of the fields that contribute to the
    activity embedding text actually changed, preventing loops when
    upsert_activity_embedding writes the embedding vector back to the activity table.
    """
    _verify(x_webhook_secret)
    activity_id = payload.safe_record.get("id")
    if not activity_id:
        raise HTTPException(400, "record.id missing")

    if payload.type == "UPDATE":
        relevant = (
            "category",
            "description",
            "difficulty_level",
            "budget",
        )
        old = payload.safe_old
        rec = payload.safe_record
        if not any(rec.get(f) != old.get(f) for f in relevant):
            log.info("embed_activity: no embedding-relevant fields changed for activity %s, skipping", activity_id)
            return {"status": "skipped", "reason": "no_embedding_fields_changed", "activity_id": activity_id}

    background_tasks.add_task(vector_service.upsert_activity_embedding, activity_id)
    return {"status": "accepted", "activity_id": activity_id}


@router.post("/embed/activity/by-suitable-for", status_code=202)
async def embed_activity_by_suitable_for(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: activity_suitable_for | Events: INSERT, DELETE
    What changed: activity added or removed a suitable_for tag
    """
    _verify(x_webhook_secret)
    rec = payload.safe_record
    activity_id = rec.get("activity_id")
    if not activity_id:
        raise HTTPException(400, "record.activity_id missing")
    background_tasks.add_task(vector_service.upsert_activity_embedding, activity_id)
    return {"status": "accepted", "activity_id": activity_id}


# ═══════════════════════════════════════════════════════════════════════════════
# TOURIST WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/tourist/invalidate", status_code=202)
async def invalidate_tourist(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhooks (register BOTH separately in Dashboard):
      Table: user_interest | Events: INSERT, DELETE
      Table: user_language | Events: INSERT, DELETE
    What changed: tourist added/removed an interest or language
    Action: null out t2g, t2s, t2a embeddings so next /recommend call recomputes it fresh
    Note: these same tables also trigger /embed/guide/by-user.
    """
    _verify(x_webhook_secret)
    rec = payload.safe_record
    user_profile_id = rec.get("user_profile_id")
    if not user_profile_id:
        raise HTTPException(400, "record.user_profile_id missing")
    background_tasks.add_task(_bg_invalidate_tourist, user_profile_id)
    return {"status": "accepted", "user_profile_id": user_profile_id}


@router.post("/embed/tourist/by-profile", status_code=202)
async def embed_tourist_by_profile(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: tourist_profile | Events: UPDATE
    What changed: travel_style, budget, or active_level on tourist_profile
    Action: invalidate vectors — lazily recomputed on next /recommend call
    Skips re-embed if none of the embedding-relevant fields actually changed.
    """
    _verify(x_webhook_secret)
    tourist_id = payload.safe_record.get("id")
    if not tourist_id:
        raise HTTPException(400, "record.id missing")

    old = payload.safe_old
    rec = payload.safe_record
    relevant = ("travel_style", "budget", "active_level")
    if not any(rec.get(f) != old.get(f) for f in relevant):
        return {"status": "skipped", "reason": "no_embedding_fields_changed"}

    background_tasks.add_task(vector_service.upsert_tourist_embedding, tourist_id)
    return {"status": "accepted", "tourist_id": tourist_id}


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED: user_profile UPDATE affects both guides and tourists
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/user-profile/update", status_code=202)
async def embed_user_profile_update(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: user_profile | Events: UPDATE
    What changed: first_name, last_name, profile_bio, or gender
    These fields are joined into BOTH guide and tourist embeddings via fetch_guide_row()
    and fetch_tourist_row(), so a change here must update both if applicable.

    Action:
      - If this user is a guide  → re-embed guide_profile
      - If this user is a tourist → invalidate tourist vectors
      - Skips if neither relevant field changed
    """
    _verify(x_webhook_secret)

    old = payload.safe_old
    rec = payload.safe_record
    user_profile_id = rec.get("id")
    if not user_profile_id:
        raise HTTPException(400, "record.id missing")

    # Check if any embedding-relevant field changed before queuing background work
    guide_fields   = ("first_name", "last_name", "profile_bio", "gender")
    tourist_fields = ("profile_bio",)
    guide_changed   = any(rec.get(f) != old.get(f) for f in guide_fields)
    tourist_changed = any(rec.get(f) != old.get(f) for f in tourist_fields)

    if not guide_changed and not tourist_changed:
        return {"status": "skipped", "reason": "no_embedding_fields_changed"}

    background_tasks.add_task(_bg_embed_user_profile_update, user_profile_id, rec, old)
    return {"status": "accepted", "user_profile_id": user_profile_id}


# ═══════════════════════════════════════════════════════════════════════════════
# DOC WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/doc", status_code=202)
async def embed_doc(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: doc_source | Events: INSERT, UPDATE, DELETE
    What changed:
      INSERT/UPDATE → fetch URL, re-chunk, re-embed, store in doc_chunk
      DELETE        → chunks removed automatically via ON DELETE CASCADE
    """
    _verify(x_webhook_secret)

    if payload.type == "DELETE":
        old = payload.safe_old
        log.info("doc_source '%s' deleted — chunks removed by cascade", old.get("name"))
        return {"status": "ok", "event": "delete"}

    source = payload.safe_record
    if not source.get("is_active", True):
        log.info("doc_source '%s' is inactive, skipping embed", source.get("name"))
        return {"status": "skipped", "reason": "inactive"}

    background_tasks.add_task(_bg_embed_doc, source)
    return {"status": "accepted", "doc": source.get("name")}


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENDPOINTS  (split by type for faster, targeted calls)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/recommend/guides", response_model=RecommendResponse)
async def recommend_guides(req: RecommendRequest):
    """
    Return only guide recommendations for the tourist.
    Use this when you only need guides — avoids computing stays & activities.
    """
    try:
        result = rec_engine.recommend_guides(
            tourist_id=req.tourist_id,
            city=req.city,
            guide_gender=req.guide_gender,
            top_k=req.top_k,
            available_guide_ids=req.available_guide_ids,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        log.exception("Guide recommendation failed for tourist %s", req.tourist_id)
        raise HTTPException(status_code=500, detail="Recommendation engine error")


@router.post("/recommend/stays", response_model=RecommendResponse)
async def recommend_stays(req: RecommendRequest):
    """
    Return only stay recommendations for the tourist.
    Use this when you only need stays — avoids computing guides & activities.
    """
    try:
        result = rec_engine.recommend_stays(
            tourist_id=req.tourist_id,
            city=req.city,
            top_k=req.top_k,
            available_stay_ids=req.available_stay_ids,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        log.exception("Stay recommendation failed for tourist %s", req.tourist_id)
        raise HTTPException(status_code=500, detail="Recommendation engine error")


@router.post("/recommend/activities", response_model=RecommendResponse)
async def recommend_activities(req: RecommendRequest):
    """
    Return only activity recommendations for the tourist.
    Use this when you only need activities — avoids computing guides & stays.
    """
    try:
        result = rec_engine.recommend_activities(
            tourist_id=req.tourist_id,
            city=req.city,
            top_k=req.top_k,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        log.exception("Activity recommendation failed for tourist %s", req.tourist_id)
        raise HTTPException(status_code=500, detail="Recommendation engine error")


@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(req: RecommendRequest):
    """
    Return all recommendations (guides + stays + activities) in one call.
    Kept for backward compatibility — prefer the split endpoints above
    when you only need one type.
    """
    try:
        result = rec_engine.recommend(
            tourist_id=req.tourist_id,
            city=req.city,
            guide_gender=req.guide_gender,
            top_k=req.top_k,
            available_guide_ids=req.available_guide_ids,
            available_stay_ids=req.available_stay_ids,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        log.exception("Recommendation failed for tourist %s", req.tourist_id)
        raise HTTPException(status_code=500, detail="Recommendation engine error")
