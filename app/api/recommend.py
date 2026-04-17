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
"""
import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

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
    row = (
        get_supabase()
        .table("guide_profile")
        .select("id")
        .eq("user_profile_id", user_profile_id)
        .maybe_single()
        .execute()
    ).data
    return row["id"] if row else None


def _stay_ids_from_host(host_id: str) -> list:
    """Return all stay.id rows for a given host_id."""
    rows = (
        get_supabase()
        .table("stay")
        .select("id")
        .eq("host_id", host_id)
        .execute()
    ).data or []
    return [r["id"] for r in rows]


def _tourist_id_from_user(user_profile_id: str) -> Optional[str]:
    """Return tourist_profile.id for a user, or None if not a tourist."""
    row = (
        get_supabase()
        .table("tourist_profile")
        .select("id")
        .eq("user_profile_id", user_profile_id)
        .maybe_single()
        .execute()
    ).data
    return row["id"] if row else None


def _host_id_from_user(user_profile_id: str) -> Optional[str]:
    """Return host_profile's user_profile_id (host_id) — same value, just confirms they're a host."""
    row = (
        get_supabase()
        .table("host_profile")
        .select("user_profile_id")
        .eq("user_profile_id", user_profile_id)
        .maybe_single()
        .execute()
    ).data
    return row["user_profile_id"] if row else None


# ═══════════════════════════════════════════════════════════════════════════════
# GUIDE WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/guide", status_code=202)
async def embed_guide(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: guide_profile | Events: INSERT, UPDATE
    What changed: any field on guide_profile itself
    (experience_years, rate_per_hour, avg_rating, active_level, city_id, is_available)
    """
    _verify(x_webhook_secret)
    guide_id = payload.record.get("id")
    if not guide_id:
        raise HTTPException(400, "record.id missing")
    ok = vector_service.upsert_guide_embedding(guide_id)
    return {"status": "ok" if ok else "not_found", "guide_id": guide_id}


@router.post("/embed/guide/by-specialization", status_code=202)
async def embed_guide_by_specialization(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: guide_specialization | Events: INSERT, DELETE
    What changed: guide added or removed a specialization
    """
    _verify(x_webhook_secret)
    # On DELETE, record may be the deleted row — use old_record if available
    rec = payload.record if payload.type != "DELETE" else (payload.old_record or payload.record)
    guide_id = rec.get("guide_profile_id")
    if not guide_id:
        raise HTTPException(400, "record.guide_profile_id missing")
    ok = vector_service.upsert_guide_embedding(guide_id)
    return {"status": "ok" if ok else "not_found", "guide_id": guide_id}


@router.post("/embed/guide/by-user", status_code=202)
async def embed_guide_by_user(
    payload: WebhookPayload,
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
    rec = payload.record if payload.type != "DELETE" else (payload.old_record or payload.record)
    user_profile_id = rec.get("user_profile_id")
    if not user_profile_id:
        raise HTTPException(400, "record.user_profile_id missing")

    guide_id = _guide_id_from_user(user_profile_id)
    if not guide_id:
        # This user is a tourist, not a guide — totally normal
        return {"status": "skipped", "reason": "not_a_guide"}

    ok = vector_service.upsert_guide_embedding(guide_id)
    return {"status": "ok" if ok else "failed", "guide_id": guide_id}


@router.post("/embed/guide/by-local-activity", status_code=202)
async def embed_guide_by_local_activity(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: local_activity | Events: INSERT, UPDATE, DELETE
    What changed: a guide added/removed/changed a local activity they offer
    Note: local_activity ALSO affects stays (via host_id) — register a second
    webhook on the same table pointing to /embed/stay/by-local-activity.
    """
    _verify(x_webhook_secret)
    rec = payload.record if payload.type != "DELETE" else (payload.old_record or payload.record)

    guide_id = rec.get("guide_id")
    if not guide_id:
        return {"status": "skipped", "reason": "no_guide_id_in_record"}

    ok = vector_service.upsert_guide_embedding(guide_id)
    return {"status": "ok" if ok else "not_found", "guide_id": guide_id}


# ═══════════════════════════════════════════════════════════════════════════════
# STAY WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/stay", status_code=202)
async def embed_stay(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: stay | Events: INSERT, UPDATE
    What changed: stay name, type, description, budget, price_per_night, city_id
    """
    _verify(x_webhook_secret)
    stay_id = payload.record.get("id")
    if not stay_id:
        raise HTTPException(400, "record.id missing")
    ok = vector_service.upsert_stay_embedding(stay_id)
    return {"status": "ok" if ok else "not_found", "stay_id": stay_id}


@router.post("/embed/stay/by-ambiance", status_code=202)
async def embed_stay_by_ambiance(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: stay_ambiance | Events: INSERT, DELETE
    What changed: stay added or removed an ambiance tag
    """
    _verify(x_webhook_secret)
    rec = payload.record if payload.type != "DELETE" else (payload.old_record or payload.record)
    stay_id = rec.get("stay_id")
    if not stay_id:
        raise HTTPException(400, "record.stay_id missing")
    ok = vector_service.upsert_stay_embedding(stay_id)
    return {"status": "ok" if ok else "not_found", "stay_id": stay_id}


@router.post("/embed/stay/by-suitable-for", status_code=202)
async def embed_stay_by_suitable_for(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: stay_suitable_for | Events: INSERT, DELETE
    What changed: stay added or removed a suitable_for tag
    """
    _verify(x_webhook_secret)
    rec = payload.record if payload.type != "DELETE" else (payload.old_record or payload.record)
    stay_id = rec.get("stay_id")
    if not stay_id:
        raise HTTPException(400, "record.stay_id missing")
    ok = vector_service.upsert_stay_embedding(stay_id)
    return {"status": "ok" if ok else "not_found", "stay_id": stay_id}


@router.post("/embed/stay/by-local-activity", status_code=202)
async def embed_stay_by_local_activity(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook (second registration on local_activity):
      Table: local_activity | Events: INSERT, UPDATE, DELETE
    What changed: a host added/removed/changed an activity — affects all their stays
    Note: this is a SECOND webhook row on local_activity in Supabase Dashboard.
    The first points to /embed/guide/by-local-activity.
    """
    _verify(x_webhook_secret)
    rec = payload.record if payload.type != "DELETE" else (payload.old_record or payload.record)

    host_id = rec.get("host_id")
    if not host_id:
        return {"status": "skipped", "reason": "no_host_id_in_record"}

    stay_ids = _stay_ids_from_host(host_id)
    if not stay_ids:
        return {"status": "skipped", "reason": "no_stays_for_host"}

    results = {}
    for sid in stay_ids:
        ok = vector_service.upsert_stay_embedding(sid)
        results[sid] = "ok" if ok else "not_found"

    return {"status": "ok", "stays_re_embedded": results}


@router.post("/embed/stay/by-host", status_code=202)
async def embed_stay_by_host(
    payload: WebhookPayload,
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
    old = payload.old_record or {}
    rec = payload.record
    if rec.get("avg_rating") == old.get("avg_rating"):
        return {"status": "skipped", "reason": "avg_rating_unchanged"}

    host_id = rec.get("user_profile_id")
    if not host_id:
        raise HTTPException(400, "record.user_profile_id missing")

    stay_ids = _stay_ids_from_host(host_id)
    if not stay_ids:
        return {"status": "skipped", "reason": "no_stays_for_host"}

    results = {}
    for sid in stay_ids:
        ok = vector_service.upsert_stay_embedding(sid)
        results[sid] = "ok" if ok else "not_found"

    return {"status": "ok", "stays_re_embedded": results}


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVITY WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/activity", status_code=202)
async def embed_activity(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: activity | Events: INSERT, UPDATE
    What changed: activity name, category, description, budget, difficulty_level, base_price
    """
    _verify(x_webhook_secret)
    activity_id = payload.record.get("id")
    if not activity_id:
        raise HTTPException(400, "record.id missing")
    ok = vector_service.upsert_activity_embedding(activity_id)
    return {"status": "ok" if ok else "not_found", "activity_id": activity_id}


@router.post("/embed/activity/by-suitable-for", status_code=202)
async def embed_activity_by_suitable_for(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: activity_suitable_for | Events: INSERT, DELETE
    What changed: activity added or removed a suitable_for tag
    """
    _verify(x_webhook_secret)
    rec = payload.record if payload.type != "DELETE" else (payload.old_record or payload.record)
    activity_id = rec.get("activity_id")
    if not activity_id:
        raise HTTPException(400, "record.activity_id missing")
    ok = vector_service.upsert_activity_embedding(activity_id)
    return {"status": "ok" if ok else "not_found", "activity_id": activity_id}


# ═══════════════════════════════════════════════════════════════════════════════
# TOURIST WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/tourist/invalidate", status_code=202)
async def invalidate_tourist(
    payload: WebhookPayload,
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
    rec = payload.record if payload.type != "DELETE" else (payload.old_record or payload.record)
    user_profile_id = rec.get("user_profile_id")
    if not user_profile_id:
        raise HTTPException(400, "record.user_profile_id missing")

    tourist_id = _tourist_id_from_user(user_profile_id)
    if not tourist_id:
        # This user is a guide, not a tourist — totally normal
        return {"status": "skipped", "reason": "not_a_tourist"}

    vector_service.invalidate_tourist_embedding(tourist_id)
    return {"status": "ok", "tourist_id": tourist_id}


@router.post("/embed/tourist/by-profile", status_code=202)
async def embed_tourist_by_profile(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Supabase webhook:
      Table: tourist_profile | Events: UPDATE
    What changed: travel_style, budget, or active_level on tourist_profile
    Action: invalidate svector — lazily recomputed on next /recommend call
    Skips re-embed if none of the embedding-relevant fields actually changed.
    """
    _verify(x_webhook_secret)
    tourist_id = payload.record.get("id")
    if not tourist_id:
        raise HTTPException(400, "record.id missing")

    old = payload.old_record or {}
    rec = payload.record
    relevant = ("travel_style", "budget", "active_level")
    if not any(rec.get(f) != old.get(f) for f in relevant):
        return {"status": "skipped", "reason": "no_embedding_fields_changed"}

    vector_service.invalidate_tourist_embedding(tourist_id)
    return {"status": "ok", "tourist_id": tourist_id}


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED: user_profile UPDATE affects both guides and tourists
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/user-profile/update", status_code=202)
async def embed_user_profile_update(
    payload: WebhookPayload,
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
      - If this user is a tourist → invalidate tourist svector
      - Skips if neither relevant field changed
    """
    _verify(x_webhook_secret)

    old = payload.old_record or {}
    rec = payload.record
    user_profile_id = rec.get("id")
    if not user_profile_id:
        raise HTTPException(400, "record.id missing")

    # Only act if embedding-relevant fields changed
    guide_fields   = ("first_name", "last_name", "profile_bio", "gender")
    tourist_fields = ("profile_bio",)

    results = {}

    # Guide side
    if any(rec.get(f) != old.get(f) for f in guide_fields):
        guide_id = _guide_id_from_user(user_profile_id)
        if guide_id:
            ok = vector_service.upsert_guide_embedding(guide_id)
            results["guide"] = "ok" if ok else "failed"
        else:
            results["guide"] = "skipped_not_a_guide"

    # Tourist side
    if any(rec.get(f) != old.get(f) for f in tourist_fields):
        tourist_id = _tourist_id_from_user(user_profile_id)
        if tourist_id:
            vector_service.invalidate_tourist_embedding(tourist_id)
            results["tourist"] = "invalidated"
        else:
            results["tourist"] = "skipped_not_a_tourist"

    if not results:
        return {"status": "skipped", "reason": "no_embedding_fields_changed"}

    return {"status": "ok", "results": results}


# ═══════════════════════════════════════════════════════════════════════════════
# DOC WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/embed/doc", status_code=202)
async def embed_doc(
    payload: WebhookPayload,
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
        old = payload.old_record or {}
        log.info("doc_source '%s' deleted — chunks removed by cascade", old.get("name"))
        return {"status": "ok", "event": "delete"}

    source = payload.record
    if not source.get("is_active", True):
        log.info("doc_source '%s' is inactive, skipping embed", source.get("name"))
        return {"status": "skipped", "reason": "inactive"}

    from scripts.embed_docs import embed_source
    ok = embed_source(source, force=True)
    return {"status": "ok" if ok else "failed", "doc": source.get("name")}


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
