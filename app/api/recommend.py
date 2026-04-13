"""
app/api/recommend.py

Endpoints:
  POST /embed/guide                — Supabase DB webhook (guide_profile changed)
  POST /embed/stay                 — Supabase DB webhook (stay changed)
  POST /embed/activity             — Supabase DB webhook (global activity changed)
  POST /embed/local-activity       — Supabase DB webhook (local_activity inserted/deleted)
  POST /embed/tourist/invalidate   — Supabase DB webhook (interest/language changed)
  POST /recommend                  — Mobile app calls this
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

def _verify_webhook(x_webhook_secret: Optional[str]) -> None:
    s = get_settings()
    if not s.supabase_webhook_secret:
        return
    if x_webhook_secret != s.supabase_webhook_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook secret")


# ── Webhook: guide re-embed ───────────────────────────────────────────────────

@router.post("/embed/guide", status_code=202)
async def embed_guide(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    _verify_webhook(x_webhook_secret)
    guide_id = payload.record.get("id")
    if not guide_id:
        raise HTTPException(400, "record.id missing")
    ok = vector_service.upsert_guide_embedding(guide_id)
    return {"status": "ok" if ok else "not_found", "guide_id": guide_id}


# ── Webhook: stay re-embed ────────────────────────────────────────────────────

@router.post("/embed/stay", status_code=202)
async def embed_stay(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    _verify_webhook(x_webhook_secret)
    stay_id = payload.record.get("id")
    if not stay_id:
        raise HTTPException(400, "record.id missing")
    ok = vector_service.upsert_stay_embedding(stay_id)
    return {"status": "ok" if ok else "not_found", "stay_id": stay_id}


# ── Webhook: global activity re-embed ────────────────────────────────────────

@router.post("/embed/activity", status_code=202)
async def embed_activity(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    _verify_webhook(x_webhook_secret)
    activity_id = payload.record.get("id")
    if not activity_id:
        raise HTTPException(400, "record.id missing")
    ok = vector_service.upsert_activity_embedding(activity_id)
    return {"status": "ok" if ok else "not_found", "activity_id": activity_id}


# ── Webhook: local_activity inserted or deleted ───────────────────────────────

@router.post("/embed/local-activity", status_code=202)
async def embed_local_activity(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Triggered on INSERT or DELETE of a local_activity row.

    A local_activity belongs to either a guide OR a host (never both).
    When it changes, the embedding of the owning guide or stay becomes
    stale — it was built without this activity — so we re-embed it.

    For DELETE: Supabase puts the deleted row in payload.old_record.
    For INSERT: the new row is in payload.record.
    """
    _verify_webhook(x_webhook_secret)

    # For DELETE events Supabase sends the deleted row in old_record
    row = payload.old_record if payload.type == "DELETE" else payload.record
    if not row:
        raise HTTPException(400, "no record data in payload")

    guide_id = row.get("guide_id")
    host_id  = row.get("host_id")
    results  = {}

    if guide_id:
        # Re-embed the guide whose local activity list just changed
        ok = vector_service.upsert_guide_embedding(guide_id)
        results["guide_id"] = guide_id
        results["guide_status"] = "ok" if ok else "not_found"

    if host_id:
        # Find the stay owned by this host and re-embed it
        stay_rows = (
            get_supabase()
            .table("stay")
            .select("id")
            .eq("host_id", host_id)
            .execute()
        ).data or []

        stay_statuses = []
        for stay in stay_rows:
            ok = vector_service.upsert_stay_embedding(stay["id"])
            stay_statuses.append({"stay_id": stay["id"], "status": "ok" if ok else "not_found"})
        results["host_id"] = host_id
        results["stays"] = stay_statuses

    if not guide_id and not host_id:
        log.warning("local_activity webhook: neither guide_id nor host_id in record")
        return {"status": "skipped", "reason": "no guide_id or host_id"}

    return {"status": "ok", "event": payload.type, **results}


# ── Webhook: tourist svector invalidation ────────────────────────────────────

@router.post("/embed/tourist/invalidate", status_code=202)
async def invalidate_tourist(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    """
    Triggered when user_interest or user_language changes.
    Nulls the cached svector so next /recommend call recomputes it.
    """
    _verify_webhook(x_webhook_secret)
    user_profile_id = payload.record.get("user_profile_id")
    if not user_profile_id:
        raise HTTPException(400, "record.user_profile_id missing")

    tp = (
        get_supabase()
        .table("tourist_profile")
        .select("id")
        .eq("user_profile_id", user_profile_id)
        .maybe_single()
        .execute()
    ).data
    if tp:
        vector_service.invalidate_tourist_embedding(tp["id"])
    return {"status": "ok", "user_profile_id": user_profile_id}


# ── Recommendation endpoint ───────────────────────────────────────────────────

@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(req: RecommendRequest):
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
    except Exception as e:
        log.exception("Recommendation failed for tourist %s", req.tourist_id)
        raise HTTPException(status_code=500, detail="Recommendation engine error")

