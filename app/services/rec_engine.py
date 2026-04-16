"""
app/services/rec_engine.py

Recommendation engine — pgvector edition, typed-RPC-only.

Tourist embedding columns (three separate vectors):
  t2g_embedding  → used when querying guides
  t2s_embedding  → used when querying stays
  t2a_embedding  → used when querying activities

All three are stored in tourist_profile and read together on each
/recommend call. If any is NULL the full set is recomputed atomically.

Scoring:
  Guides     : final = vec_sim                          (pure cosine)
  Stays      : final = 0.80*vec_sim + 0.12*budget_bonus
  Activities : final = 0.80*vec_sim + 0.12*budget_bonus + 0.08*active_bonus
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.database import get_supabase
from app.services.vector_service import upsert_tourist_embedding
from app.schemas.payloads import GuideResult, StayResult, ActivityResult, RecommendResponse

log = logging.getLogger(__name__)
s = get_settings()


# ── Reranker constants ────────────────────────────────────────────────────────

BUDGET_ORDER = ["budget", "mid-range", "luxury"]

ACTIVITY_DIFFICULTY_MATCH: Dict[str, List[str]] = {
    "very high": ["hard", "very hard", "extreme", "moderate"],
    "high":      ["moderate", "hard", "very hard"],
    "moderate":  ["easy", "moderate", "hard"],
    "low":       ["none", "easy"],
    "none":      ["none", "easy"],
}


def _budget_bonus(tourist_budget: str, item_budget: str) -> float:
    t = str(tourist_budget).strip().lower()
    i = str(item_budget).strip().lower()
    if t not in BUDGET_ORDER or i not in BUDGET_ORDER:
        return 0.5
    dist = abs(BUDGET_ORDER.index(t) - BUDGET_ORDER.index(i))
    return max(0.0, 1.0 - dist * 0.5)


def _active_bonus(tourist_active: str, item_difficulty: str) -> float:
    compatible = ACTIVITY_DIFFICULTY_MATCH.get(
        str(tourist_active).strip().lower(),
        ["none", "easy", "moderate", "hard", "very hard"],
    )
    return 1.0 if str(item_difficulty).strip().lower() in compatible else 0.0


# ── Typed RPC KNN helpers ─────────────────────────────────────────────────────

def _knn_guides(
    vec: List[float],
    city: Optional[str],
    gender: Optional[str],
    n: int,
    available_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if available_ids is not None and len(available_ids) == 0:
        return []
    params: Dict[str, Any] = {"query_embedding": vec, "match_count": n * 3}
    if city:
        params["city_filter"] = city
    if gender and gender.lower() not in ("any", ""):
        params["gender_filter"] = gender.lower()
    if available_ids is not None:
        params["available_ids"] = available_ids
    result = get_supabase().rpc("match_guides", params).execute()
    return result.data or []


def _knn_stays(
    vec: List[float],
    city: Optional[str],
    n: int,
    available_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if available_ids is not None and len(available_ids) == 0:
        return []
    params: Dict[str, Any] = {"query_embedding": vec, "match_count": n * 3}
    if city:
        params["city_filter"] = city
    if available_ids is not None:
        params["available_ids"] = available_ids
    result = get_supabase().rpc("match_stays", params).execute()
    return result.data or []


def _knn_activities(vec: List[float], n: int) -> List[Dict[str, Any]]:
    result = get_supabase().rpc(
        "match_activities",
        {"query_embedding": vec, "match_count": n * 3},
    ).execute()
    return result.data or []


# ── Entity label helpers ──────────────────────────────────────────────────────

def _get_stay_labels(stay_id: str) -> Dict[str, str]:
    db = get_supabase()
    amb_rows = db.table("stay_ambiance").select("ambiance_id").eq("stay_id", stay_id).execute().data or []
    amb_ids = [r["ambiance_id"] for r in amb_rows]
    ambiance = ""
    if amb_ids:
        rows = db.table("ambiance").select("label").in_("id", amb_ids).execute().data or []
        ambiance = ", ".join(r["label"] for r in rows)

    sf_rows = db.table("stay_suitable_for").select("suitable_for_id").eq("stay_id", stay_id).execute().data or []
    sf_ids = [r["suitable_for_id"] for r in sf_rows]
    suitable_for = ""
    if sf_ids:
        rows = db.table("suitable_for").select("label").in_("id", sf_ids).execute().data or []
        suitable_for = ", ".join(r["label"] for r in rows)

    return {"ambiance": ambiance, "suitable_for": suitable_for}


def _get_activity_labels(activity_id: str) -> str:
    db = get_supabase()
    sf_rows = db.table("activity_suitable_for").select("suitable_for_id").eq("activity_id", activity_id).execute().data or []
    sf_ids = [r["suitable_for_id"] for r in sf_rows]
    if not sf_ids:
        return ""
    rows = db.table("suitable_for").select("label").in_("id", sf_ids).execute().data or []
    return ", ".join(r["label"] for r in rows)


def _get_guide_labels(user_profile_id: str) -> Dict[str, str]:
    db = get_supabase()
    int_rows = db.table("user_interest").select("interest_id").eq("user_profile_id", user_profile_id).execute().data or []
    int_ids = [r["interest_id"] for r in int_rows]
    interests = ""
    if int_ids:
        rows = db.table("interest").select("name").in_("id", int_ids).execute().data or []
        interests = ", ".join(r["name"] for r in rows)

    lang_rows = db.table("user_language").select("language_id").eq("user_profile_id", user_profile_id).execute().data or []
    lang_ids = [r["language_id"] for r in lang_rows]
    languages = ""
    if lang_ids:
        rows = db.table("language").select("name").in_("id", lang_ids).execute().data or []
        languages = ", ".join(r["name"] for r in rows)

    return {"interests": interests, "languages": languages}


# ── Tourist vector resolution ─────────────────────────────────────────────────

def _get_tourist_vectors(tourist_id: str) -> Dict[str, List[float]]:
    """
    Read all three cached vectors from tourist_profile.
    If ANY is NULL → recompute and store all three atomically.
    Returns {"guide": [...], "stay": [...], "activity": [...]}
    """
    db = get_supabase()
    row = (
        db.table("tourist_profile")
        .select("id, t2g_embedding, t2s_embedding, t2a_embedding")
        .eq("id", tourist_id)
        .single()
        .execute()
    ).data

    if not row:
        raise ValueError(f"tourist_profile not found: {tourist_id}")

    t2g = row.get("t2g_embedding")
    t2s = row.get("t2s_embedding")
    t2a = row.get("t2a_embedding")

    # If all three are cached, return them directly — zero embed calls
    if t2g and t2s and t2a:
        return {"guide": t2g, "stay": t2s, "activity": t2a}

    # Any column is NULL → recompute all three and store atomically
    log.info("Tourist %s: missing vectors, recomputing all three", tourist_id)
    return upsert_tourist_embedding(tourist_id)


# ── Public recommendation function ───────────────────────────────────────────

def recommend(
    tourist_id: str,
    city: Optional[str] = None,
    guide_gender: Optional[str] = None,
    top_k: int = 5,
    available_guide_ids: Optional[List[str]] = None,
    available_stay_ids: Optional[List[str]] = None,
) -> RecommendResponse:
    """
    Main entry point — called by POST /recommend.
    Reads t2g/t2s/t2a vectors from tourist_profile (or recomputes if NULL).
    """
    db = get_supabase()

    tp = (
        db.table("tourist_profile")
        .select("budget, active_level")
        .eq("id", tourist_id)
        .single()
        .execute()
    ).data or {}
    tourist_budget = tp.get("budget", "")
    tourist_active = tp.get("active_level", "")

    vecs = _get_tourist_vectors(tourist_id)

    # ── GUIDES ───────────────────────────────────────────────────────────────
    raw_guides = _knn_guides(
        vecs["guide"], city, guide_gender,
        n=top_k, available_ids=available_guide_ids,
    )
    guide_results: List[GuideResult] = []
    seen_guides: set = set()
    for g in raw_guides:
        gid = str(g.get("guide_profile_id", ""))
        if gid in seen_guides:
            continue
        seen_guides.add(gid)
        vec_sim = float(g.get("vec_sim", 0))
        labels = _get_guide_labels(str(g.get("user_profile_id", "")))
        guide_results.append(GuideResult(
            guide_profile_id=gid,
            user_profile_id=str(g.get("user_profile_id", "")),
            full_name=g.get("full_name", ""),
            city_name=g.get("city_name"),
            gender=g.get("gender"),
            avg_rating=g.get("avg_rating"),
            experience_years=g.get("experience_years"),
            rate_per_hour=g.get("rate_per_hour"),
            specializations=None,
            languages=labels.get("languages"),
            profile_bio=g.get("profile_bio"),
            vec_sim=round(vec_sim, 4),
            final_score=round(vec_sim, 4),
        ))
        if len(guide_results) >= top_k:
            break

    # ── STAYS ────────────────────────────────────────────────────────────────
    raw_stays = _knn_stays(
        vecs["stay"], city,
        n=top_k, available_ids=available_stay_ids,
    )
    stay_results: List[StayResult] = []
    seen_stays: set = set()
    for st in raw_stays:
        sid = str(st.get("stay_id", ""))
        if sid in seen_stays:
            continue
        seen_stays.add(sid)
        vec_sim = float(st.get("vec_sim", 0))
        budget_bonus = _budget_bonus(tourist_budget, st.get("budget", ""))
        final = round(
            s.rerank_vec_weight * vec_sim + s.rerank_budget_weight * budget_bonus, 4
        )
        labels = _get_stay_labels(sid)
        stay_results.append(StayResult(
            stay_id=sid,
            name=st.get("name", ""),
            type=st.get("type"),
            city_name=st.get("city_name"),
            description=st.get("description"),
            budget=st.get("budget"),
            price_per_night=st.get("price_per_night"),
            ambiance=labels.get("ambiance"),
            suitable_for=labels.get("suitable_for"),
            avg_rating=st.get("avg_rating"),
            vec_sim=round(vec_sim, 4),
            final_score=final,
        ))
        if len(stay_results) >= top_k:
            break

    stay_results.sort(key=lambda r: r.final_score, reverse=True)

    # ── ACTIVITIES ───────────────────────────────────────────────────────────
    raw_activities = _knn_activities(vecs["activity"], n=top_k)
    activity_results: List[ActivityResult] = []
    seen_acts: set = set()
    for act in raw_activities:
        aid = str(act.get("activity_id", ""))
        if aid in seen_acts:
            continue
        seen_acts.add(aid)
        vec_sim = float(act.get("vec_sim", 0))
        budget_bonus = _budget_bonus(tourist_budget, act.get("budget", ""))
        active_bonus = _active_bonus(tourist_active, act.get("difficulty_level", ""))
        final = round(
            s.rerank_vec_weight * vec_sim
            + s.rerank_budget_weight * budget_bonus
            + s.rerank_active_weight * active_bonus,
            4,
        )
        suitable_for = _get_activity_labels(aid)
        activity_results.append(ActivityResult(
            activity_id=aid,
            name=act.get("name", ""),
            category=act.get("category"),
            description=act.get("description"),
            budget=act.get("budget"),
            difficulty_level=act.get("difficulty_level"),
            base_price=act.get("base_price"),
            suitable_for=suitable_for,
            vec_sim=round(vec_sim, 4),
            final_score=final,
        ))
        if len(activity_results) >= top_k:
            break

    activity_results.sort(key=lambda r: r.final_score, reverse=True)

    return RecommendResponse(
        tourist_id=tourist_id,
        guides=guide_results,
        stays=stay_results,
        activities=activity_results,
    )
