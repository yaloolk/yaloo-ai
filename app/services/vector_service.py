"""
app/services/vector_service.py

Responsibilities:
  1. Load the HuggingFace embedding model once at startup (singleton).
  2. Fetch enriched entity rows from Supabase (joining related tables).
  3. Compute embeddings and upsert them back into Supabase.

Tourist embedding columns (three separate, one per query target):
  t2g_embedding  — tourist-to-guide query vector
  t2s_embedding  — tourist-to-stay query vector
  t2a_embedding  — tourist-to-activity query vector

All three are stored and kept in sync together. No lazy computation —
all three are written atomically on every tourist embed call.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings
from app.core.database import get_supabase
from app.services.text_builder import (
    guide_text,
    stay_text,
    activity_text,
    tourist_text_for_guide,
    tourist_text_for_stay,
    tourist_text_for_activity,
)

log = logging.getLogger(__name__)


# ── Embedding model singleton ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    s = get_settings()
    log.info("Loading embedding model %s on %s …", s.embedding_model, s.embedding_device)
    model = HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        model_kwargs={"device": s.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info("Embedding model ready (%d dims)", s.embedding_dim)
    return model


def embed(text: str) -> List[float]:
    """Return a 768-dim normalised vector for a single text string."""
    return get_embedding_model().embed_query(text)


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Batch embed — more efficient for backfill scripts."""
    return get_embedding_model().embed_documents(texts)


# ── Shared label helper ───────────────────────────────────────────────────────

def _join_labels(rows: List[Dict], label_key: str, sep: str = ", ") -> str:
    return sep.join(r[label_key] for r in rows if r.get(label_key))


# ── Data fetchers ─────────────────────────────────────────────────────────────

def fetch_guide_row(guide_profile_id: str) -> Optional[Dict[str, Any]]:
    """
    Joins: guide_profile → user_profile, city, specializations,
           interests, languages, local_activities (weight-1)
    """
    db = get_supabase()

    gp = (
        db.table("guide_profile")
        .select("id, user_profile_id, city_id, experience_years, avg_rating, rate_per_hour, active_level")
        .eq("id", guide_profile_id)
        .single()
        .execute()
    ).data
    if not gp:
        return None

    up = (
        db.table("user_profile")
        .select("first_name, last_name, profile_bio, gender")
        .eq("id", gp["user_profile_id"])
        .single()
        .execute()
    ).data or {}

    city = (
        db.table("city")
        .select("name")
        .eq("id", gp["city_id"])
        .single()
        .execute()
    ).data or {}

    spec_joins = (
        db.table("guide_specialization")
        .select("specialization_id")
        .eq("guide_profile_id", guide_profile_id)
        .execute()
    ).data or []
    spec_ids = [r["specialization_id"] for r in spec_joins]
    specializations = ""
    if spec_ids:
        rows = db.table("specialization").select("label").in_("id", spec_ids).execute().data or []
        specializations = _join_labels(rows, "label")

    int_joins = (
        db.table("user_interest")
        .select("interest_id")
        .eq("user_profile_id", gp["user_profile_id"])
        .execute()
    ).data or []
    int_ids = [r["interest_id"] for r in int_joins]
    interests = ""
    if int_ids:
        rows = db.table("interest").select("name").in_("id", int_ids).execute().data or []
        interests = _join_labels(rows, "name")

    lang_joins = (
        db.table("user_language")
        .select("language_id")
        .eq("user_profile_id", gp["user_profile_id"])
        .execute()
    ).data or []
    lang_ids = [r["language_id"] for r in lang_joins]
    languages = ""
    if lang_ids:
        rows = db.table("language").select("name").in_("id", lang_ids).execute().data or []
        languages = _join_labels(rows, "name")

    # local_activities this guide personally offers (weight-1 enrichment)
    la_rows = (
        db.table("local_activity")
        .select("activity_id")
        .eq("guide_id", guide_profile_id)
        .execute()
    ).data or []
    la_ids = [r["activity_id"] for r in la_rows]
    local_activities = ""
    if la_ids:
        rows = db.table("activity").select("name").in_("id", la_ids).execute().data or []
        local_activities = _join_labels(rows, "name")

    return {
        "guide_profile_id": guide_profile_id,
        "user_profile_id":  gp["user_profile_id"],
        "full_name":        f"{up.get('first_name','')} {up.get('last_name','')}".strip(),
        "gender":           up.get("gender"),
        "profile_bio":      up.get("profile_bio"),
        "city_name":        city.get("name"),
        "experience_years": gp.get("experience_years"),
        "avg_rating":       gp.get("avg_rating"),
        "rate_per_hour":    gp.get("rate_per_hour"),
        "active_level":     gp.get("active_level"),
        "specializations":  specializations,
        "interests":        interests,
        "languages":        languages,
        "local_activities": local_activities,
    }


def fetch_stay_row(stay_id: str) -> Optional[Dict[str, Any]]:
    """
    Joins: stay → city, host_profile (avg_rating),
           stay_ambiance, stay_suitable_for, local_activities (weight-1)

    IMPORTANT: stay.host_id = host_profile.id  (NOT user_profile_id)
    host_profile query uses .eq("id", stay["host_id"])
    """
    db = get_supabase()

    stay = (
        db.table("stay")
        .select("id, host_id, name, type, description, budget, price_per_night, city_id")
        .eq("id", stay_id)
        .single()
        .execute()
    ).data
    if not stay:
        return None

    city = (
        db.table("city")
        .select("name")
        .eq("id", stay["city_id"])
        .single()
        .execute()
    ).data or {}

    # stay.host_id = host_profile.id  (not user_profile_id — confirmed from CSV)
    hp = (
        db.table("host_profile")
        .select("avg_rating")
        .eq("id", stay["host_id"])
        .maybe_single()
        .execute()
    ).data or {}

    amb_joins = (
        db.table("stay_ambiance")
        .select("ambiance_id")
        .eq("stay_id", stay_id)
        .execute()
    ).data or []
    amb_ids = [r["ambiance_id"] for r in amb_joins]
    ambiance = ""
    if amb_ids:
        rows = db.table("ambiance").select("label").in_("id", amb_ids).execute().data or []
        ambiance = _join_labels(rows, "label")

    sf_joins = (
        db.table("stay_suitable_for")
        .select("suitable_for_id")
        .eq("stay_id", stay_id)
        .execute()
    ).data or []
    sf_ids = [r["suitable_for_id"] for r in sf_joins]
    suitable_for = ""
    if sf_ids:
        rows = db.table("suitable_for").select("label").in_("id", sf_ids).execute().data or []
        suitable_for = _join_labels(rows, "label")

    # local_activities offered by this host (weight-1 enrichment)
    # stay.host_id = host_profile.id, local_activity.host_id = host_profile.id
    la_rows = (
        db.table("local_activity")
        .select("activity_id")
        .eq("host_id", stay["host_id"])
        .execute()
    ).data or []
    la_ids = [r["activity_id"] for r in la_rows]
    local_activities = ""
    if la_ids:
        rows = db.table("activity").select("name").in_("id", la_ids).execute().data or []
        local_activities = _join_labels(rows, "name")

    return {
        "stay_id":          stay_id,
        "name":             stay.get("name"),
        "type":             stay.get("type"),
        "description":      stay.get("description"),
        "budget":           stay.get("budget"),
        "price_per_night":  stay.get("price_per_night"),
        "city_name":        city.get("name"),
        "avg_rating":       hp.get("avg_rating"),
        "ambiance":         ambiance,
        "suitable_for":     suitable_for,
        "local_activities": local_activities,
    }


def fetch_activity_row(activity_id: str) -> Optional[Dict[str, Any]]:
    """Global activity with suitable_for labels joined."""
    db = get_supabase()

    act = (
        db.table("activity")
        .select("id, name, category, description, budget, difficulty_level, base_price")
        .eq("id", activity_id)
        .single()
        .execute()
    ).data
    if not act:
        return None

    sf_joins = (
        db.table("activity_suitable_for")
        .select("suitable_for_id")
        .eq("activity_id", activity_id)
        .execute()
    ).data or []
    sf_ids = [r["suitable_for_id"] for r in sf_joins]
    suitable_for = ""
    if sf_ids:
        rows = db.table("suitable_for").select("label").in_("id", sf_ids).execute().data or []
        suitable_for = _join_labels(rows, "label")

    return {
        "activity_id":     activity_id,
        "name":            act.get("name"),
        "category":        act.get("category"),
        "description":     act.get("description"),
        "budget":          act.get("budget"),
        "difficulty_level": act.get("difficulty_level"),
        "base_price":      act.get("base_price"),
        "suitable_for":    suitable_for,
    }


def fetch_tourist_row(tourist_profile_id: str) -> Optional[Dict[str, Any]]:
    """Tourist data flattened — used by text_builder to produce all 3 query texts."""
    db = get_supabase()

    tp = (
        db.table("tourist_profile")
        .select("id, user_profile_id, travel_style, budget, active_level")
        .eq("id", tourist_profile_id)
        .single()
        .execute()
    ).data
    if not tp:
        return None

    up = (
        db.table("user_profile")
        .select("profile_bio")
        .eq("id", tp["user_profile_id"])
        .single()
        .execute()
    ).data or {}

    int_joins = (
        db.table("user_interest")
        .select("interest_id")
        .eq("user_profile_id", tp["user_profile_id"])
        .execute()
    ).data or []
    int_ids = [r["interest_id"] for r in int_joins]
    interests = ""
    if int_ids:
        rows = db.table("interest").select("name").in_("id", int_ids).execute().data or []
        interests = _join_labels(rows, "name")

    lang_joins = (
        db.table("user_language")
        .select("language_id")
        .eq("user_profile_id", tp["user_profile_id"])
        .execute()
    ).data or []
    lang_ids = [r["language_id"] for r in lang_joins]
    languages = ""
    if lang_ids:
        rows = db.table("language").select("name").in_("id", lang_ids).execute().data or []
        languages = _join_labels(rows, "name")

    return {
        "tourist_profile_id": tourist_profile_id,
        "user_profile_id":    tp["user_profile_id"],
        "travel_style":       tp.get("travel_style"),
        "budget":             tp.get("budget"),
        "active_level":       tp.get("active_level"),
        "profile_bio":        up.get("profile_bio"),
        "interests":          interests,
        "languages":          languages,
    }


# ── Upsert helpers ────────────────────────────────────────────────────────────

def upsert_guide_embedding(guide_profile_id: str) -> bool:
    row = fetch_guide_row(guide_profile_id)
    if not row:
        log.warning("guide_profile %s not found", guide_profile_id)
        return False
    vec = embed(guide_text(row))
    get_supabase().table("guide_profile").update(
        {"embedding": vec}
    ).eq("id", guide_profile_id).execute()
    log.info("Guide %s embedded (%d dims)", guide_profile_id, len(vec))
    return True


def upsert_stay_embedding(stay_id: str) -> bool:
    row = fetch_stay_row(stay_id)
    if not row:
        log.warning("stay %s not found", stay_id)
        return False
    vec = embed(stay_text(row))
    get_supabase().table("stay").update(
        {"embedding": vec}
    ).eq("id", stay_id).execute()
    log.info("Stay %s embedded", stay_id)
    return True


def upsert_activity_embedding(activity_id: str) -> bool:
    row = fetch_activity_row(activity_id)
    if not row:
        log.warning("activity %s not found", activity_id)
        return False
    vec = embed(activity_text(row))
    get_supabase().table("activity").update(
        {"embedding": vec}
    ).eq("id", activity_id).execute()
    log.info("Activity %s embedded", activity_id)
    return True


def upsert_tourist_embedding(tourist_profile_id: str) -> Dict[str, List[float]]:
    """
    Compute all three tourist query vectors and store them atomically.

    Columns written:
      t2g_embedding  — for querying guides
      t2s_embedding  — for querying stays
      t2a_embedding  — for querying activities

    Returns {"guide": [...], "stay": [...], "activity": [...]}
    so rec_engine can use them immediately without a second DB read.
    """
    row = fetch_tourist_row(tourist_profile_id)
    if not row:
        raise ValueError(f"tourist_profile {tourist_profile_id} not found")

    vec_guide    = embed(tourist_text_for_guide(row))
    vec_stay     = embed(tourist_text_for_stay(row))
    vec_activity = embed(tourist_text_for_activity(row))

    get_supabase().table("tourist_profile").update({
        "t2g_embedding": vec_guide,
        "t2s_embedding": vec_stay,
        "t2a_embedding": vec_activity,
    }).eq("id", tourist_profile_id).execute()

    log.info("Tourist %s embedded — t2g, t2s, t2a written", tourist_profile_id)
    return {
        "guide":    vec_guide,
        "stay":     vec_stay,
        "activity": vec_activity,
    }


def invalidate_tourist_embedding(tourist_profile_id: str) -> None:
    """
    Null all three tourist vectors so next /recommend call recomputes them.
    Called when interests, languages, travel_style, budget, or active_level change.
    """
    get_supabase().table("tourist_profile").update({
        "t2g_embedding": None,
        "t2s_embedding": None,
        "t2a_embedding": None,
    }).eq("id", tourist_profile_id).execute()
    log.info("Tourist %s embeddings invalidated (t2g, t2s, t2a → NULL)", tourist_profile_id)


def upsert_doc_chunk_embedding(chunk_id: str, content: str) -> bool:
    """
    Embed a single doc_chunk row and write back its vector.
    Called by embed_docs.py script and the /embed/doc webhook.
    """
    try:
        vec = embed(content)
        get_supabase().table("doc_chunk").update(
            {"embedding": vec}
        ).eq("id", chunk_id).execute()
        log.info("doc_chunk %s embedded", chunk_id)
        return True
    except Exception as e:
        log.error("doc_chunk %s embedding failed: %s", chunk_id, e)
        return False
