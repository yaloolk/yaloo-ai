"""
app/schemas/payloads.py
Pydantic models for:
  - Supabase DB webhook payloads (INSERT / UPDATE events)
  - FastAPI request / response bodies
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, field_validator, Field


# ── Supabase DB webhook envelope ────────────────────────────────────────────
#
# Supabase sends different nulls depending on the event type:
#
#   Event    │ record  │ old_record
#   ─────────┼─────────┼───────────
#   INSERT   │ new row │ null
#   UPDATE   │ new row │ old row
#   DELETE   │ null    │ old row
#
# Both fields must be Optional to avoid 422s on DELETE (record=null)
# and INSERT (old_record=null) webhooks.
# Use .safe_record and .safe_old instead of accessing .record / .old_record
# directly — they always return a plain dict and handle the null cases for you.

class WebhookPayload(BaseModel):
    type: str
    table: str
    # Use an alias to map incoming 'schema' to 'schema_'
    schema_: str = Field(alias="schema")
    record: Optional[Dict[str, Any]] = None       # null on DELETE
    old_record: Optional[Dict[str, Any]] = None   # null on INSERT

    class Config:
        # This allows you to still use 'schema_' when creating the object manually
        populate_by_name = True

    @property
    def safe_record(self) -> Dict[str, Any]:
        """
        The authoritative row for this event — never None.
          INSERT / UPDATE → payload.record   (the new / current row)
          DELETE          → payload.old_record (the row that was removed)
        Use this everywhere instead of payload.record directly.
        """
        if self.type == "DELETE":
            return self.old_record or {}
        return self.record or {}

    @property
    def safe_old(self) -> Dict[str, Any]:
        """
        The previous state of the row — never None.
          UPDATE → payload.old_record  (values before the change)
          INSERT → {}  (no previous state)
          DELETE → {}  (safe_record already holds the row)
        Use this for change-detection guards: rec.get(f) != old.get(f).
        """
        return self.old_record or {}


# ── Recommendation request ───────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    tourist_id: str
    city: Optional[str] = None               # filter by city name (case-insensitive)
    guide_gender: Optional[str] = None       # "male" | "female" | "any"
    top_k: int = 5

    # ── Availability filter (provided by Django booking backend) ────────────
    # Django checks the requested date/time slot and returns the IDs of
    # guides/stays that are free. FastAPI only ranks within this pool.
    # If None → no availability filter applied (browse mode, no date selected).
    available_guide_ids: Optional[List[str]] = None
    available_stay_ids: Optional[List[str]] = None


class GuideResult(BaseModel):
    guide_profile_id: str
    user_profile_id: str
    full_name: str
    city_name: Optional[str]
    gender: Optional[str]
    avg_rating: Optional[float]
    experience_years: Optional[int]
    rate_per_hour: Optional[float]
    specializations: Optional[str]
    languages: Optional[str]
    profile_bio: Optional[str]
    vec_sim: float
    final_score: float


class StayResult(BaseModel):
    stay_id: str
    name: str
    type: Optional[str]
    city_name: Optional[str]
    description: Optional[str]
    budget: Optional[str]
    price_per_night: Optional[float]
    ambiance: Optional[str]
    suitable_for: Optional[str]
    avg_rating: Optional[float]
    vec_sim: float
    final_score: float


class ActivityResult(BaseModel):
    activity_id: str
    name: str
    category: Optional[str]
    description: Optional[str]
    budget: Optional[str]
    difficulty_level: Optional[str]
    base_price: Optional[float]
    suitable_for: Optional[str]
    vec_sim: float
    final_score: float


class RecommendResponse(BaseModel):
    tourist_id: str
    guides: List[GuideResult]
    stays: List[StayResult]
    activities: List[ActivityResult]


# ── Chat ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str    # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    # FIX: tourist_id is always required in the request body, but an empty string
    # is now accepted and treated as "not logged in" (guest user).
    # This avoids 422 errors when the Flutter client sends tourist_id: "" for
    # unauthenticated users instead of omitting the field entirely.
    tourist_id: str
    messages: List[ChatMessage]

    @field_validator("tourist_id")
    def tourist_id_must_be_string(cls, v: str) -> str:
        # Allow empty string — chatbot.py checks `if req.tourist_id` before
        # fetching tourist context, so empty string = guest mode gracefully.
        return v.strip()


class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = []
