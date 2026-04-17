"""
app/schemas/payloads.py
Pydantic models for:
  - Supabase DB webhook payloads (INSERT / UPDATE events)
  - FastAPI request / response bodies
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel,field_validator,Field


# ── Supabase DB webhook envelope ────────────────────────────────────────────

class WebhookPayload(BaseModel):
    type: str
    table: str
    # Use an alias to map incoming 'schema' to 'schema_'
    schema_: str = Field(alias="schema")
    record: Dict[str, Any]
    old_record: Optional[Dict[str, Any]] = None

    class Config:
        # This allows you to still use 'schema_' when creating the object manually
        populate_by_name = True


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
    tourist_id: str
    messages: List[ChatMessage]

    @field_validator("tourist_id")
    def must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("tourist_id is required")
        return v


class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = []
