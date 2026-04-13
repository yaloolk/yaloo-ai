"""
app/services/text_builder.py

Converts a flat entity dict (fetched from Supabase) into a weighted
text string ready for embedding.

Logic is a direct port of the notebook's `row_to_text()` approach,
with two additions:
  - local_activities field injected into guide and stay texts (weight=1)
  - travel-style → stay/activity bridge strings preserved exactly
"""
from __future__ import annotations
from typing import Any, Dict, Optional


# ── Field weights ─────────────────────────────────────────────────────────────
# Same as notebook.  Higher weight = repeated more times before embedding.

TOURIST_W: Dict[str, int] = {
    "travel_style":  4,
    "interests":     3,
    "budget":        3,
    "active_level":  2,
    "profile_bio":   2,
    "languages":     1,
}

GUIDE_W: Dict[str, int] = {
    "specializations": 4,
    "interests":       3,
    "profile_bio":     3,
    "active_level":    2,
    "languages":       2,
    "local_activities": 1,   # new — activities the guide personally offers
}

STAY_W: Dict[str, int] = {
    "suitable_for":    4,
    "ambiance":        4,
    "description":     3,
    "budget":          3,
    "type":            2,
    "local_activities": 1,   # new — activities the host offers at the stay
}

ACTIVITY_W: Dict[str, int] = {
    "suitable_for":    4,
    "category":        3,
    "description":     3,
    "difficulty_level": 2,
    "budget":          2,
}


# ── Travel-style bridge strings ───────────────────────────────────────────────
# Appended to tourist text when querying stays / activities.
# Preserved verbatim from notebook.

TRAVEL_TO_STAY: Dict[str, str] = {
    "adventure":   "outdoor rustic eco jungle mountain trek wild adventure seekers hiking",
    "luxury":      "luxury exclusive premium butler private indulgent luxury travelers fine",
    "backpacker":  "social budget hostel communal cheap shared backpackers affordable",
    "cultural":    "heritage historic traditional authentic colonial cultural tourists",
    "wellness":    "serene peaceful healing spiritual retreat meditation tranquil wellness",
    "solo":        "safe independent flexible solo travelers welcoming solo traveler",
    "eco":         "sustainable eco friendly green organic nature conservation eco travelers",
    "culinary":    "food cooking gourmet culinary dining local flavors foodies",
    "slow travel": "immersive local authentic community long stay peaceful heritage",
}

TRAVEL_TO_ACTIVITY: Dict[str, str] = {
    "adventure":   "adventure seekers outdoor trekking hiking nature water sports rugged",
    "luxury":      "luxury travelers premium fine dining cultural exclusive private",
    "backpacker":  "budget backpackers social affordable group activities communal",
    "cultural":    "cultural tourists heritage history traditional religious temples",
    "wellness":    "wellness spiritual healing meditation yoga retreat spiritual travelers",
    "solo":        "solo travelers independent flexible easy accessible self guided",
    "eco":         "eco travelers nature sustainable wildlife conservation organic",
    "culinary":    "foodies food local cuisine cooking culinary dining flavors",
    "slow travel": "immersive authentic local heritage traditional cultural community",
}


# ── Core builder ──────────────────────────────────────────────────────────────

def _is_valid(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip()
    return s not in ("", "nan", "None", "NaN", "null")


def row_to_text(row: Dict[str, Any], weights: Dict[str, int]) -> str:
    """
    Repeat each field [weight] times and join with ' | '.
    Mirrors the notebook's weighted-repetition strategy.
    """
    parts: list[str] = []
    for field, w in weights.items():
        val = row.get(field)
        if _is_valid(val):
            parts.extend([str(val).strip()] * w)
    return " | ".join(parts)


# ── Per-entity text functions ─────────────────────────────────────────────────

def guide_text(row: Dict[str, Any]) -> str:
    """Text for embedding a guide (used as the guide-side vector)."""
    return row_to_text(row, GUIDE_W)


def stay_text(row: Dict[str, Any]) -> str:
    """Text for embedding a stay (used as the stay-side vector)."""
    return row_to_text(row, STAY_W)


def activity_text(row: Dict[str, Any]) -> str:
    """Text for embedding a global activity."""
    return row_to_text(row, ACTIVITY_W)


# ── Tourist text — three variants depending on query target ───────────────────

def tourist_text_for_guide(row: Dict[str, Any]) -> str:
    """Tourist query vector used when searching guides."""
    return row_to_text(row, TOURIST_W)


def tourist_text_for_stay(row: Dict[str, Any]) -> str:
    """Tourist query vector used when searching stays (adds bridge string)."""
    base = row_to_text(row, TOURIST_W)
    style = str(row.get("travel_style", "")).lower().strip()
    bridge = TRAVEL_TO_STAY.get(style, "")
    return f"{base} | {bridge}" if bridge else base


def tourist_text_for_activity(row: Dict[str, Any]) -> str:
    """Tourist query vector used when searching activities (adds bridge string)."""
    base = row_to_text(row, TOURIST_W)
    style = str(row.get("travel_style", "")).lower().strip()
    bridge = TRAVEL_TO_ACTIVITY.get(style, "")
    return f"{base} | {bridge}" if bridge else base
