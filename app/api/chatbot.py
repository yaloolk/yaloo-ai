"""
app/api/chatbot.py

POST /chat  — Single-call intent-routed RAG chatbot.

KEY DESIGN: ONE Gemini call per user message (not two).

Previous design made two calls:
  1. _classify_intent()   → one Gemini call to get JSON intent
  2. chat_session.send_message() → one Gemini call for the actual reply

This burned through the free tier quota (20 req/day) in 10 messages
and also caused failures when the classifier returned malformed JSON.

New design — all in ONE call:
  1. Before calling Gemini, we check the message ourselves for city info
     using a simple keyword scan (zero API cost).
  2. We build a system_instruction that tells Gemini exactly what to do
     based on what context we have available (rec data / doc chunks / nothing).
  3. Gemini reads the system instruction + user message and responds.
     It implicitly understands the intent from the instructions — no JSON.

Token budget per request:
  general            : ~200–400 tokens  (system + message + reply)
  docs               : ~700–900 tokens  (system + chunks + reply)
  recommend (no city): ~200 tokens      (asks for city, no data fetch)
  recommend (city)   : ~700–1000 tokens (system + live data + reply)

Multi-language: Gemini handles English, Sinhala, Tamil natively.
"""
import logging
import re
from typing import Optional

import google.generativeai as genai
from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.database import get_supabase
from app.schemas.payloads import ChatRequest, ChatResponse
from app.services import rec_engine
from app.services.vector_service import embed
from app.core.api_fallback import api_config

log = logging.getLogger(__name__)
router = APIRouter()


# ── Gemini model factory ──────────────────────────────────────────────────────

def _make_model(system_instruction: str, api_key: str) -> genai.GenerativeModel:
    """
    Build a GenerativeModel using the provided api_key.
    Called inside the fallback retry loop — api_key changes per attempt
    as the fallback chain advances through primary → secondary → tertiary.
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
    )


# ── City extractor — zero API cost ────────────────────────────────────────────
# Simple keyword scan over the message text.
# Catches the most common cases without any API call.
# If the user wrote the city name, we find it here.

_SRI_LANKA_CITIES = [
    "colombo", "kandy", "galle", "ella", "sigiriya", "mirissa",
    "negombo", "trincomalee", "jaffna", "nuwara eliya", "arugam bay",
    "batticaloa", "ratnapura", "kurunegala", "matara", "badulla",
]


def _extract_city(text: str) -> Optional[str]:
    """
    Scan a text string for a known Sri Lanka city name.
    Returns the properly capitalised city name, or None if not found.
    """
    lower = text.lower()
    for city in _SRI_LANKA_CITIES:
        if city in lower:
            return city.title()
    return None


def _extract_city_from_history(messages) -> Optional[str]:
    """
    Walk the full conversation history (oldest → newest) looking for a city.
    Returns the most recently mentioned city, or None.
    This ensures the city provided in a previous turn (e.g. answering
    "Which city are you visiting?") is not lost on subsequent messages.
    """
    found = None
    for msg in messages:
        city = _extract_city(msg.content)
        if city:
            found = city   # keep updating so we get the most recent mention
    return found


# ── Intent detector — zero API cost ──────────────────────────────────────────
# Keyword-based classification. Fast, free, deterministic.
# Handles English, and common Sinhala/Tamil romanisations reasonably well
# since tourists typically mix English words when asking about services.

_RECOMMEND_KEYWORDS = [
    "guide", "stay", "homestay", "hotel", "accommodation", "activity",
    "activities", "recommend", "find", "suggest", "book", "visit",
    "tour", "place", "where", "what to do", "things to do",
]

# Keywords that signal a specific entity type within a recommend intent.
# Checked in priority order: guide → stay → activity → None (means all).
_GUIDE_KEYWORDS    = ["guide", "guides", "local guide", "tour guide", "escort"]
_STAY_KEYWORDS     = ["stay", "stays", "homestay", "homestays", "hotel", "accommodation",
                      "lodging", "place to sleep", "place to stay", "room", "host"]
_ACTIVITY_KEYWORDS = ["activity", "activities", "things to do", "what to do", "experience",
                      "experiences", "adventure", "hike", "hiking", "excursion"]

_DOC_KEYWORDS = [
    "policy", "cancel", "cancellation", "refund", "fee", "cost", "price",
    "how does yaloo work", "how yaloo works", "booking", "payment", "safe",
    "safety", "verify", "verified", "sltda", "contact", "support",
    "register", "sign up", "sign-up", "account",
]


def _detect_intent(message: str) -> str:
    """
    Returns "recommend", "docs", or "general".
    Checks recommend first, then docs, falls back to general.
    """
    lower = message.lower()
    if any(kw in lower for kw in _RECOMMEND_KEYWORDS):
        return "recommend"
    if any(kw in lower for kw in _DOC_KEYWORDS):
        return "docs"
    return "general"


def _detect_entity(messages) -> Optional[str]:
    """
    Scan the full conversation history for entity keywords.
    Returns "guide", "stay", "activity", or None (meaning all three).
    Scans oldest → newest so the most recent specific request wins.
    None means the user was vague → run all three.
    """
    found = None
    for msg in messages:
        lower = msg.content.lower()
        if any(kw in lower for kw in _GUIDE_KEYWORDS):
            found = "guide"
        elif any(kw in lower for kw in _STAY_KEYWORDS):
            found = "stay"
        elif any(kw in lower for kw in _ACTIVITY_KEYWORDS):
            found = "activity"
    return found


# ── Context fetchers ──────────────────────────────────────────────────────────

def _fetch_linked_providers(activity_ids: list) -> str:
    """
    For a list of activity IDs, query local_activity to find guides and stays
    that offer them, and return a compact summary block.
    Returns empty string if nothing found or on failure.
    """
    if not activity_ids:
        return ""
    try:
        db = get_supabase()
        rows = (
            db.table("local_activity")
            .select("activity_id, guide_id, stay_id, name")
            .in_("activity_id", activity_ids)
            .execute()
        ).data or []
        if not rows:
            return ""

        guide_ids = list({r["guide_id"] for r in rows if r.get("guide_id")})
        stay_ids  = list({r["stay_id"]  for r in rows if r.get("stay_id")})
        lines: list = []

        if guide_ids:
            g_rows = (
                db.table("guide_profile")
                .select("id, full_name, avg_rating, rate_per_hour")
                .in_("id", guide_ids)
                .execute()
            ).data or []
            if g_rows:
                lines.append("Guides who offer these activities:")
                for g in g_rows:
                    lines.append(
                        f"  {g.get('full_name', '?')} | "
                        f"Rating {g.get('avg_rating') or '?'} | "
                        f"LKR {g.get('rate_per_hour') or '?'}/hr"
                    )

        if stay_ids:
            s_rows = (
                db.table("stay")
                .select("id, name, budget, price_per_night, avg_rating")
                .in_("id", stay_ids)
                .execute()
            ).data or []
            if s_rows:
                lines.append("Stays that offer these activities:")
                for st in s_rows:
                    lines.append(
                        f"  {st.get('name', '?')} | "
                        f"{st.get('budget') or 'N/A'} | "
                        f"LKR {st.get('price_per_night') or '?'}/night | "
                        f"Rating {st.get('avg_rating') or '?'}"
                    )

        return "\n".join(lines)
    except Exception as e:
        log.warning("_fetch_linked_providers failed: %s", e)
        return ""


def _fetch_recommendation_context(tourist_id: str, city: str, entity: Optional[str]) -> str:
    """
    Calls only the rec_engine function needed for the detected entity type.
    For activities, also fetches guides/stays linked via local_activity.

    entity: "guide" | "stay" | "activity" | None (all three)
    Returns empty string on failure — bot degrades gracefully.
    """
    lines: list = []

    def _fmt_guides(guides) -> None:
        if not guides:
            return
        lines.append("Guides:")
        for g in guides:
            lines.append(
                f"  {g.full_name} | {g.city_name or 'N/A'} | "
                f"Rating {g.avg_rating or '?'} | {g.experience_years or '?'} yrs | "
                f"LKR {g.rate_per_hour or '?'}/hr | {g.languages or 'N/A'}"
            )

    def _fmt_stays(stays) -> None:
        if not stays:
            return
        lines.append("Stays:")
        for st in stays:
            lines.append(
                f"  {st.name} | {st.city_name or 'N/A'} | {st.type or 'N/A'} | "
                f"{st.budget or 'N/A'} | LKR {st.price_per_night or '?'}/night | "
                f"Rating {st.avg_rating or '?'}"
            )

    def _fmt_activities(activities) -> None:
        if not activities:
            return
        lines.append("Activities:")
        for a in activities:
            lines.append(
                f"  {a.name} | {a.category or 'N/A'} | "
                f"{a.difficulty_level or 'N/A'} | LKR {a.base_price or '?'}"
            )

    try:
        if entity == "guide":
            result = rec_engine.recommend_guides(tourist_id=tourist_id, city=city, top_k=3)
            _fmt_guides(result.guides)

        elif entity == "stay":
            result = rec_engine.recommend_stays(tourist_id=tourist_id, city=city, top_k=3)
            _fmt_stays(result.stays)

        elif entity == "activity":
            result = rec_engine.recommend_activities(tourist_id=tourist_id, city=city, top_k=3)
            _fmt_activities(result.activities)
            # Enrich: find guides/stays linked to these activities via local_activity
            activity_ids = [a.activity_id for a in result.activities]
            linked = _fetch_linked_providers(activity_ids)
            if linked:
                lines.append("")  # blank separator
                lines.append(linked)

        else:
            # Vague request — run all three (original behaviour)
            result = rec_engine.recommend(tourist_id=tourist_id, city=city, top_k=3)
            _fmt_guides(result.guides)
            _fmt_stays(result.stays)
            _fmt_activities(result.activities)

    except Exception as e:
        log.warning("rec_engine failed (entity=%s): %s", entity, e)
        return ""

    return "\n".join(lines) if lines else ""


def _fetch_doc_context(message: str) -> str:
    """
    Embeds the user message and retrieves top-4 doc_chunk rows via pgvector.
    Returns empty string if nothing found or on failure.
    """
    try:
        query_vec = embed(message)
        result = get_supabase().rpc(
            "match_doc_chunks",
            {"query_embedding": query_vec, "category_filter": None, "match_count": 4},
        ).execute()
        chunks = result.data or []
        if not chunks:
            return ""
        return "\n\n".join(f"[{c['doc_name']}]\n{c['content']}" for c in chunks)
    except Exception as e:
        log.warning("Doc search failed: %s", e)
        return ""


def _tourist_context(tourist_id: str) -> str:
    """Compact tourist profile line injected into system prompt."""
    tp = (
        get_supabase()
        .table("tourist_profile")
        .select("travel_style, budget, active_level")
        .eq("id", tourist_id)
        .maybe_single()
        .execute()
    ).data
    if not tp:
        return ""
    return (
        f"Tourist profile: style={tp.get('travel_style')}, "
        f"budget={tp.get('budget')}, activity_level={tp.get('active_level')}."
    )


# ── System prompt builders ────────────────────────────────────────────────────
# One builder per scenario. Each is tight — only the context that path needs.

_BASE = """Rules:
- Reply in the SAME language the user wrote in (English / Sinhala / Tamil).
- Be warm and concise. Do not repeat yourself.
- Never ask more than ONE follow-up question per reply."""


def _prompt_recommend_no_city(tourist_ctx: str, entity: str) -> str:
    entity_label = entity or "options"
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).
Yaloo connects tourists with local volunteer guides and family homestays in Sri Lanka.

{tourist_ctx}

The user wants {entity_label} recommendations but has not mentioned which city they are visiting.
Ask ONLY: which city or area in Sri Lanka are they heading to?
Do NOT ask anything else. Do NOT give recommendations yet.

{_BASE}"""


def _prompt_recommend_with_data(tourist_ctx: str, city: str, rec_data: str) -> str:
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).
Yaloo connects tourists with local volunteer guides and family homestays in Sri Lanka.

{tourist_ctx}

The user wants recommendations in {city}. Use the live data below to answer.
Highlight 1–2 best options and briefly explain why they match this tourist.
Be selective and warm — do not dump the full list.

LIVE DATA:
{rec_data}

{_BASE}
- Do not invent prices, ratings, or names not in the live data."""


def _prompt_recommend_no_tourist(city: str) -> str:
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).
Yaloo connects tourists with local volunteer guides and family homestays in Sri Lanka.

The user wants recommendations in {city} but is not logged in.
Briefly describe what Yaloo offers (local guides, family homestays, activities).
Encourage them to sign up or log in for personalised matches.

{_BASE}"""


def _prompt_docs(chunks: str) -> str:
    if chunks:
        return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).

Answer the user's question using ONLY the documentation below. Be direct.
If the answer is not in the docs, say so honestly and suggest they contact support.

DOCUMENTATION:
{chunks}

{_BASE}"""
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).

No matching documentation was found for this question.
Apologise briefly and suggest the user contacts Yaloo support or checks the app.

{_BASE}"""


def _prompt_general(tourist_ctx: str) -> str:
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).
Yaloo connects tourists with local volunteer guides and family homestays in Sri Lanka.

{tourist_ctx}

Answer this general Sri Lanka travel question from your own knowledge.
Be helpful, warm, and concise. If you are unsure, say so.

{_BASE}"""


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages list is empty")

    user_message = req.messages[-1].content

    # ── Step 1: Classify intent + extract city/entity from full history ────────
    # Intent is detected from the current message only (what does the user want NOW).
    # City and entity are scanned across the ENTIRE history so that a city named
    # in a previous turn (e.g. answering "Which city?") is not lost.
    intent_type = _detect_intent(user_message)
    city        = _extract_city_from_history(req.messages)   # scans all turns
    entity      = _detect_entity(req.messages) if intent_type == "recommend" else None
    tourist_ctx = _tourist_context(req.tourist_id) if req.tourist_id else ""

    log.info(
        "Intent: %s | Entity: %s | City: %s | tourist_id: %s",
        intent_type, entity, city, req.tourist_id,
    )

    # ── Step 2: Build system prompt + fetch context if needed ─────────────────
    # Context is only fetched when we actually need it — never speculatively.

    if intent_type == "recommend":
        if not city:
            # No city anywhere in conversation → ask for it once. Zero data fetch.
            system_prompt = _prompt_recommend_no_city(tourist_ctx, entity or "travel")

        elif req.tourist_id:
            # City known + logged in → fetch only the needed entity's data
            rec_data = _fetch_recommendation_context(req.tourist_id, city, entity)
            if rec_data:
                system_prompt = _prompt_recommend_with_data(tourist_ctx, city, rec_data)
            else:
                # rec_engine returned nothing (embeddings not ready / all filtered)
                system_prompt = _prompt_recommend_no_tourist(city)
        else:
            # City known, not logged in → generic pitch
            system_prompt = _prompt_recommend_no_tourist(city)

    elif intent_type == "docs":
        chunks = _fetch_doc_context(user_message)
        system_prompt = _prompt_docs(chunks)

    else:
        # general — no data fetch at all
        system_prompt = _prompt_general(tourist_ctx)

    # ── Step 3: Build conversation history (all turns except the last) ─────────
    history = []
    for msg in req.messages[:-1]:
        history.append({
            "role":  "user" if msg.role == "user" else "model",
            "parts": [msg.content],
        })

    # ── Step 4: Gemini call with fallback tier chain ──────────────────────────
    # api_fallback walks primary → secondary → tertiary automatically.
    # Each tier gets up to 3 attempts before advancing to the next key.
    # On success, resets back to primary for the next request.
    last_error: Exception | None = None
    max_attempts = 9  # 3 tiers × 3 retries each

    for attempt in range(max_attempts):
        try:
            current_key  = api_config.get_current_api_key()
            model        = _make_model(system_prompt, current_key)
            chat_session = model.start_chat(history=history)
            response     = chat_session.send_message(user_message)
            api_config.reset_to_primary()   # success — reset for next request
            return ChatResponse(reply=response.text.strip(), sources=[])

        except Exception as e:
            last_error = e
            err = str(e)

            log.warning(
                "Gemini attempt %d/%d failed on %s: %s",
                attempt + 1, max_attempts,
                api_config.get_status()["api_name"], err[:120],
            )

            should_retry = api_config.handle_api_error(e)
            if not should_retry:
                break

            # Brief pause before next attempt — exponential up to 10s
            import asyncio
            await asyncio.sleep(min(2 ** attempt, 10))

    # All tiers and retries exhausted
    err_str = str(last_error) if last_error else ""
    if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
        match = re.search(r"retry in (\d+)", err_str, re.IGNORECASE)
        retry_msg = f" Please try again in {match.group(1)} seconds." if match else ""
        raise HTTPException(
            status_code=429,
            detail=f"All API keys have reached their daily limit.{retry_msg}",
        )
    log.exception("All Gemini tiers exhausted. Last error: %s", last_error)
    raise HTTPException(status_code=502, detail="Chat service temporarily unavailable")
