"""
app/api/chatbot.py

POST /chat  — Single-call intent-routed RAG chatbot.

KEY DESIGN: ONE Gemini call per user message (not two).

State is derived from the FULL conversation history on every request
(city, intent, entity type). This is the correct approach because:
  - The frontend sends the full message list on every POST /chat call.
  - There is no server-side session; all state lives in the message list.
  - Scanning history is zero API cost (pure Python keyword matching).

Intent / entity / city resolution rules:
  intent  — detected from the CURRENT message first; if "general", fall back
             to scanning user turns in history (oldest → newest) so that a
             bare city reply like "Kandy" doesn't erase the earlier intent.
  entity  — scanned across ALL user turns (oldest → newest); most recent wins.
  city    — scanned across ALL turns (oldest → newest); most recent wins.

Token budget per request:
  general            : ~200-400 tokens  (system + message + reply)
  docs               : ~700-900 tokens  (system + chunks + reply)
  recommend (no city): ~200 tokens      (asks for city, no data fetch)
  recommend (city)   : ~700-1000 tokens (system + live data + reply)

Multi-language: Gemini handles English, Sinhala, Tamil natively.
"""
import logging
import re
from typing import List, Optional

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


# -- Gemini model factory -----------------------------------------------------

def _make_model(system_instruction: str, api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
    )


# -- Known cities -------------------------------------------------------------

_SRI_LANKA_CITIES = [
    "colombo", "kandy", "galle", "ella", "sigiriya", "mirissa",
    "negombo", "trincomalee", "jaffna", "nuwara eliya", "arugam bay",
    "batticaloa", "ratnapura", "kurunegala", "matara", "badulla",
]


# -- Keyword lists ------------------------------------------------------------

_RECOMMEND_KEYWORDS = [
    "guide", "stay", "homestay", "hotel", "accommodation", "activity",
    "activities", "recommend", "find", "suggest", "book", "visit",
    "tour", "place", "where", "what to do", "things to do",
]

_DOC_KEYWORDS = [
    "policy", "cancel", "cancellation", "refund", "fee", "cost", "price",
    "how does yaloo work", "how yaloo works", "booking", "payment", "safe",
    "safety", "verify", "verified", "sltda", "contact", "support",
    "register", "sign up", "sign-up", "account",
]

_GUIDE_KEYWORDS    = ["guide", "guides", "local guide", "tour guide"]
_STAY_KEYWORDS     = ["stay", "stays", "homestay", "homestays", "hotel",
                      "accommodation", "lodging", "place to stay", "room", "host"]
_ACTIVITY_KEYWORDS = ["activity", "activities", "things to do", "what to do",
                      "experience", "experiences", "adventure", "hike", "hiking",
                      "excursion"]


# -- History-aware scanners (all zero API cost) --------------------------------

def _extract_city(text: str) -> Optional[str]:
    lower = text.lower()
    for city in _SRI_LANKA_CITIES:
        if city in lower:
            return city.title()
    return None


def _scan_city(messages) -> Optional[str]:
    """Scan ALL turns oldest->newest; return most recently mentioned city."""
    found = None
    for msg in messages:
        c = _extract_city(msg.content)
        if c:
            found = c
    return found


def _detect_intent_text(text: str) -> str:
    """Classify a single text string. Returns 'recommend', 'docs', or 'general'."""
    lower = text.lower()
    if any(kw in lower for kw in _RECOMMEND_KEYWORDS):
        return "recommend"
    if any(kw in lower for kw in _DOC_KEYWORDS):
        return "docs"
    return "general"


def _scan_intent(messages) -> str:
    """
    Determine intent for the current turn.

    Strategy:
      1. Classify the current (last) user message.
      2. If it is 'general', walk prior USER turns newest->oldest and inherit
         the first non-general intent found.
      This handles bare replies like "Kandy" or "ok" that have no intent
      keywords but belong to an active recommendation conversation.
    """
    user_msgs = [m for m in messages if m.role == "user"]
    if not user_msgs:
        return "general"

    current_intent = _detect_intent_text(user_msgs[-1].content)
    if current_intent != "general":
        return current_intent

    # Fallback: walk history newest->oldest skipping the last message
    for msg in reversed(user_msgs[:-1]):
        intent = _detect_intent_text(msg.content)
        if intent != "general":
            return intent

    return "general"


def _scan_entity(messages) -> Optional[str]:
    """
    Scan ALL user turns oldest->newest for entity keywords.
    Most recent specific mention wins.
    Returns 'guide', 'stay', 'activity', or None (all three).
    """
    found = None
    for msg in messages:
        if msg.role != "user":
            continue
        lower = msg.content.lower()
        if any(kw in lower for kw in _GUIDE_KEYWORDS):
            found = "guide"
        elif any(kw in lower for kw in _STAY_KEYWORDS):
            found = "stay"
        elif any(kw in lower for kw in _ACTIVITY_KEYWORDS):
            found = "activity"
    return found


# -- Context fetchers ---------------------------------------------------------

def _fetch_linked_providers(activity_ids: list) -> str:
    """
    For a list of activity IDs, find guides and stays linked via local_activity.
    Returns a compact summary block, or empty string on failure / nothing found.
    """
    if not activity_ids:
        return ""
    try:
        db = get_supabase()
        rows = (
            db.table("local_activity")
            .select("activity_id, guide_id, stay_id")
            .in_("activity_id", activity_ids)
            .execute()
        ).data or []
        if not rows:
            return ""

        guide_ids = list({r["guide_id"] for r in rows if r.get("guide_id")})
        stay_ids  = list({r["stay_id"]  for r in rows if r.get("stay_id")})
        lines: List[str] = []

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
    Call only the rec_engine function needed for the detected entity type.
    entity: 'guide' | 'stay' | 'activity' | None (all three)
    Returns empty string on failure.
    """
    lines: List[str] = []

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
            activity_ids = [a.activity_id for a in result.activities]
            linked = _fetch_linked_providers(activity_ids)
            if linked:
                lines.append("")
                lines.append(linked)

        else:
            result = rec_engine.recommend(tourist_id=tourist_id, city=city, top_k=3)
            _fmt_guides(result.guides)
            _fmt_stays(result.stays)
            _fmt_activities(result.activities)

    except Exception as e:
        log.warning("rec_engine failed (entity=%s): %s", entity, e)
        return ""

    return "\n".join(lines) if lines else ""


def _fetch_doc_context(message: str) -> str:
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


# -- System prompt builders ---------------------------------------------------

_BASE = """Rules:
- Reply in the SAME language the user wrote in (English / Sinhala / Tamil).
- Be warm and concise. Do not repeat yourself.
- Do NOT ask any follow-up questions unless you are explicitly waiting for the city name."""


def _prompt_recommend_no_city(tourist_ctx: str, entity: str) -> str:
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).
Yaloo connects tourists with local volunteer guides and family homestays in Sri Lanka.

{tourist_ctx}

The user wants {entity} recommendations but has not mentioned which city they are visiting.
Ask ONLY this: which city or area in Sri Lanka are they heading to?
Do NOT ask anything else. Do NOT give recommendations yet. One question only.

{_BASE}"""


def _prompt_recommend_with_data(tourist_ctx: str, city: str, rec_data: str) -> str:
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).
Yaloo connects tourists with local volunteer guides and family homestays in Sri Lanka.

{tourist_ctx}

The user wants recommendations in {city}. Use ONLY the live data below to answer.
Highlight 1-2 best options and briefly explain why they suit this tourist.
Be warm and selective. Do not ask any follow-up questions.

LIVE DATA:
{rec_data}

{_BASE}
- Do not invent prices, ratings, or names not in the live data."""


def _prompt_recommend_no_tourist(city: str) -> str:
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).
Yaloo connects tourists with local volunteer guides and family homestays in Sri Lanka.

The user wants recommendations in {city} but is not logged in.
Briefly describe what Yaloo offers (local guides, family homestays, activities).
Encourage them to sign up or log in for personalised matches. Do not ask questions.

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


# -- Chat endpoint ------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages list is empty")

    user_message = req.messages[-1].content

    # -- Step 1: Derive state from FULL history — zero API cost ---------------
    #
    # All three values are scanned across the entire message list so that
    # context established in earlier turns is never lost.
    #
    # Example flow that previously broke:
    #   Turn 1 (user):  "recommend a guide"     -> intent=recommend, entity=guide, city=None
    #   Turn 2 (model): "Which city?"
    #   Turn 3 (user):  "Colombo"               <- current message, no intent keywords
    #     OLD: _detect_intent("Colombo") = "general"  -> wrong branch, lost everything
    #     NEW: _scan_intent walks history, finds "recommend" from Turn 1 -> correct

    intent_type = _scan_intent(req.messages)
    city        = _scan_city(req.messages)
    entity      = _scan_entity(req.messages) if intent_type == "recommend" else None
    tourist_ctx = _tourist_context(req.tourist_id) if req.tourist_id else ""

    log.info(
        "Intent: %s | Entity: %s | City: %s | tourist_id: %s",
        intent_type, entity, city, req.tourist_id,
    )

    # -- Step 2: Build system prompt + fetch context if needed ----------------

    if intent_type == "recommend":
        if not city:
            # No city anywhere in conversation -> ask for it once. Zero data fetch.
            system_prompt = _prompt_recommend_no_city(tourist_ctx, entity or "travel")

        elif req.tourist_id:
            # City known + logged in -> fetch only the needed entity's data
            rec_data = _fetch_recommendation_context(req.tourist_id, city, entity)
            if rec_data:
                system_prompt = _prompt_recommend_with_data(tourist_ctx, city, rec_data)
            else:
                system_prompt = _prompt_recommend_no_tourist(city)
        else:
            # City known, not logged in -> generic pitch
            system_prompt = _prompt_recommend_no_tourist(city)

    elif intent_type == "docs":
        chunks = _fetch_doc_context(user_message)
        system_prompt = _prompt_docs(chunks)

    else:
        system_prompt = _prompt_general(tourist_ctx)

    # -- Step 3: Build conversation history (all turns except the last) -------
    history = []
    for msg in req.messages[:-1]:
        history.append({
            "role":  "user" if msg.role == "user" else "model",
            "parts": [msg.content],
        })

    # -- Step 4: Gemini call with fallback tier chain -------------------------
    last_error: Exception | None = None
    max_attempts = 9  # 3 tiers x 3 retries each

    for attempt in range(max_attempts):
        try:
            current_key  = api_config.get_current_api_key()
            model        = _make_model(system_prompt, current_key)
            chat_session = model.start_chat(history=history)
            response     = chat_session.send_message(user_message)
            api_config.reset_to_primary()
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

            import asyncio
            await asyncio.sleep(min(2 ** attempt, 10))

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
