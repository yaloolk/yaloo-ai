"""
app/api/chatbot.py

POST /chat  — Single-call intent-routed RAG chatbot.

KEY DESIGN: ONE Gemini call per user message (not two).

State resolution from full conversation history (zero API cost):

  intent  — Current message is classified first. "general" fallback to history
             ONLY if the bot is still mid-flow waiting for a city (i.e. city has
             not been provided yet). Once a recommendation has been delivered
             (city known, rec context fetched), the next unrelated message like
             "what's the climate?" correctly classifies as "general" and breaks
             out of recommend mode.

  entity  — Scanned across all user turns oldest→newest; most recent wins.

  city    — Scanned across all turns oldest→newest; most recent wins.

  date    — Scanned across all user turns; extracted via regex. Used to filter
             available_guide_ids / available_stay_ids before rec_engine call.
"""
import logging
import re
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import google.generativeai as genai
from fastapi import APIRouter, HTTPException

from app.core.database import get_supabase
from app.schemas.payloads import ChatRequest, ChatResponse
from app.services import rec_engine
from app.services.vector_service import embed
from app.core.api_fallback import api_config

log = logging.getLogger(__name__)
router = APIRouter()


# ── Gemini model factory ──────────────────────────────────────────────────────

def _make_model(system_instruction: str, api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
    )


# ── Keyword lists ─────────────────────────────────────────────────────────────

_SRI_LANKA_CITIES = [
    "colombo", "kandy", "galle", "ella", "sigiriya", "mirissa",
    "negombo", "trincomalee", "jaffna", "nuwara eliya", "arugam bay",
    "batticaloa", "ratnapura", "kurunegala", "matara", "badulla",
]

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


# ── History-aware scanners ────────────────────────────────────────────────────

def _extract_city(text: str) -> Optional[str]:
    lower = text.lower()
    for city in _SRI_LANKA_CITIES:
        if city in lower:
            return city.title()
    return None


def _scan_city(messages) -> Optional[str]:
    """Scan ALL turns oldest→newest; return most recently mentioned city."""
    found = None
    for msg in messages:
        c = _extract_city(msg.content)
        if c:
            found = c
    return found


def _detect_intent_text(text: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in _RECOMMEND_KEYWORDS):
        return "recommend"
    if any(kw in lower for kw in _DOC_KEYWORDS):
        return "docs"
    return "general"


def _scan_intent(messages) -> str:
    """
    Determine intent for the current turn.

    The key rule: only inherit 'recommend' from history if the conversation
    is STILL in the pending state — i.e. the bot asked for a city and has
    not yet delivered a recommendation.

    'Pending' means: recommend intent was found in history AND no city has
    appeared in any turn up to (but not including) the current user message.

    Once a city was provided and a recommendation delivered, the tourist is
    free to ask anything else and it classifies on its own merits.
    """
    user_msgs = [m for m in messages if m.role == "user"]
    if not user_msgs:
        return "general"

    # Always classify the current message first
    current_intent = _detect_intent_text(user_msgs[-1].content)
    if current_intent != "general":
        return current_intent

    # Current message is "general" (e.g. bare city reply "Colombo", or "what's the climate?")
    # Only inherit recommend from history if we are still waiting for city input.
    # "Waiting" = recommend found in a prior turn AND no city found in any prior turn.
    prior_msgs = messages[:-1]  # all turns before the current user message
    prior_user_msgs = [m for m in prior_msgs if m.role == "user"]

    prior_had_recommend = any(
        _detect_intent_text(m.content) == "recommend" for m in prior_user_msgs
    )
    prior_had_city = any(
        _extract_city(m.content) for m in prior_msgs  # includes bot turns
    )

    if prior_had_recommend and not prior_had_city:
        # Bot asked for city, tourist is now replying with it — stay in recommend
        return "recommend"

    # Otherwise: recommendation already delivered, or no prior recommend.
    # Let the current message stand as "general".
    return "general"


def _scan_entity(messages) -> Optional[str]:
    """Scan all user turns oldest→newest; most recent entity keyword wins."""
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


# ── Date extractor ────────────────────────────────────────────────────────────
# Scans user turns for a travel date. Supports common formats and relative
# expressions. Returns an ISO date string (YYYY-MM-DD) or None.

# Relative-word patterns (today / tomorrow / next <weekday>)
_REL_DATE_PATTERNS = [
    (re.compile(r"\btoday\b", re.I),    lambda: date.today()),
    (re.compile(r"\btomorrow\b", re.I), lambda: date.today() + timedelta(days=1)),
]

_WEEKDAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
_MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "january":1,"february":2,"march":3,"april":4,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}

# Absolute date patterns: "15 March", "March 15", "15/03/2025", "2025-03-15"
_ABS_DATE_PATTERNS = [
    # ISO: 2025-03-15
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),
    # d/m/yyyy or d/m/yy
    re.compile(r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b"),
    # "15 March" / "15th March" / "March 15" / "March 15th"
    re.compile(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        re.I,
    ),
    re.compile(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        re.I,
    ),
]


def _parse_abs_date(m: re.Match, pattern_idx: int) -> Optional[date]:
    """Try to parse a regex match into a date. Returns None on failure."""
    today = date.today()
    try:
        g = m.groups()
        if pattern_idx == 0:          # ISO yyyy-mm-dd
            return date(int(g[0]), int(g[1]), int(g[2]))
        if pattern_idx == 1:          # d/m/yyyy
            d, mo, y = int(g[0]), int(g[1]), int(g[2])
            if y < 100:
                y += 2000
            return date(y, mo, d)
        if pattern_idx == 2:          # "15 March"
            d  = int(g[0])
            mo = _MONTH_MAP.get(g[1].lower()[:3])
            if not mo:
                return None
            y = today.year
            candidate = date(y, mo, d)
            if candidate < today:
                candidate = date(y + 1, mo, d)
            return candidate
        if pattern_idx == 3:          # "March 15"
            mo = _MONTH_MAP.get(g[0].lower()[:3])
            d  = int(g[1])
            if not mo:
                return None
            y = today.year
            candidate = date(y, mo, d)
            if candidate < today:
                candidate = date(y + 1, mo, d)
            return candidate
    except (ValueError, TypeError):
        return None
    return None


def _scan_date(messages) -> Optional[str]:
    """
    Scan all user turns oldest→newest for a travel date mention.
    Returns ISO string 'YYYY-MM-DD' of the most recently mentioned date, or None.
    """
    found: Optional[date] = None

    for msg in messages:
        if msg.role != "user":
            continue
        text = msg.content

        # Relative expressions
        for pattern, resolver in _REL_DATE_PATTERNS:
            if pattern.search(text):
                found = resolver()

        # "next <weekday>"
        nw = re.search(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text, re.I)
        if nw:
            target_wd = _WEEKDAYS.index(nw.group(1).lower())
            today_wd  = date.today().weekday()
            delta     = (target_wd - today_wd) % 7 or 7
            found = date.today() + timedelta(days=delta)

        # Absolute date patterns
        for idx, pat in enumerate(_ABS_DATE_PATTERNS):
            m = pat.search(text)
            if m:
                parsed = _parse_abs_date(m, idx)
                if parsed:
                    found = parsed

    return found.isoformat() if found else None


# ── Availability helpers ──────────────────────────────────────────────────────

def _available_guide_ids(city: Optional[str], travel_date: Optional[str]) -> Optional[List[str]]:
    """
    Return guide_profile IDs available on travel_date in city.
    Returns None (no filter) if date is not provided.
    Returns [] if date provided but no guides available (hard filter).
    """
    if not travel_date:
        return None
    try:
        db = get_supabase()
        q  = (
            db.table("guide_availability")
            .select("guide_profile_id")
            .eq("available_date", travel_date)
            .eq("is_available", True)
        )
        if city:
            # Join via guide_profile to filter by city_name
            # Supabase doesn't support cross-table filters in .select directly,
            # so fetch all available guides then intersect with city filter below.
            pass
        rows = q.execute().data or []
        ids  = [r["guide_profile_id"] for r in rows]

        if city and ids:
            # Filter to guides in the requested city
            city_rows = (
                db.table("guide_profile")
                .select("id")
                .in_("id", ids)
                .ilike("city_name", city)
                .execute()
            ).data or []
            ids = [r["id"] for r in city_rows]

        log.info("Available guides on %s in %s: %d", travel_date, city, len(ids))
        return ids
    except Exception as e:
        log.warning("_available_guide_ids failed: %s", e)
        return None  # fail open — don't block recommendations


def _available_stay_ids(city: Optional[str], travel_date: Optional[str]) -> Optional[List[str]]:
    """
    Return stay IDs available on travel_date.
    Returns None if date not provided; [] if date provided but nothing available.
    """
    if not travel_date:
        return None
    try:
        db   = get_supabase()
        rows = (
            db.table("stay_availability")
            .select("stay_id")
            .eq("available_date", travel_date)
            .eq("is_available", True)
            .execute()
        ).data or []
        ids  = [r["stay_id"] for r in rows]

        if city and ids:
            city_rows = (
                db.table("stay")
                .select("id")
                .in_("id", ids)
                .ilike("city_name", city)
                .execute()
            ).data or []
            ids = [r["id"] for r in city_rows]

        log.info("Available stays on %s in %s: %d", travel_date, city, len(ids))
        return ids
    except Exception as e:
        log.warning("_available_stay_ids failed: %s", e)
        return None


# ── Context fetchers ──────────────────────────────────────────────────────────

def _fetch_linked_providers(activity_ids: list) -> str:
    if not activity_ids:
        return ""
    try:
        db    = get_supabase()
        rows  = (
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
                        f"  {g.get('full_name','?')} | "
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
                        f"  {st.get('name','?')} | "
                        f"{st.get('budget') or 'N/A'} | "
                        f"LKR {st.get('price_per_night') or '?'}/night | "
                        f"Rating {st.get('avg_rating') or '?'}"
                    )
        return "\n".join(lines)
    except Exception as e:
        log.warning("_fetch_linked_providers failed: %s", e)
        return ""


def _fetch_recommendation_context(
    tourist_id: str,
    city: str,
    entity: Optional[str],
    travel_date: Optional[str],
) -> str:
    """
    Call only the rec_engine function needed for the detected entity type.
    Passes availability-filtered IDs when a travel date was detected.
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
            avail = _available_guide_ids(city, travel_date)
            result = rec_engine.recommend_guides(
                tourist_id=tourist_id, city=city, top_k=3,
                available_guide_ids=avail,
            )
            _fmt_guides(result.guides)

        elif entity == "stay":
            avail = _available_stay_ids(city, travel_date)
            result = rec_engine.recommend_stays(
                tourist_id=tourist_id, city=city, top_k=3,
                available_stay_ids=avail,
            )
            _fmt_stays(result.stays)

        elif entity == "activity":
            result = rec_engine.recommend_activities(
                tourist_id=tourist_id, city=city, top_k=3,
            )
            _fmt_activities(result.activities)
            activity_ids = [a.activity_id for a in result.activities]
            linked = _fetch_linked_providers(activity_ids)
            if linked:
                lines.append("")
                lines.append(linked)

        else:
            # All three — apply availability filters to guides and stays
            avail_g = _available_guide_ids(city, travel_date)
            avail_s = _available_stay_ids(city, travel_date)
            result  = rec_engine.recommend(
                tourist_id=tourist_id, city=city, top_k=3,
                available_guide_ids=avail_g,
                available_stay_ids=avail_s,
            )
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
        result    = get_supabase().rpc(
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


# ── System prompt builders ────────────────────────────────────────────────────

_BASE = """Rules:
- Reply in the SAME language the user wrote in (English / Sinhala / Tamil).
- Be warm and concise. Do not repeat yourself.
- Do NOT ask any follow-up questions unless you are explicitly waiting for the city name."""


def _prompt_recommend_no_city(tourist_ctx: str, entity: str) -> str:
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).
Yaloo connects tourists with local volunteer guides and family homestays in Sri Lanka.

{tourist_ctx}

The user wants {entity} recommendations but has not mentioned which city they are visiting.
Ask ONLY: which city or area in Sri Lanka are they heading to?
Do NOT ask anything else. Do NOT give recommendations yet.

{_BASE}"""


def _prompt_recommend_with_data(
    tourist_ctx: str, city: str, rec_data: str, travel_date: Optional[str]
) -> str:
    date_note = f"These results are filtered for availability on {travel_date}." if travel_date else ""
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).
Yaloo connects tourists with local volunteer guides and family homestays in Sri Lanka.

{tourist_ctx}

The user wants recommendations in {city}. {date_note}
Use ONLY the live data below. Highlight 1-2 best options and briefly explain why they suit this tourist.
Be warm and selective. Do not ask follow-up questions.

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


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages list is empty")

    user_message = req.messages[-1].content

    # ── Step 1: Derive state from full history — zero API cost ────────────────
    intent_type  = _scan_intent(req.messages)
    city         = _scan_city(req.messages)
    entity       = _scan_entity(req.messages) if intent_type == "recommend" else None
    travel_date  = _scan_date(req.messages)   if intent_type == "recommend" else None
    tourist_ctx  = _tourist_context(req.tourist_id) if req.tourist_id else ""

    log.info(
        "Intent: %s | Entity: %s | City: %s | Date: %s | tourist_id: %s",
        intent_type, entity, city, travel_date, req.tourist_id,
    )

    # ── Step 2: Build system prompt + fetch context ───────────────────────────

    if intent_type == "recommend":
        if not city:
            system_prompt = _prompt_recommend_no_city(tourist_ctx, entity or "travel")

        elif req.tourist_id:
            rec_data = _fetch_recommendation_context(req.tourist_id, city, entity, travel_date)
            if rec_data:
                system_prompt = _prompt_recommend_with_data(tourist_ctx, city, rec_data, travel_date)
            else:
                system_prompt = _prompt_recommend_no_tourist(city)
        else:
            system_prompt = _prompt_recommend_no_tourist(city)

    elif intent_type == "docs":
        chunks = _fetch_doc_context(user_message)
        system_prompt = _prompt_docs(chunks)

    else:
        system_prompt = _prompt_general(tourist_ctx)

    # ── Step 3: Build conversation history (all turns except the last) ────────
    history = []
    for msg in req.messages[:-1]:
        history.append({
            "role":  "user" if msg.role == "user" else "model",
            "parts": [msg.content],
        })

    # ── Step 4: Gemini call with fallback tier chain ──────────────────────────
    last_error: Exception | None = None
    max_attempts = 9  # 3 tiers × 3 retries each

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
            err        = str(e)
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
        match     = re.search(r"retry in (\d+)", err_str, re.IGNORECASE)
        retry_msg = f" Please try again in {match.group(1)} seconds." if match else ""
        raise HTTPException(
            status_code=429,
            detail=f"All API keys have reached their daily limit.{retry_msg}",
        )
    log.exception("All Gemini tiers exhausted. Last error: %s", last_error)
    raise HTTPException(status_code=502, detail="Chat service temporarily unavailable")
