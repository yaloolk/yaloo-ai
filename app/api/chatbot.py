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

log = logging.getLogger(__name__)
router = APIRouter()


# ── Gemini model factory ──────────────────────────────────────────────────────

def _make_model(system_instruction: str) -> genai.GenerativeModel:
    s = get_settings()
    if not s.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=s.gemini_api_key)
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
    Scan message text for a known Sri Lanka city name.
    Returns the properly capitalised city name, or None if not found.
    """
    lower = text.lower()
    for city in _SRI_LANKA_CITIES:
        if city in lower:
            return city.title()
    return None


# ── Intent detector — zero API cost ──────────────────────────────────────────
# Keyword-based classification. Fast, free, deterministic.
# Handles English, and common Sinhala/Tamil romanisations reasonably well
# since tourists typically mix English words when asking about services.

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


# ── Context fetchers ──────────────────────────────────────────────────────────

def _fetch_recommendation_context(tourist_id: str, city: str) -> str:
    """
    Calls rec_engine and formats top-3 results per entity type.
    Returns empty string on failure — bot will degrade gracefully.
    """
    try:
        result = rec_engine.recommend(tourist_id=tourist_id, city=city, top_k=3)
    except Exception as e:
        log.warning("rec_engine failed: %s", e)
        return ""

    lines = []

    if result.guides:
        lines.append("Guides:")
        for g in result.guides:
            lines.append(
                f"  {g.full_name} | {g.city_name or 'N/A'} | "
                f"Rating {g.avg_rating or '?'} | {g.experience_years or '?'} yrs | "
                f"LKR {g.rate_per_hour or '?'}/hr | {g.languages or 'N/A'}"
            )

    if result.stays:
        lines.append("Stays:")
        for st in result.stays:
            lines.append(
                f"  {st.name} | {st.city_name or 'N/A'} | {st.type or 'N/A'} | "
                f"{st.budget or 'N/A'} | LKR {st.price_per_night or '?'}/night | "
                f"Rating {st.avg_rating or '?'}"
            )

    if result.activities:
        lines.append("Activities:")
        for a in result.activities:
            lines.append(
                f"  {a.name} | {a.category or 'N/A'} | "
                f"{a.difficulty_level or 'N/A'} | LKR {a.base_price or '?'}"
            )

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


def _prompt_recommend_no_city(tourist_ctx: str) -> str:
    return f"""You are Yaloo's AI travel assistant (යාළු means Friend in Sinhala).
Yaloo connects tourists with local volunteer guides and family homestays in Sri Lanka.

{tourist_ctx}

The user wants a recommendation but has not mentioned which city they are visiting.
Ask them which city or area in Sri Lanka they are heading to. One question only.
Do not give recommendations yet.

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

    # ── Step 1: Classify intent — zero API cost ───────────────────────────────
    intent_type = _detect_intent(user_message)
    city        = _extract_city(user_message)
    tourist_ctx = _tourist_context(req.tourist_id) if req.tourist_id else ""

    log.info(
        "Intent: %s | City: %s | tourist_id: %s",
        intent_type, city, req.tourist_id,
    )

    # ── Step 2: Build system prompt + fetch context if needed ─────────────────
    # Context is only fetched when we actually need it — never speculatively.

    if intent_type == "recommend":
        if not city:
            # No city in message → ask for it. Zero data fetch.
            system_prompt = _prompt_recommend_no_city(tourist_ctx)

        elif req.tourist_id:
            # City + logged in → fetch live rec data
            rec_data = _fetch_recommendation_context(req.tourist_id, city)
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

    # ── Step 4: ONE Gemini call ────────────────────────────────────────────────
    try:
        model        = _make_model(system_prompt)
        chat_session = model.start_chat(history=history)
        response     = chat_session.send_message(user_message)
        return ChatResponse(reply=response.text.strip(), sources=[])

    except Exception as e:
        err = str(e)
        if "RESOURCE_EXHAUSTED" in err or "429" in err:
            # Extract retry delay from the error message if present
            match = re.search(r"retry in (\d+)", err, re.IGNORECASE)
            retry_msg = f" Please try again in {match.group(1)} seconds." if match else ""
            log.warning("Gemini quota exhausted.%s", retry_msg)
            raise HTTPException(
                status_code=429,
                detail=f"The AI assistant has hit its daily request limit.{retry_msg}",
            )
        log.exception("Gemini error")
        raise HTTPException(status_code=502, detail="Chat service temporarily unavailable")
