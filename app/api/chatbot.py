"""
app/api/chatbot.py

POST /chat

Intent-routed RAG chatbot — three paths, minimal questions asked:

  recommend → calls rec_engine with live data, injects results into prompt.
              ONLY asks for city if missing. Never asks for anything else.
  docs      → embeds user message, retrieves top-4 doc_chunk rows via pgvector.
              Answers directly from those chunks.
  general   → answers directly from Gemini training knowledge.
              Zero context injection, lowest token cost.

Token budget per request (approximate):
  Classifier call  :  ~100 tokens   (always)
  general path     :  ~200 tokens   (no injection)
  docs path        :  ~700 tokens   (200 system + 400 chunks + 100 reply)
  recommend path   :  ~900 tokens   (200 system + 350 data + 350 reply)
  recommend (no city): ~200 tokens  (just asks for city — no data fetch at all)

Multi-language: English, Sinhala, Tamil handled automatically by Gemini.
"""
import json
import logging
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
# Returns a fresh GenerativeModel with system_instruction baked in.
# Called once per /chat request — avoids re-configuring the API key on every
# sub-call like _classify_intent was doing before.

def _make_model(system_instruction: str) -> genai.GenerativeModel:
    s = get_settings()
    if not s.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=s.gemini_api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
    )


# ── Intent classifier ─────────────────────────────────────────────────────────
# Tiny dedicated call — max 30 output tokens, temperature 0 for determinism.
# Separate from the main model so it doesn't pollute conversation history.

CLASSIFIER_PROMPT = """You are a query classifier for Yaloo, a Sri Lanka tourism platform.
Classify the user message into exactly one category.
Also extract the city name if the user mentions one, otherwise return null.

Categories:
- "recommend" → user wants to find guides, stays/homestays, or activities
- "docs"       → user asks about Yaloo policies, booking process, fees, cancellation, safety, how Yaloo works
- "general"    → general Sri Lanka travel questions, culture, weather, food, transport, tips

Reply ONLY with a valid JSON object. No explanation. No markdown. No backticks.
Format: {"type": "recommend"|"docs"|"general", "city": "city name or null"}

User message: "{message}"
"""


def _classify_intent(message: str) -> dict:
    """
    One small Gemini call to classify intent and extract city.
    Falls back to {"type": "general", "city": null} on any failure.
    Uses a bare GenerativeModel (no system_instruction) to keep it lean.
    """
    try:
        s = get_settings()
        genai.configure(api_key=s.gemini_api_key)
        classifier = genai.GenerativeModel("gemini-2.5-flash")
        response = classifier.generate_content(
            CLASSIFIER_PROMPT.format(message=message),
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=30,
                temperature=0.0,
            ),
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        if result.get("type") not in ("recommend", "docs", "general"):
            result["type"] = "general"
        return result
    except Exception as e:
        log.warning("Intent classifier failed (%s) — defaulting to general", e)
        return {"type": "general", "city": None}


# ── Context fetchers ──────────────────────────────────────────────────────────

def _fetch_recommendation_context(tourist_id: str, city: str) -> str:
    """
    Call rec_engine and format top-3 results per entity as a compact text block.
    Only called when tourist_id AND city are both known — never speculatively.
    Returns empty string on any failure so the bot still responds gracefully.
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

    if not lines:
        return ""

    return "\n".join(lines)


def _fetch_doc_context(message: str) -> str:
    """
    Embed message → retrieve top-4 doc_chunk rows via pgvector KNN.
    Returns empty string if no chunks found or on failure.
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
        return "\n\n".join(
            f"[{c['doc_name']}]\n{c['content']}" for c in chunks
        )
    except Exception as e:
        log.warning("Doc search failed: %s", e)
        return ""


def _tourist_context(tourist_id: str) -> str:
    """Single-line tourist profile summary — injected into system prompt."""
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
        f"Tourist: style={tp.get('travel_style')}, "
        f"budget={tp.get('budget')}, activity={tp.get('active_level')}."
    )


# ── System prompt builders ────────────────────────────────────────────────────
# One builder per intent path. Each produces a tight system_instruction
# with exactly the context that path needs — nothing more.

_YALOO_INTRO = "You are Yaloo Assistant (යාළු = Friend in Sinhala), helping tourists explore Sri Lanka."
_BASE_RULES = "Reply in the user's language. Be concise. Ask at most one question per reply."


def _system_recommend_no_city(tourist_ctx: str) -> str:
    ctx_line = f"\n{tourist_ctx}" if tourist_ctx else ""
    return f"""{_YALOO_INTRO}{ctx_line}
Ask the user which city or area in Sri Lanka they plan to visit. One sentence only. No recommendations yet.
{_BASE_RULES}"""


def _system_recommend_with_data(tourist_ctx: str, city: str, rec_data: str) -> str:
    ctx_line = f"\n{tourist_ctx}" if tourist_ctx else ""
    return f"""{_YALOO_INTRO}{ctx_line}
Recommend options in {city} using ONLY the data below. Highlight the best 1–3 guide/stay fits for this tourist and briefly say why.
For activities, list ALL of them and mention who offers each one.
If guides or stays have no results, tell the user none are available right now and suggest they check back or try different dates/filters.
Never invent prices, ratings, or names not in the data.

DATA:
{rec_data}

{_BASE_RULES}"""


def _system_recommend_no_tourist(city: str) -> str:
    return f"""{_YALOO_INTRO}
The user wants recommendations in {city} but is not logged in.
Briefly describe what Yaloo offers (local guides, homestays, activities) and encourage sign-up for personalised picks.
{_BASE_RULES}"""


def _system_docs(chunks: str) -> str:
    if chunks:
        return f"""{_YALOO_INTRO}
Answer using ONLY the documentation below. If the answer isn't there, say so and suggest contacting Yaloo support.

DOCS:
{chunks}

{_BASE_RULES}"""
    else:
        return f"""{_YALOO_INTRO}
No matching documentation found. Apologise briefly and suggest the user contact Yaloo support or check the app.
{_BASE_RULES}"""


def _system_general() -> str:
    return f"""{_YALOO_INTRO}
Answer this general Sri Lanka travel question from your knowledge. Be helpful and concise.
{_BASE_RULES}"""


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages list is empty")

    user_message = req.messages[-1].content

    # ── Step 1: Classify intent (always, ~100 tokens) ─────────────────────────
    intent     = _classify_intent(user_message)
    intent_type = intent.get("type", "general")
    city        = intent.get("city")  # may be None
    log.info("Intent: %s | City: %s | tourist_id: %s", intent_type, city, req.tourist_id)

    # ── Step 2: Build system prompt for this specific request ─────────────────
    if intent_type == "recommend":
        tourist_ctx = _tourist_context(req.tourist_id) if req.tourist_id else ""
        if not city:
            # City unknown → just ask. Zero data fetch. Cheapest possible path.
            system_prompt = _system_recommend_no_city(tourist_ctx)

        elif req.tourist_id:
            # City known + logged in → fetch live rec data and inject
            rec_data = _fetch_recommendation_context(req.tourist_id, city)
            if rec_data:
                system_prompt = _system_recommend_with_data(tourist_ctx, city, rec_data)
            else:
                # rec_engine returned nothing (no embeddings yet / all filtered out)
                system_prompt = _system_recommend_no_tourist(city)
        else:
            # City known but not logged in → generic Yaloo pitch
            system_prompt = _system_recommend_no_tourist(city)

    elif intent_type == "docs":
        chunks = _fetch_doc_context(user_message)
        system_prompt = _system_docs(chunks)

    else:
        # general — no injection at all
        system_prompt = _system_general()

    # ── Step 3: Build conversation history (all turns except the last) ─────────
    history = []
    for msg in req.messages[:-1]:
        history.append({
            "role":  "user" if msg.role == "user" else "model",
            "parts": [msg.content],
        })

    # ── Step 4: Call Gemini with system_instruction baked in ──────────────────
    try:
        model        = _make_model(system_prompt)
        chat_session = model.start_chat(history=history)
        response     = chat_session.send_message(user_message)
        return ChatResponse(reply=response.text.strip(), sources=[])

    except Exception:
        log.exception("Gemini error")
        raise HTTPException(status_code=502, detail="Chat service temporarily unavailable")
