"""
app/api/chatbot.py

POST /chat

Upgraded RAG chatbot:
  - Intent classification via Gemini (recommend / docs / general)
  - Recommendation intent  → calls rec_engine directly, injects live results
  - Docs intent            → vector search over doc_chunk table (pgvector)
  - General intent         → Gemini answers from training knowledge only
  - Tourist context injected when tourist_id is provided
  - Multi-language: English, Sinhala, Tamil (classifier handles all three)
"""
import json
import logging
from typing import Optional, List

import google.generativeai as genai
from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.database import get_supabase
from app.schemas.payloads import ChatRequest, ChatResponse
from app.services import rec_engine
from app.services.vector_service import embed

log = logging.getLogger(__name__)
router = APIRouter()


# ── Gemini setup ──────────────────────────────────────────────────────────────

def _get_gemini():
    s = get_settings()
    if not s.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=s.gemini_api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


# ── Step 1: Intent classification ────────────────────────────────────────────
# One small Gemini call (~80–100 tokens) to classify the query type.
# Works in English, Sinhala, and Tamil automatically.
# Returns: {"type": "recommend"|"docs"|"general", "city": "Galle"|null}

CLASSIFIER_PROMPT = """You are a query classifier for Yaloo, a Sri Lanka tourism platform.
Classify the user message into exactly one category.
Also extract the city name if the user mentions one, otherwise return null.

Categories:
- "recommend" → user wants to find guides, stays/homestays, or activities
- "docs"       → user asks about Yaloo policies, booking process, fees, cancellation, safety, how yaloo works
- "general"    → general Sri Lanka travel questions, culture, weather, food, transport, tips

Reply ONLY with a valid JSON object. No explanation. No markdown. Just the JSON.
Format: {"type": "recommend"|"docs"|"general", "city": "city name or null"}

User message: "{message}"
"""

def _classify_intent(message: str) -> dict:
    """
    Classify message intent using a minimal Gemini call.
    Falls back to "general" if classification fails.
    ~80–100 tokens total.
    """
    try:
        model = _get_gemini()
        response = model.generate_content(
            CLASSIFIER_PROMPT.format(message=message),
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=30,
                temperature=0.0,
            )
        )
        raw = response.text.strip()
        # Strip any accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        # Validate shape
        if result.get("type") not in ("recommend", "docs", "general"):
            result["type"] = "general"
        return result
    except Exception as e:
        log.warning("Intent classification failed (%s), defaulting to general", e)
        return {"type": "general", "city": None}


# ── Step 2a: Recommendation context ──────────────────────────────────────────

def _fetch_recommendation_context(tourist_id: str, city: Optional[str]) -> str:
    """
    Call rec_engine directly and format top-3 results per category
    as a concise text block for Gemini's prompt.
    Adds ~250–350 tokens.
    """
    try:
        result = rec_engine.recommend(
            tourist_id=tourist_id,
            city=city,
            top_k=3,
        )
    except Exception as e:
        log.warning("rec_engine failed: %s", e)
        return ""

    lines = ["[LIVE YALOO DATA — use this to answer the user]\n"]

    if result.guides:
        lines.append("Guides available:")
        for g in result.guides:
            lines.append(
                f"  • {g.full_name} | City: {g.city_name or 'N/A'} | "
                f"Rating: {g.avg_rating or 'N/A'} | "
                f"Experience: {g.experience_years or '?'} yrs | "
                f"Rate: LKR {g.rate_per_hour or '?'}/hr | "
                f"Languages: {g.languages or 'N/A'}"
            )

    if result.stays:
        lines.append("\nStays available:")
        for s in result.stays:
            lines.append(
                f"  • {s.name} | City: {s.city_name or 'N/A'} | "
                f"Type: {s.type or 'N/A'} | "
                f"Budget: {s.budget or 'N/A'} | "
                f"Price: LKR {s.price_per_night or '?'}/night | "
                f"Rating: {s.avg_rating or 'N/A'}"
            )

    if result.activities:
        lines.append("\nActivities suited for this tourist:")
        for a in result.activities:
            lines.append(
                f"  • {a.name} | Category: {a.category or 'N/A'} | "
                f"Difficulty: {a.difficulty_level or 'N/A'} | "
                f"Price: LKR {a.base_price or '?'} | "
                f"Suits: {a.suitable_for or 'all'}"
            )

    lines.append(
        "\nInstructions: Use the above real data naturally in your reply. "
        "Don't list everything — highlight the best fits and explain why. "
        "Mention prices and ratings only if relevant to the user's question."
    )
    return "\n".join(lines)


# ── Step 2b: Vector doc search ────────────────────────────────────────────────

def _fetch_doc_context(message: str) -> str:
    """
    Embed the user's message and retrieve top-4 most relevant
    doc_chunk rows via pgvector cosine similarity.
    Adds ~300–500 tokens depending on chunk sizes.
    """
    try:
        query_vec = embed(message)
        db = get_supabase()
        result = db.rpc(
            "match_doc_chunks",
            {
                "query_embedding": query_vec,
                "category_filter": None,
                "match_count": 4,
            }
        ).execute()

        chunks = result.data or []
        if not chunks:
            return ""

        lines = ["[YALOO PLATFORM KNOWLEDGE — use this to answer the user]\n"]
        for chunk in chunks:
            lines.append(f"[{chunk['doc_name']}]\n{chunk['content']}\n")

        lines.append(
            "\nInstructions: Answer using the above platform knowledge. "
            "If the answer isn't in the above content, say so honestly."
        )
        return "\n".join(lines)

    except Exception as e:
        log.warning("Doc vector search failed: %s", e)
        return ""


# ── Step 3: Tourist profile context ──────────────────────────────────────────

def _tourist_context(tourist_id: str) -> str:
    db = get_supabase()
    tp = (
        db.table("tourist_profile")
        .select("travel_style, budget, active_level")
        .eq("id", tourist_id)
        .maybe_single()
        .execute()
    ).data
    if not tp:
        return ""
    return (
        f"Tourist profile: travel style = {tp.get('travel_style')}, "
        f"budget = {tp.get('budget')}, activity level = {tp.get('active_level')}."
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are Yaloo's friendly AI assistant. Yaloo (යාළු) means "Friend" in Sinhala.
Yaloo is a community tourism platform in Sri Lanka that connects tourists with local  volunteer as guides and family homestays, routing tourism revenue directly to ordinary Sri Lankans.

Your role:
- Help tourists plan their Sri Lanka trip using Yaloo's platform
- Answer questions about Yaloo's services, policies, and booking process
- Provide general Sri Lanka travel information (culture, etiquette, places, safety)
- Be warm, helpful, and reflect Yaloo's community-first values

{tourist_context}

Rules:
- Respond in the same language the user writes in 
- If you don't know something specific, say so honestly
- Always encourage booking through Yaloo to support local communities
- Do not make up prices, availability, or booking details not provided to you
- If no live data or docs are provided below, rely on your general knowledge
"""


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        model = _get_gemini()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    user_message = req.messages[-1].content if req.messages else ""

    # ── 1. Classify intent (~80–100 tokens) ──────────────────────────────────
    intent = _classify_intent(user_message)
    intent_type = intent.get("type", "general")
    city = intent.get("city")
    log.info("Intent: %s | City: %s", intent_type, city)

    # ── 2. Build conversation history ─────────────────────────────────────────
    history = []
    if len(req.messages) > 1:
        for msg in req.messages[:-1]:
            history.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [msg.content],
            })

    # ── 3. Route based on intent ──────────────────────────────────────────────
    try:
        chat_session = model.start_chat(history=history)

        if intent_type == "general":
            # ── Fast path: no system prompt, no context injection ─────────────
            # Gemini answers directly from its own training knowledge.
            # Saves ~300 tokens per general request.
            full_message = (
                f"You are Yaloo's friendly travel assistant for Sri Lanka. "
                f"Answer this travel question helpfully and concisely. "
                f"Respond in the same language as the question.\n\n"
                f"{user_message}"
            )

        else:
            # ── Context path: inject rec data or doc chunks ───────────────────
            tourist_ctx = ""
            injected_ctx = ""

            if req.tourist_id:
                tourist_ctx = _tourist_context(req.tourist_id)

            if intent_type == "recommend":
                if req.tourist_id:
                    injected_ctx = _fetch_recommendation_context(req.tourist_id, city)
                else:
                    injected_ctx = (
                        "[NOTE: User is not logged in. Encourage them to log in "
                        "for personalised recommendations. Describe what Yaloo offers generally.]"
                    )

            elif intent_type == "docs":
                injected_ctx = _fetch_doc_context(user_message)

            system_prompt = SYSTEM_TEMPLATE.format(tourist_context=tourist_ctx)
            if injected_ctx:
                system_prompt += f"\n\n{injected_ctx}"

            if not history:
                full_message = f"{system_prompt}\n\nUser: {user_message}"
            else:
                full_message = user_message

        response = chat_session.send_message(full_message)
        reply = response.text.strip()

    except Exception:
        log.exception("Gemini error")
        raise HTTPException(status_code=502, detail="Chat service temporarily unavailable")

    return ChatResponse(reply=reply, sources=[])