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
    return genai.GenerativeModel("gemini-2.5-flash")


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


        

def _get_intent_instructions(intent_type: str, city: Optional[str], context: str) -> str:
    if intent_type == "recommend":
        if not city:
            return "COMMAND: The user wants recommendations. You do not have a city yet. Ask the user which city/area they are visiting. Do not give generic advice yet."
        return f"COMMAND: Provide recommendations for {city} using the LIVE DATA provided below. Highlight top picks.\n\nDATA:\n{context}"
    
    if intent_type == "docs":
        return f"COMMAND: Answer the user's platform/policy question using ONLY the documentation below. Be concise.\n\nDOCS:\n{context}"
    
    # General intent
    return "COMMAND: Answer this general Sri Lanka travel question using your knowledge. Be helpful but brief."

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

SYSTEM_TEMPLATE = """You are Yaloo's AI assistant (යාළු - Friend). 
Role: Connect tourists with local guides and homestays in Sri Lanka.

{intent_instructions}

{tourist_context}

Rules:
1. Language: Always reply in the same language as the user.
2. Directness: Answer general or doc-related questions immediately. 
3. Recommendations: If the user wants a recommendation but the city is unknown, your ONLY task is to ask for the city.
"""


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_message = req.messages[-1].content if req.messages else ""
    
    # 1. Classify
    intent = _classify_intent(user_message)
    intent_type = intent.get("type", "general")
    city = intent.get("city")

    # 2. Fetch Context (Only if needed)
    injected_ctx = ""
    if intent_type == "recommend" and req.tourist_id and city:
        injected_ctx = _fetch_recommendation_context(req.tourist_id, city)
    elif intent_type == "docs":
        injected_ctx = _fetch_doc_context(user_message)

    # 3. Build the Per-Turn System Instruction
    tourist_ctx = _tourist_context(req.tourist_id) if req.tourist_id else ""
    intent_instructions = _get_intent_instructions(intent_type, city, injected_ctx)
    
    final_system_msg = SYSTEM_TEMPLATE.format(
        intent_instructions=intent_instructions,
        tourist_context=tourist_ctx
    )

    # 4. Start Session with PERSISTENT System Instructions
    try:
        # Re-initialize the model for this specific request to bake in the instructions
        s = get_settings()
        genai.configure(api_key=s.gemini_api_key)
        
        # KEY CHANGE: We pass the prompt as system_instruction
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=final_system_msg 
        )

        history = []
        if len(req.messages) > 1:
            for msg in req.messages[:-1]:
                history.append({
                    "role": "user" if msg.role == "user" else "model",
                    "parts": [msg.content],
                })

        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_message)
        
        return ChatResponse(reply=response.text.strip(), sources=[])

    except Exception as e:
        log.exception("Gemini error")
        raise HTTPException(status_code=502, detail="Service unavailable")