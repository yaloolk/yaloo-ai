"""
app/api/chatbot.py

POST /chat

RAG chatbot:
  - Yaloo policies / documents (stored as text files in /docs)
  - Tourist's own recommendation context (injected if tourist_id provided)
  - Gemini 1.5 Flash as the LLM (fast, free tier available)
"""
import logging
import os
from pathlib import Path
from typing import List

import google.generativeai as genai
from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.database import get_supabase
from app.schemas.payloads import ChatRequest, ChatResponse

log = logging.getLogger(__name__)
router = APIRouter()

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"


# ── Gemini setup ──────────────────────────────────────────────────────────────

def _get_gemini():
    s = get_settings()
    if not s.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=s.gemini_api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


# ── Doc loader ────────────────────────────────────────────────────────────────

def _load_docs() -> str:
    """
    Load all .txt / .md files from the /docs folder.
    These should contain:
      - Yaloo platform overview
      - Booking policies
      - Guide & homestay standards
      - FAQ
      - SLTDA information
      - General Sri Lanka travel tips
    """
    if not DOCS_DIR.exists():
        return ""
    chunks: List[str] = []
    for f in sorted(DOCS_DIR.glob("**/*.{txt,md}")):
        try:
            text = f.read_text(encoding="utf-8").strip()
            if text:
                chunks.append(f"[{f.stem}]\n{text}")
        except Exception:
            pass
    return "\n\n---\n\n".join(chunks)


_DOCS_CACHE: str = ""   # loaded once at first call


def _get_docs() -> str:
    global _DOCS_CACHE
    if not _DOCS_CACHE:
        _DOCS_CACHE = _load_docs()
    return _DOCS_CACHE


# ── Tourist context ───────────────────────────────────────────────────────────

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
        f"This tourist's profile: travel style = {tp.get('travel_style')}, "
        f"budget = {tp.get('budget')}, activity level = {tp.get('active_level')}."
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are Yaloo's friendly AI assistant. Yaloo (යාළු) means "Friend" in Sinhala.
Yaloo is a community tourism platform in Sri Lanka that connects tourists with local student volunteer guides and family homestays, routing tourism revenue directly to ordinary Sri Lankans.

Your role:
- Help tourists plan their Sri Lanka trip using Yaloo's platform
- Answer questions about Yaloo's services, policies, and booking process
- Provide general Sri Lanka travel information (culture, etiquette, places, safety)
- Be warm, helpful, and reflect Yaloo's community-first values

{tourist_context}

Platform knowledge:
{docs}

Rules:
- If you don't know something specific, say so honestly and offer what you do know
- Always encourage booking through Yaloo to support local communities
- Do not make up prices, availability, or booking details — direct users to the app for live info
- Respond in the same language the user writes in (English, Sinhala, or Tamil)
"""


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        model = _get_gemini()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    tourist_ctx = ""
    if req.tourist_id:
        tourist_ctx = _tourist_context(req.tourist_id)

    system_prompt = SYSTEM_TEMPLATE.format(
        tourist_context=tourist_ctx,
        docs=_get_docs() or "(No additional platform documents loaded yet.)",
    )

    # Build Gemini conversation history
    history = []
    messages = req.messages
    if len(messages) > 1:
        for msg in messages[:-1]:
            history.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [msg.content],
            })

    user_message = messages[-1].content if messages else ""

    try:
        chat_session = model.start_chat(history=history)
        # Prepend system prompt to first user message (Gemini doesn't have a system role)
        if not history:
            full_message = f"{system_prompt}\n\nUser: {user_message}"
        else:
            full_message = user_message

        response = chat_session.send_message(full_message)
        reply = response.text.strip()
    except Exception as e:
        log.exception("Gemini error")
        raise HTTPException(status_code=502, detail="Chat service temporarily unavailable")

    return ChatResponse(reply=reply, sources=[])
