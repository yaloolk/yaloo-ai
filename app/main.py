"""
app/main.py
Root FastAPI application.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import recommend, chatbot
from app.services.vector_service import get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)

app = FastAPI(
    title="Yaloo AI",
    description="Recommendation engine and chatbot for the Yaloo community tourism platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock this down to your mobile app's domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, tags=["recommendations"])
app.include_router(chatbot.router, tags=["chat"])


@app.on_event("startup")
async def startup():
    """Pre-load the embedding model so the first request isn't slow."""
    get_embedding_model()


@app.get("/health")
async def health():
    return {"status": "ok"}

#uncomment below when u only need to tun embed_all file via huggingface 
#set a secret key in hugging face before starting [Key: RUN_EMBED_ON_START Value: true]

# import os
# from scripts.embed_all import run_embed_all 

# @app.on_event("startup")
# async def startup():
#     get_embedding_model()

#     if os.getenv("RUN_EMBED_ON_START") == "1":
#         logging.info("RUN_EMBED_ON_START=1 — running embed backfill ...")
#         import asyncio
#         loop = asyncio.get_event_loop()
#         await loop.run_in_executor(None, lambda: run_embed_all(only_nulls=True))
#         logging.info("Embed backfill finished.")