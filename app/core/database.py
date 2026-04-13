"""
app/core/database.py
Supabase client pool — one instance shared across the app.
Uses the service-role key so it can read/write any row.
"""
from functools import lru_cache
from supabase import create_client, Client
from app.core.config import get_settings


@lru_cache
def get_supabase() -> Client:
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_key)
