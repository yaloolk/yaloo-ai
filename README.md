# Yaloo AI

Recommendation engine + chatbot for the Yaloo community tourism platform.

## Project layout

```
yaloo_ai/
├── app/
│   ├── main.py               # FastAPI root
│   ├── api/
│   │   ├── recommend.py      # POST /recommend  +  webhook endpoints
│   │   └── chatbot.py        # POST /chat
│   ├── services/
│   │   ├── vector_service.py # embedding model + Supabase upsert
│   │   ├── rec_engine.py     # pgvector KNN + soft reranker
│   │   └── text_builder.py   # weighted field assembly
│   ├── core/
│   │   ├── config.py         # pydantic-settings
│   │   └── database.py       # Supabase client singleton
│   └── schemas/
│       └── payloads.py       # Pydantic request/response models
├── sql/
│   ├── 01_add_embeddings.sql  # pgvector migration — run first
│   └── 02_rpc_functions.sql   # KNN helper functions — run second
├── scripts/
│   └── embed_all.py           # one-shot backfill
├── docs/                      # Yaloo policy / FAQ text files for chatbot RAG
├── requirements.txt
└── .env.example
```

## Setup

1. **SQL migration** — run `sql/01_add_embeddings.sql` then `sql/02_rpc_functions.sql` in Supabase SQL editor.

2. **Environment**
   ```bash
   cp .env.example .env
   # edit .env with your Supabase URL, service key, Gemini key
   ```

3. **Install**
   ```bash
   pip install -r requirements.txt
   ```

4. **Backfill existing data**
   ```bash
   python -m scripts.embed_all
   ```

5. **Run**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Supabase Webhook setup

In Supabase Dashboard → Database → Webhooks, create:

| Table | Events | URL |
|---|---|---|
| `guide_profile` | INSERT, UPDATE | `https://YOUR_API/embed/guide` |
| `stay` | INSERT, UPDATE | `https://YOUR_API/embed/stay` |
| `activity` | INSERT, UPDATE | `https://YOUR_API/embed/activity` |
| `user_interest` | INSERT, DELETE | `https://YOUR_API/embed/tourist/invalidate` |
| `user_language` | INSERT, DELETE | `https://YOUR_API/embed/tourist/invalidate` |

Set header: `x-webhook-secret: YOUR_SECRET` (matches `SUPABASE_WEBHOOK_SECRET` in .env).

## Chatbot docs

Place `.txt` or `.md` files in `docs/` for the chatbot to use:
- `docs/platform_overview.md`
- `docs/booking_policy.md`
- `docs/guide_standards.md`
- `docs/homestay_standards.md`
- `docs/faq.md`
- `docs/sri_lanka_travel_tips.md`
