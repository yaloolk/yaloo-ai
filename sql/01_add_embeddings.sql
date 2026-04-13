-- ============================================================
-- Yaloo AI — pgvector migration
-- Run once against your Supabase project via SQL editor
-- ============================================================

-- 1. Enable pgvector extension (already available on Supabase)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Add embedding columns
--    all-mpnet-base-v2 produces 768-dimensional vectors

ALTER TABLE guide_profile
    ADD COLUMN IF NOT EXISTS embedding vector(768);

ALTER TABLE stay
    ADD COLUMN IF NOT EXISTS embedding vector(768);

ALTER TABLE activity
    ADD COLUMN IF NOT EXISTS embedding vector(768);

-- tourist_svector already exists in your schema (confirmed in CSV)
-- but it may be text/null — re-create as proper vector type if needed.
-- Safe approach: add a new column and migrate later if type is wrong.
ALTER TABLE tourist_profile
    ADD COLUMN IF NOT EXISTS svector vector(768);

-- 3. HNSW indexes for fast cosine-similarity KNN search
--    hnsw is faster at query time than ivfflat for small-medium datasets

CREATE INDEX IF NOT EXISTS idx_guide_embedding
    ON guide_profile
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_stay_embedding
    ON stay
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_activity_embedding
    ON activity
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_tourist_svector
    ON tourist_profile
    USING hnsw (svector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- Supabase DB Webhook setup (do this in Supabase Dashboard)
-- Dashboard → Database → Webhooks → Create new webhook
--
-- Guide re-embed webhook:
--   Table: guide_profile       Events: INSERT, UPDATE
--   URL:   https://YOUR_API/embed/guide
--   HTTP method: POST
--   Headers: x-webhook-secret: YOUR_SECRET
--
-- Stay re-embed webhook:
--   Table: stay                Events: INSERT, UPDATE
--   URL:   https://YOUR_API/embed/stay
--
-- Activity re-embed webhook:
--   Table: activity            Events: INSERT, UPDATE
--   URL:   https://YOUR_API/embed/activity
--
-- Tourist invalidation webhook:
--   Table: user_interest       Events: INSERT, DELETE
--   URL:   https://YOUR_API/embed/tourist/invalidate
--   (also add for user_language on INSERT/DELETE)
-- ============================================================
