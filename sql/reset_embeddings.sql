-- ============================================================
-- Yaloo AI — reset all embeddings
-- Run in Supabase SQL editor to wipe and rebuild from scratch.
--
-- USE CASES:
--   - Switching embedding model (must re-embed everything)
--   - Corrupted or mismatched vectors
--   - Starting fresh after schema changes
--
-- AFTER running this, run in your terminal:
--   python -m scripts.embed_all
-- ============================================================

-- ── Step 1: Drop all HNSW indexes ────────────────────────────
-- Must drop first — Postgres blocks efficient NULL updates
-- on indexed vector columns.

DROP INDEX IF EXISTS idx_guide_embedding;
DROP INDEX IF EXISTS idx_stay_embedding;
DROP INDEX IF EXISTS idx_activity_embedding;
DROP INDEX IF EXISTS idx_tourist_t2g;
DROP INDEX IF EXISTS idx_tourist_t2s;
DROP INDEX IF EXISTS idx_tourist_t2a;
DROP INDEX IF EXISTS idx_doc_chunk_embedding;

-- ── Step 2: Wipe all stored vectors ──────────────────────────

UPDATE guide_profile   SET embedding     = NULL;
UPDATE stay            SET embedding     = NULL;
UPDATE activity        SET embedding     = NULL;
UPDATE doc_chunk       SET embedding     = NULL;

-- Tourist has three separate columns — null all three atomically
UPDATE tourist_profile SET
    t2g_embedding = NULL,
    t2s_embedding = NULL,
    t2a_embedding = NULL;

-- ── Step 3: Confirm everything is cleared ─────────────────────
-- You should see 0 in every "embedded" column before proceeding.

SELECT 'guide_profile'   AS entity, COUNT(*) AS total, COUNT(embedding)     AS embedded FROM guide_profile
UNION ALL
SELECT 'stay',                       COUNT(*),          COUNT(embedding)               FROM stay
UNION ALL
SELECT 'activity',                   COUNT(*),          COUNT(embedding)               FROM activity
UNION ALL
SELECT 'tourist t2g',                COUNT(*),          COUNT(t2g_embedding)           FROM tourist_profile
UNION ALL
SELECT 'tourist t2s',                COUNT(*),          COUNT(t2s_embedding)           FROM tourist_profile
UNION ALL
SELECT 'tourist t2a',                COUNT(*),          COUNT(t2a_embedding)           FROM tourist_profile
UNION ALL
SELECT 'doc_chunk',                  COUNT(*),          COUNT(embedding)               FROM doc_chunk;

-- ── Step 4: Recreate HNSW indexes on now-empty columns ────────
-- Building on empty columns is faster than building after backfill.

CREATE INDEX idx_guide_embedding
    ON guide_profile
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_stay_embedding
    ON stay
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_activity_embedding
    ON activity
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Three indexes for three tourist query vectors
CREATE INDEX idx_tourist_t2g
    ON tourist_profile
    USING hnsw (t2g_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_tourist_t2s
    ON tourist_profile
    USING hnsw (t2s_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_tourist_t2a
    ON tourist_profile
    USING hnsw (t2a_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_doc_chunk_embedding
    ON doc_chunk
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- Step 3 should show 0 in every "embedded" column.
-- Step 4 recreates all 7 indexes on empty tables (fast).
-- Now run the backfill:
--
--   python -m scripts.embed_all
--
-- Or selectively:
--   python -m scripts.embed_all --guides --stays
--   python -m scripts.embed_all --tourists
--   python -m scripts.embed_all --docs
-- ============================================================
