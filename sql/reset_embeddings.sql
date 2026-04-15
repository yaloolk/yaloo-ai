-- ============================================================
-- Yaloo AI — reset all embeddings
-- Run this in Supabase SQL editor when you want to
-- wipe everything and re-embed from scratch.
--
-- USE CASES:
--   - Switching to a different embedding model
--   - Suspecting corrupted or mismatched vectors
--   - Starting fresh after schema changes
--
-- AFTER running this, run:
--   python -m scripts.embed_all
-- to re-populate all embeddings.
-- ============================================================

-- Step 1: Drop all HNSW indexes
-- (must drop before clearing columns — Postgres won't let you
-- truncate a column that an index is built on efficiently)
DROP INDEX IF EXISTS idx_guide_embedding;
DROP INDEX IF EXISTS idx_stay_embedding;
DROP INDEX IF EXISTS idx_activity_embedding;
DROP INDEX IF EXISTS idx_tourist_svector;

-- Step 2: Wipe all stored vectors
UPDATE guide_profile   SET embedding        = NULL;
UPDATE stay            SET embedding        = NULL;
UPDATE activity        SET embedding        = NULL;
UPDATE tourist_profile SET tourist_svector  = NULL;

-- Step 3: Confirm everything is cleared
SELECT
    'guide_profile'   AS table_name,
    COUNT(*)          AS total_rows,
    COUNT(embedding)  AS rows_with_embedding
FROM guide_profile
UNION ALL
SELECT
    'stay',
    COUNT(*),
    COUNT(embedding)
FROM stay
UNION ALL
SELECT
    'activity',
    COUNT(*),
    COUNT(embedding)
FROM activity
UNION ALL
SELECT
    'tourist_profile',
    COUNT(*),
    COUNT(tourist_svector)
FROM tourist_profile;

-- Step 4: Recreate HNSW indexes on empty columns
-- (faster to build on empty table than after backfill)
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

CREATE INDEX idx_tourist_svector
    ON tourist_profile
    USING hnsw (tourist_svector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- Now run embed_all.py from your terminal:
--   python -m scripts.embed_all
-- ============================================================
