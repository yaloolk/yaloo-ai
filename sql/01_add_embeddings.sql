-- ============================================================
-- Yaloo AI — pgvector migration (final)
-- Run once in Supabase SQL editor.
-- ============================================================

-- 1. Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 2. GUIDE — one embedding column
-- ============================================================
ALTER TABLE guide_profile
    ADD COLUMN IF NOT EXISTS embedding vector(768);

-- ============================================================
-- 3. STAY — one embedding column
-- ============================================================
ALTER TABLE stay
    ADD COLUMN IF NOT EXISTS embedding vector(768);

-- ============================================================
-- 4. ACTIVITY (global) — one embedding column
-- ============================================================
ALTER TABLE activity
    ADD COLUMN IF NOT EXISTS embedding vector(768);

-- ============================================================
-- 5. TOURIST — three separate embedding columns
--    tourist_svector was the old single column — drop it first.
--    Each column stores a different query vector variant:
--      t2g_embedding : tourist querying GUIDES  (travel_style + bridge strings for guide)
--      t2s_embedding : tourist querying STAYS   (travel_style + bridge strings for stay)
--      t2a_embedding : tourist querying ACTIVITIES (travel_style + bridge strings for activity)
-- ============================================================

-- Drop old single column if it exists
ALTER TABLE tourist_profile DROP COLUMN IF EXISTS tourist_svector;
ALTER TABLE tourist_profile DROP COLUMN IF EXISTS svector;

-- Add the three new columns
ALTER TABLE tourist_profile
    ADD COLUMN IF NOT EXISTS t2g_embedding vector(768);

ALTER TABLE tourist_profile
    ADD COLUMN IF NOT EXISTS t2s_embedding vector(768);

ALTER TABLE tourist_profile
    ADD COLUMN IF NOT EXISTS t2a_embedding vector(768);

-- ============================================================
-- 6. DOC CHUNKS — stores Yaloo policy/FAQ docs for chatbot RAG
--    doc_source  : metadata (name, category, url)
--    doc_chunk   : chunked text + embedding, FK to doc_source
-- ============================================================
CREATE TABLE IF NOT EXISTS doc_source (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL,
    category    text,             -- e.g. 'policy', 'faq', 'guide_standards'
    url         text,
    is_active   boolean DEFAULT true,
    created_at  timestamptz DEFAULT now(),
    updated_at  timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS doc_chunk (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   uuid REFERENCES doc_source(id) ON DELETE CASCADE,
    doc_name    text NOT NULL,    -- denormalised from doc_source.name for fast retrieval
    category    text,
    chunk_index int  NOT NULL,
    content     text NOT NULL,
    embedding   vector(768),
    created_at  timestamptz DEFAULT now()
);

-- ============================================================
-- 7. HNSW indexes — cosine similarity, fast KNN at query time
-- ============================================================
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

-- Tourist — one index per query-target column
CREATE INDEX IF NOT EXISTS idx_tourist_t2g
    ON tourist_profile
    USING hnsw (t2g_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_tourist_t2s
    ON tourist_profile
    USING hnsw (t2s_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_tourist_t2a
    ON tourist_profile
    USING hnsw (t2a_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_doc_chunk_embedding
    ON doc_chunk
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- 8. Verify — run this SELECT after migration to confirm
-- ============================================================
SELECT
    'guide_profile'   AS tbl, COUNT(*) AS rows, COUNT(embedding)     AS embedded FROM guide_profile
UNION ALL SELECT
    'stay',                    COUNT(*),          COUNT(embedding)               FROM stay
UNION ALL SELECT
    'activity',                COUNT(*),          COUNT(embedding)               FROM activity
UNION ALL SELECT
    'tourist t2g',             COUNT(*),          COUNT(t2g_embedding)           FROM tourist_profile
UNION ALL SELECT
    'tourist t2s',             COUNT(*),          COUNT(t2s_embedding)           FROM tourist_profile
UNION ALL SELECT
    'tourist t2a',             COUNT(*),          COUNT(t2a_embedding)           FROM tourist_profile
UNION ALL SELECT
    'doc_chunk',               COUNT(*),          COUNT(embedding)               FROM doc_chunk;
