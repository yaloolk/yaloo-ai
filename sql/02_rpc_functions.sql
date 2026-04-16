-- ============================================================
-- Yaloo AI — typed RPC functions (final)
-- Run AFTER 01_add_embeddings.sql
-- exec_sql is intentionally absent — use typed functions only.
-- If you previously created exec_sql, drop it:
--   DROP FUNCTION IF EXISTS exec_sql(text);
-- ============================================================

-- ============================================================
-- match_guides
-- ============================================================
CREATE OR REPLACE FUNCTION match_guides(
    query_embedding  vector(768),
    city_filter      text     DEFAULT NULL,
    gender_filter    text     DEFAULT NULL,
    match_count      int      DEFAULT 15,
    available_ids    uuid[]   DEFAULT NULL
)
RETURNS TABLE (
    guide_profile_id uuid,
    user_profile_id  uuid,
    experience_years int,
    avg_rating       float,
    rate_per_hour    float,
    active_level     text,
    full_name        text,
    gender           text,
    profile_bio      text,
    city_name        text,
    vec_sim          float
)
LANGUAGE sql STABLE AS $$
    SELECT
        gp.id,
        gp.user_profile_id,
        gp.experience_years,
        gp.avg_rating,
        gp.rate_per_hour,
        gp.active_level,
        up.first_name || ' ' || up.last_name,
        up.gender::text,
        up.profile_bio,
        c.name,
        1 - (gp.embedding <=> query_embedding)
    FROM guide_profile gp
    JOIN user_profile  up ON up.id = gp.user_profile_id
    JOIN city           c ON c.id  = gp.city_id
    WHERE gp.embedding IS NOT NULL
      AND up.profile_status = 'active'
      AND (
            (available_ids IS NOT NULL AND gp.id = ANY(available_ids))
            OR
            (available_ids IS NULL AND gp.is_available = true)
      )
      AND (city_filter   IS NULL OR c.name ILIKE city_filter)
      AND (gender_filter IS NULL OR gender_filter = 'any' OR up.gender::text = gender_filter)
    ORDER BY gp.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- ============================================================
-- match_stays
-- ============================================================
CREATE OR REPLACE FUNCTION match_stays(
    query_embedding  vector(768),
    city_filter      text     DEFAULT NULL,
    match_count      int      DEFAULT 15,
    available_ids    uuid[]   DEFAULT NULL
)
RETURNS TABLE (
    stay_id          uuid,
    name             text,
    type             text,
    description      text,
    budget           text,
    price_per_night  float,
    city_name        text,
    avg_rating       float,
    vec_sim          float
)
LANGUAGE sql STABLE AS $$
    SELECT
        s.id,
        s.name,
        s.type,
        s.description,
        s.budget,
        s.price_per_night,
        c.name,
        hp.avg_rating,
        1 - (s.embedding <=> query_embedding)
    FROM stay s
    JOIN city c ON c.id = s.city_id
    LEFT JOIN host_profile hp ON hp.id = s.host_id
    WHERE s.embedding IS NOT NULL
      AND (
            (available_ids IS NOT NULL AND s.id = ANY(available_ids))
            OR
            (available_ids IS NULL AND s.is_active = true)
      )
      AND (city_filter IS NULL OR c.name ILIKE city_filter)
    ORDER BY s.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- ============================================================
-- match_activities
-- ============================================================
CREATE OR REPLACE FUNCTION match_activities(
    query_embedding vector(768),
    match_count     int DEFAULT 15
)
RETURNS TABLE (
    activity_id      uuid,
    name             text,
    category         text,
    description      text,
    budget           text,
    difficulty_level text,
    base_price       float,
    vec_sim          float
)
LANGUAGE sql STABLE AS $$
    SELECT
        a.id,
        a.name,
        a.category,
        a.description,
        a.budget,
        a.difficulty_level,
        a.base_price,
        1 - (a.embedding <=> query_embedding)
    FROM activity a
    WHERE a.embedding IS NOT NULL AND a.is_active = true
    ORDER BY a.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- ============================================================
-- match_doc_chunks
-- Used by chatbot RAG to find relevant Yaloo policy/FAQ chunks
-- ============================================================
CREATE OR REPLACE FUNCTION match_doc_chunks(
    query_embedding  vector(768),
    category_filter  text  DEFAULT NULL,
    match_count      int   DEFAULT 4
)
RETURNS TABLE (
    chunk_id    uuid,
    doc_name    text,
    category    text,
    content     text,
    vec_sim     float
)
LANGUAGE sql STABLE AS $$
    SELECT
        dc.id,
        dc.doc_name,
        dc.category,
        dc.content,
        1 - (dc.embedding <=> query_embedding)
    FROM doc_chunk dc
    WHERE dc.embedding IS NOT NULL
      AND (category_filter IS NULL OR dc.category = category_filter)
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
$$;
