-- ============================================================
-- Yaloo AI — exec_sql RPC helper
-- Run this in Supabase SQL editor AFTER the migration script.
--
-- This gives rec_engine.py a way to run parameterised KNN queries
-- that the Supabase PostgREST API can't express natively
-- (pgvector ORDER BY embedding <=> $vec isn't supported in the
-- PostgREST query builder).
--
-- Security: this function uses SECURITY DEFINER with a restricted
-- search_path.  Only call it from your service-role key (server-side).
-- Never expose it to the anon/authenticated roles via RLS.
-- ============================================================

-- ============================================================
-- exec_sql has been intentionally omitted.
--
-- A generic exec_sql(query text) function accepts arbitrary SQL,
-- giving any caller who holds the service-role key full database
-- access. Even with the service role restricted to server-side code,
-- a compromised backend would hand an attacker unrestricted query
-- power.
--
-- Instead, every query is expressed as a typed function below.
-- Each function accepts only the specific parameters it needs.
-- Postgres enforces the types — no string interpolation, no
-- injection surface, no surprise queries.
--
-- If you previously ran a version of this file that created exec_sql,
-- drop it now:
--   DROP FUNCTION IF EXISTS exec_sql(text);
-- ============================================================

-- ============================================================
-- match_guides
--
-- available_ids  uuid[]  DEFAULT NULL
--   NULL  → browse mode. Uses is_available flag on guide_profile.
--   '{}'  → empty array means no guides free — returns 0 rows fast.
--   '{id1,id2,...}' → only rank within this confirmed-available pool.
--           Django booking backend passes this after checking the
--           tourist's requested date/time slot.
--
-- "less matches" behaviour:
--   If only 2 guides are in available_ids, you get 2 back — not top_k.
--   FastAPI never pads with unavailable guides. Mobile app should
--   handle a shorter-than-expected list gracefully.
-- ============================================================

CREATE OR REPLACE FUNCTION match_guides(
    query_embedding  vector(768),
    city_filter      text     DEFAULT NULL,
    gender_filter    text     DEFAULT NULL,
    match_count      int      DEFAULT 15,
    available_ids    uuid[]   DEFAULT NULL   -- NULL = browse mode
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
LANGUAGE sql
STABLE
AS $$
    SELECT
        gp.id,
        gp.user_profile_id,
        gp.experience_years,
        gp.avg_rating,
        gp.rate_per_hour,
        gp.active_level,
        up.first_name || ' ' || up.last_name,
        up.gender,
        up.profile_bio,
        c.name,
        1 - (gp.embedding <=> query_embedding)
    FROM guide_profile gp
    JOIN user_profile  up ON up.id = gp.user_profile_id
    JOIN city           c ON c.id  = gp.city_id
    WHERE gp.embedding IS NOT NULL
      AND up.profile_status = 'active'
      -- Availability: use provided pool OR fall back to coarse flag
      AND (
            available_ids IS NOT NULL
            AND gp.id = ANY(available_ids)          -- time-slot search
          OR
            available_ids IS NULL
            AND gp.is_available = true              -- browse mode
      )
      AND (city_filter IS NULL OR c.name ILIKE city_filter)
      AND (gender_filter IS NULL OR gender_filter = 'any' OR up.gender::text = gender_filter)
    ORDER BY gp.embedding <=> query_embedding
    LIMIT match_count;
$$;


-- ============================================================
-- match_stays
-- Same available_ids logic as match_guides.
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
LANGUAGE sql
STABLE
AS $$
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
    LEFT JOIN host_profile hp ON hp.user_profile_id = s.host_id
    WHERE s.embedding IS NOT NULL
      AND (
            available_ids IS NOT NULL
            AND s.id = ANY(available_ids)
          OR
            available_ids IS NULL
            AND s.is_active = true
      )
      AND (city_filter IS NULL OR c.name ILIKE city_filter)
    ORDER BY s.embedding <=> query_embedding
    LIMIT match_count;
$$;


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
LANGUAGE sql
STABLE
AS $$
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
