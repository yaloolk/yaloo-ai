-- ============================================================
-- Yaloo AI — Doc RAG migration
-- Run in Supabase SQL editor AFTER 01 and 02 scripts.
-- ============================================================

-- Table 1: source documents (you manage rows here manually or via admin)
CREATE TABLE IF NOT EXISTS doc_source (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name             text NOT NULL,          -- e.g. "cancellation_policy"
    url              text NOT NULL,          -- public URL to .txt or .pdf file
    category         text NOT NULL,          -- "policy" | "faq" | "travel_tips" | "guide_info"
    is_active        boolean DEFAULT true,
    last_embedded_at timestamptz,
    created_at       timestamptz DEFAULT now()
);

-- Table 2: chunks (auto-populated by embed_docs.py script)
CREATE TABLE IF NOT EXISTS doc_chunk (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   uuid NOT NULL REFERENCES doc_source(id) ON DELETE CASCADE,
    doc_name    text NOT NULL,
    category    text NOT NULL,
    chunk_idx   int  NOT NULL,
    content     text NOT NULL,
    embedding   vector(768)
);

-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_doc_chunk_embedding
    ON doc_chunk
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- RPC function: match_doc_chunks
-- Called by chatbot to find relevant doc chunks for a query
-- ============================================================
CREATE OR REPLACE FUNCTION match_doc_chunks(
    query_embedding vector(768),
    category_filter text DEFAULT NULL,
    match_count     int  DEFAULT 4
)
RETURNS TABLE (
    chunk_id    uuid,
    doc_name    text,
    category    text,
    content     text,
    similarity  float
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        dc.id,
        dc.doc_name,
        dc.category,
        dc.content,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM doc_chunk dc
    JOIN doc_source ds ON ds.id = dc.source_id
    WHERE dc.embedding IS NOT NULL
      AND ds.is_active = true
      AND (category_filter IS NULL OR dc.category = category_filter)
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
$$;
