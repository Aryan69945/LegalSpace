import hashlib
import json
import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()

_DATABASE_URL = os.getenv("DATABASE_URL")

if not _DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")


def _get_conn():
    return psycopg2.connect(_DATABASE_URL)


# ── Table setup

def init_cache():
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash  CHAR(64)    PRIMARY KEY,
            question    TEXT        NOT NULL,
            answer      TEXT        NOT NULL,
            sources     JSONB       NOT NULL DEFAULT '[]',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Cache table ready")


# ── Hashing

def make_hash(text: str) -> str:
    """
    Deterministic SHA-256 hash of a query.
    Normalised (stripped + lowercased) so that:
      "What is Article 14?" == "what is article 14?" == "  What is Article 14?  "
    """
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ── Cache read

def get_cached(question: str) -> dict | None:
    """
    Return cached result for a question, or None on miss.

    Returns:
        {
            "answer":  str,
            "sources": list,
            "cached":  True
        }
    """
    query_hash = make_hash(question)

    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute(
        "SELECT answer, sources FROM query_cache WHERE query_hash = %s",
        (query_hash,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        return {
            "answer":  row[0],
            "sources": row[1],   # psycopg2 auto-deserialises JSONB to list
            "cached":  True,
        }

    return None


# ── Cache write

def set_cached(question: str, answer: str, sources: list) -> None:
    """
    Store a Gemini response in the cache.
    ON CONFLICT DO NOTHING — safe to call even if another
    request cached the same query simultaneously.
    """
    query_hash = make_hash(question)

    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute(
        """
        INSERT INTO query_cache (query_hash, question, answer, sources)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (query_hash) DO NOTHING
        """,
        (query_hash, question, answer, json.dumps(sources)),
    )
    conn.commit()
    cur.close()
    conn.close()
