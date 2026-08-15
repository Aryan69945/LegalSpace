RETRIEVAL_CANDIDATE_COUNT = 20

RETRIEVAL_TOP_K = 5

RELEVANCE_THRESHOLD = 1.0


# ── Models

GEMINI_MODEL        = "gemini-3.6-flash"
DENSE_EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_EMBED_MODEL  = "prithivida/Splade_PP_en_v1"
RERANKER_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Qdrant

COLLECTION_NAME     = "indian_laws"
DENSE_VECTOR_SIZE   = 384


# ── Ingestion

EMBED_BATCH_SIZE    = 16
MIN_ARTICLE_LENGTH  = 30  