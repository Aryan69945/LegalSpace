from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ( 
    COLLECTION_NAME,
    DENSE_VECTOR_SIZE,
    GEMINI_MODEL,
    DENSE_EMBED_MODEL,
    SPARSE_EMBED_MODEL,
    RERANKER_MODEL,
    EMBED_BATCH_SIZE,
    MIN_ARTICLE_LENGTH,
)

PROJECT_ROOT    = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"        # drop PDFs here
PROCESSED_DIR   = PROJECT_ROOT / "processed"   # auto-generated JSONs
QDRANT_DIR      = PROJECT_ROOT / "qdrant_data" # local vector DB


# ── Processed file names ──────────────────────────────────────

ARTICLES_FILE   = PROCESSED_DIR / "all_articles.json"     # parsed chunks
EMBEDDINGS_FILE = PROCESSED_DIR / "all_embeddings.json"   # vectors (temp)

ARTICLE_PATTERN = r"(?m)^\s*(\d+[A-Z]?)\.\s+"

ARTICLE_LABEL   = "Article"
