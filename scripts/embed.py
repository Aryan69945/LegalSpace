import json
from fastembed import TextEmbedding, SparseTextEmbedding
from config import (
    DENSE_EMBED_MODEL,
    SPARSE_EMBED_MODEL,
    EMBED_BATCH_SIZE,
    ARTICLES_FILE,
    EMBEDDINGS_FILE,
)

with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

print("\nLoading dense embedding model...")

dense_model = TextEmbedding(model_name=DENSE_EMBED_MODEL)

print("Loading sparse embedding model...")
sparse_model = SparseTextEmbedding(model_name=SPARSE_EMBED_MODEL)

print("✅ Models loaded!")

# Process in small batches
BATCH_SIZE = EMBED_BATCH_SIZE

all_embeddings = []

for start in range(0, len(articles), BATCH_SIZE):

    batch = articles[start:start + BATCH_SIZE]
    texts = [article["text"] for article in batch]

    print(
        f"\nEmbedding articles "
        f"{start + 1}-{min(start + BATCH_SIZE, len(articles))}"
    )

    # Dense embeddings
    dense_vectors = list(
        dense_model.embed(texts)
    )

    # Sparse embeddings
    sparse_vectors = list(
        sparse_model.embed(texts)
    )

    for article, dense, sparse in zip(
        batch,
        dense_vectors,
        sparse_vectors
    ):
        all_embeddings.append({
            "source": article["source"],
            "article": article["article"],
            "text": article["text"],
            "dense": dense.tolist(),
            "sparse_indices": sparse.indices.tolist(),
            "sparse_values": sparse.values.tolist()
        })

    print(
        f"Progress: "
        f"{min(start + BATCH_SIZE, len(articles))}/{len(articles)}"
    )

print("\n✅ All embeddings generated!")

# Save embeddings temporarily
with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_embeddings, f, ensure_ascii=False)

print(f"Saved to {EMBEDDINGS_FILE}")