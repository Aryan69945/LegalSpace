import json
from fastembed import TextEmbedding, SparseTextEmbedding

# Load structured articles
with open(
    "processed/constitution_articles.json",
    "r",
    encoding="utf-8"
) as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles")

# Load models
print("\nLoading dense embedding model...")
dense_model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading sparse embedding model...")
sparse_model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1"
)

print("✅ Models loaded!")

# Process in small batches
BATCH_SIZE = 16

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
with open(
    "processed/constitution_embeddings.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        all_embeddings,
        f,
        ensure_ascii=False
    )

print("Saved to processed/constitution_embeddings.json")