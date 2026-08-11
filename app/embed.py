import json
from fastembed import TextEmbedding, SparseTextEmbedding

# Load our processed legal articles
with open(
    "processed/constitution_articles.json",
    "r",
    encoding="utf-8"
) as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles")


# Dense embedding model
print("Loading dense model...")

dense_model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Sparse embedding model
print("Loading sparse model...")

sparse_model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1"
)

print("Models loaded!")


# Generate embeddings
texts = [article["text"] for article in articles]

print("Generating dense embeddings...")

dense_embeddings = list(
    dense_model.embed(texts)
)

print("Generating sparse embeddings...")

sparse_embeddings = list(
    sparse_model.embed(texts)
)

print("✅ Embeddings generated!")


# Check the first article
print("\nExample:")
print("Article:", articles[0]["article"])

print(
    "Dense vector dimensions:",
    len(dense_embeddings[0])
)

print(
    "Sparse vector dimensions:",
    len(sparse_embeddings[0].indices)
)