from fastembed import TextEmbedding
from qdrant_client import QdrantClient

COLLECTION_NAME = "indian_laws"

# Load the same embedding model
print("Loading embedding model...")

dense_model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# IMPORTANT:
# This is currently in-memory, so this only works if the
# Qdrant ingestion and search happen in the same process.
qdrant = QdrantClient(path="qdrant_data")

query = "Who is a citizen?"

print(f"\nSearching for: {query}")

# Convert question → vector
query_vector = list(
    dense_model.embed([query])
)[0]

# Search Qdrant
results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector.tolist(),
    using="text-dense",
    limit=5,
).points

for result in results:
    print("\n--------------------")
    print("Article:", result.payload["article"])
    print("Score:", result.score)
    print("Text:", result.payload["text"][:500])