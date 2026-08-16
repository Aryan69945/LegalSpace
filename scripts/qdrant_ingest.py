import json

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from config import (
    COLLECTION_NAME,
    DENSE_VECTOR_SIZE,
    EMBEDDINGS_FILE,
    QDRANT_DIR,
)


with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} embeddings")


# Start local Qdrant
print("Starting Qdrant...")
qdrant = QdrantClient(path=str(QDRANT_DIR))

# Delete existing collection if it exists, then recreate fresh
existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME in existing:
    print(f"Deleting existing collection '{COLLECTION_NAME}'...")
    qdrant.delete_collection(COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "text-dense": VectorParams(
            size=DENSE_VECTOR_SIZE,
            distance=Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "text-sparse": SparseVectorParams()
    },
)

print("Collection created!")


# Insert documents
print("Inserting documents...")

points = []

for i, document in enumerate(documents):

    point = PointStruct(
        id=i,
        vector={
            "text-dense": document["dense"],
            "text-sparse": SparseVector(
                indices=document["sparse_indices"],
                values=document["sparse_values"],
            ),
        },
        payload={
            "article": document["article"],
            "text": document["text"],
            "source": document["source"],
        },
    )

    points.append(point)


qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)

print(f"✅ Inserted {len(points)} documents into Qdrant")


# Check collection
info = qdrant.get_collection(COLLECTION_NAME)

print("\nQdrant collection:")
print(info)