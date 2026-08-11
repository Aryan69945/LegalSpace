import json

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


COLLECTION_NAME = "indian_laws"


# Load embeddings
print("Loading embeddings...")

with open(
    "processed/constitution_embeddings.json",
    "r",
    encoding="utf-8",
) as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} documents")


# Start local Qdrant
print("Starting Qdrant...")
qdrant = QdrantClient(path="qdrant_data")


# Create collection
print("Creating collection...")

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "text-dense": VectorParams(
            size=384,
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