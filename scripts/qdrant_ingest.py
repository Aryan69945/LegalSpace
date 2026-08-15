import json

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,              # to decide how similar two vectors are
    PointStruct,           # to represent one complete point/document in Qdrant
    SparseVector,          # to store a vector using only non-zero values
    SparseVectorParams,    # to configure sparse vectors in the collection
    VectorParams,          # to configure dense vectors in the collection
)


COLLECTION_NAME = "indian_laws"

with open(
    "processed/constitution_embeddings.json",
    "r",
    encoding="utf-8",
) as f:
    documents = json.load(f)


# Start local Qdrant
print("Starting Qdrant...")
qdrant = QdrantClient(path="qdrant_data")   # creates a local Qdrant database and connects to it


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