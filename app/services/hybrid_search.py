import json
import os
from pathlib import Path

from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from app.config import (
    COLLECTION_NAME,
    RRF_CONSTANT
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


print("Loading legal documents...")

with open(
    _PROJECT_ROOT / "processed" / "constitution_articles.json",
    "r",
    encoding="utf-8",
) as f:
    documents = json.load(f)

texts = [document["text"] for document in documents]

print(f"Loaded {len(documents)} legal articles")


print("Building BM25 index...")

tokenized_documents = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_documents)

print("BM25 index ready")


print("Loading MiniLM...")

dense_model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Dense model ready")

print("Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker ready")


# 5. Connect to persistent Qdrant

qdrant = QdrantClient(
    path=str(_PROJECT_ROOT / "qdrant_data")
)

print("Qdrant connected")


# 6. Hybrid search


def hybrid_search(query, limit=5):

    query_tokens = query.lower().split()         #creates a list of every node of query.

    bm25_scores = bm25.get_scores(query_tokens)

    # Get top 20 BM25 candidates
    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:20]


    # --------------------------------------------------------
    # B. Dense retrieval
    # --------------------------------------------------------

    query_vector = list(
        dense_model.embed([query])
    )[0]

    dense_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),       # once again made it an array because initially it may be numpy array
        using="text-dense",
        limit=20,
    ).points


    # C. Reciprocal Rank Fusion (RRF)

    scores = {}

    k = RRF_CONSTANT


    # Add BM25 ranking contribution
    for rank, index in enumerate(bm25_indices):

        scores[index] = (
            scores.get(index, 0)
            + 1 / (k + rank + 1)
        )


    # Add dense ranking contribution
    for rank, result in enumerate(dense_results):

        index = result.id

        scores[index] = (
            scores.get(index, 0)
            + 1 / (k + rank + 1)
        )


    # D. Select candidates for reranking

    candidate_count = 20

    ranked_candidates = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:candidate_count]


    # E. Cross-Encoder reranking

    rerank_inputs = []

    for index, _ in ranked_candidates:

        rerank_inputs.append(
            (
                query,
                documents[index]["text"],
            )
        )


    rerank_scores = reranker.predict(
        rerank_inputs
    )


    # --------------------------------------------------------
    # F. Sort by reranker score
    # --------------------------------------------------------

    reranked = sorted(
        zip(
            ranked_candidates,
            rerank_scores,
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:limit]


    # --------------------------------------------------------
    # G. Build final results
    # --------------------------------------------------------

    results = []

    for ((index, rrf_score), rerank_score) in reranked:

        results.append(
            {
                "article": documents[index]["article"],
                "text": documents[index]["text"],
                "source": documents[index]["source"],
                "rrf_score": float(rrf_score),
                "rerank_score": float(rerank_score),
            }
        )


    return results

# 7. Test the search system

if __name__ == "__main__":

    query = "What protects equality before the law?"

    print("\n")
    print("=" * 60)
    print("HYBRID SEARCH + RERANKING")
    print("=" * 60)

    results = hybrid_search(query)


    for i, result in enumerate(results, 1):

        print("\n" + "-" * 60)

        print(f"Rank: {i}")

        print(
            f"Article: {result['article']}"
        )

        print(
            f"RRF Score: {result['rrf_score']}"
        )

        print(
            f"Rerank Score: {result['rerank_score']}"
        )

        print(
            f"Text:\n{result['text'][:500]}"
        )

        print(
            f"Source: {result['source']}"
        )