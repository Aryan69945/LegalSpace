import os

from dotenv import load_dotenv
from google import genai
from app.config import RELEVANCE_THRESHOLD

from app.services.hybrid_search import hybrid_search
from app.services.cache import init_cache, get_cached, set_cached


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


client = genai.Client(api_key=api_key)

# Ensure cache table exists on startup
init_cache()


def build_context(results):

    context_parts = []
    filtered_sources = []

    for i, result in enumerate(results, 1):

        if result["rerank_score"] < RELEVANCE_THRESHOLD:
            continue

        filtered_sources.append(result)

        context_parts.append(
            f"""
SOURCE {i}
Article: {result["article"]}
Source: {result["source"]}

Text:
{result["text"]}
"""
        )

    if not context_parts:
        return None, []

    return "\n".join(context_parts), filtered_sources


# 4. Generate grounded answer

def answer_question(question):

    cached = get_cached(question)

    if cached:
        print(f"[CACHE HIT] {question[:60]}")
        return {
            "question": question,
            "answer":   cached["answer"],
            "sources":  cached["sources"],
            "cached":   True,
        }

    print(f"[CACHE MISS] {question[:60]}")

    results = hybrid_search(question, limit=5)


    context, filtered_sources = build_context(results)

    # --------------------------------------------------------
    # Stop if retrieval failed
    # --------------------------------------------------------

    if context is None:
        return {
            "question": question,
            "answer": (
                "Sorry, I don't have enough relevant context "
                "in the available legal documents to answer "
                "this question."
            ),
            "sources": [],
            "cached":  False,
        }

    # --------------------------------------------------------
    # Construct grounded prompt
    # --------------------------------------------------------

    prompt = f"""
You are LegalSpace, a legal-information assistant
specializing in the Constitution of India.

Answer the user's question using ONLY the legal
provisions provided in the SOURCES below.

Do not rely on information that is not present in
the supplied sources.

Rules:

1. Give a clear and concise answer.
2. Cite the relevant Article number whenever possible.
3. Explain the reasoning using the retrieved text.
4. Do not invent legal provisions.
5. If the supplied sources are insufficient to answer
   the question, explicitly say that the available
   sources are insufficient.
6. Distinguish the retrieved legal text from your
   explanation.

USER QUESTION:
{question}

SOURCES:
{context}

Now provide the answer.
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    answer = response.text

    # --------------------------------------------------------
    # Store in cache for future requests
    # --------------------------------------------------------

    set_cached(question, answer, filtered_sources)

    # --------------------------------------------------------
    # Return structured response
    # --------------------------------------------------------

    return {
        "question": question,
        "answer":   answer,
        "sources":  filtered_sources,
        "cached":   False,
    }


# ============================================================
# 5. Local test
# ============================================================

if __name__ == "__main__":

    question = "What protects equality before the law?"

    result = answer_question(question)

    print("\n")
    print("=" * 60)
    print("LEGALSPACE")
    print("=" * 60)

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:

        print(
            f"- {source['article']} "
            f"(rerank score: {source['rerank_score']:.4f})"
        )
