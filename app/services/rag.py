import os

from dotenv import load_dotenv
from google import genai

from app.hybrid_search import hybrid_search


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# ============================================================
# 2. Gemini client
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# 3. Build context from retrieved documents
# ============================================================

def build_context(results):

    RELEVANCE_THRESHOLD = -2.0

    context_parts = []

    for i, result in enumerate(results, 1):

        if result["rerank_score"] < RELEVANCE_THRESHOLD:
            continue

        context_parts.append(
            f"""
SOURCE {i}
Article: {result["article"]}
Source: {result["source"]}

Text:
{result["text"]}
"""
        )

    # No sufficiently relevant evidence
    if not context_parts:
        return None

    return "\n".join(context_parts)


# ============================================================
# 4. Generate grounded answer
# ============================================================

def answer_question(question):

    # --------------------------------------------------------
    # Retrieve relevant legal provisions
    # --------------------------------------------------------

    results = hybrid_search(
        question,
        limit=5,
    )

    # --------------------------------------------------------
    # Build evidence context
    # --------------------------------------------------------

    context = build_context(results)

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

    # --------------------------------------------------------
    # Return structured response
    # --------------------------------------------------------

    return {
        "question": question,
        "answer": response.text,
        "sources": results,
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
