import os
import json

from dotenv import load_dotenv
from google import genai

from app.hybrid_search import hybrid_search


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


client = genai.Client(
    api_key=api_key
)


def retrieve_legal_context(bill_a, bill_b):

    query = f"""
Analyze the constitutional and legal implications of
the following two legislative provisions.

Bill A:
{bill_a}

Bill B:
{bill_b}

Identify relevant constitutional rights, provisions,
and potential conflicts between the two.
"""

    results = hybrid_search(
        query,
        limit=5,
    )

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
Article: {result["article"]}

Text:
{result["text"]}

Rerank score:
{result["rerank_score"]}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# 4. Analyze conflict
# ============================================================

def analyze_conflict(bill_a, bill_b, legal_context):

    prompt = f"""
You are the Cross-Bill Conflict Agent in LegalSpace.

Your job is to determine whether two legislative provisions
potentially conflict, using ONLY the supplied legal evidence.

BILL A:
{bill_a}

BILL B:
{bill_b}

RETRIEVED LEGAL CONTEXT:
{legal_context}


Analyze:

1. What does Bill A require or permit?

2. What does Bill B require or permit?

3. Determine whether they conflict.

4. If there is a conflict, classify the conflict as exactly
   one of the following:

   - "direct_conflict"
     The two provisions explicitly impose incompatible
     requirements.

   - "potential_conflict"
     The provisions may conflict depending on interpretation,
     implementation, or facts not provided.

   - "constitutional_conflict"
     The provision appears to conflict with a constitutional
     provision contained in the retrieved evidence.

   - "no_conflict"
     The provisions are compatible based on the supplied text.

   - "insufficient_evidence"
     The supplied evidence is not sufficient to determine
     whether a conflict exists.

5. Explain why you selected the conflict type.

6. Identify the retrieved constitutional/legal provisions
   relevant to the conflict.

7. Support the conclusion using the supplied evidence.

8. Do not invent facts, laws, or constitutional provisions.

9. If the evidence is insufficient, explicitly say so.


Return ONLY valid JSON in exactly this structure:

{{
    "conflict_detected": true,
    "conflict_type": "constitutional_conflict",
    "confidence": "high",
    "bill_a_effect": "...",
    "bill_b_effect": "...",
    "conflict_explanation": "...",
    "relevant_articles": [
        "Article 16"
    ],
    "supporting_evidence": [
        "..."
    ]
}}
"""

    # ========================================================
    # Call Gemini
    # ========================================================

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    text = response.text.strip()


    # ========================================================
    # Remove markdown JSON fences if Gemini adds them
    # ========================================================

    if text.startswith("```"):

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()


    # ========================================================
    # Parse JSON
    # ========================================================

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        return {
            "conflict_detected": None,
            "conflict_type": "insufficient_evidence",
            "confidence": "unknown",
            "bill_a_effect": "",
            "bill_b_effect": "",
            "conflict_explanation": text,
            "relevant_articles": [],
            "supporting_evidence": [],
        }


# ============================================================
# 5. Local test
# ============================================================

if __name__ == "__main__":

    bill_a = """
Every citizen shall have equal access to public
employment without discrimination based on caste.
"""

    bill_b = """
Public employment in certain categories shall be
restricted exclusively to members of a particular caste.
"""


    # ========================================================
    # Retrieve evidence
    # ========================================================

    print("\nRetrieving relevant constitutional provisions...")

    legal_context = retrieve_legal_context(
        bill_a,
        bill_b,
    )


    print("\nRetrieved Context:")
    print(legal_context)


    # ========================================================
    # Run conflict agent
    # ========================================================

    print("\nRunning Conflict Agent...")

    result = analyze_conflict(
        bill_a,
        bill_b,
        legal_context,
    )


    # ========================================================
    # Display result
    # ========================================================

    print("\n")
    print("=" * 60)
    print("LEGALSPACE — CROSS-BILL CONFLICT ANALYSIS")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=4,
        )
    )
