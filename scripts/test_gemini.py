import os

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")


# 2. Initialize Gemini
client = genai.Client(api_key=GEMINI_API_KEY)


# 3. Initialize Qdrant
print("Initializing Qdrant Database...")

qdrant = QdrantClient(":memory:")


# 4. Initialize dense embedding model
print("Loading dense embedding model...")

dense_model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 5. Initialize sparse embedding model
print("Loading sparse embedding model...")

sparse_model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1"
)

print("✅ Qdrant and embedding models are ready!")


# 6. Test Gemini
print("\nTesting Gemini...")

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain the concept of 'Habeas Corpus' in one short sentence."
)

print(f"🤖 Gemini says: {response.text}")