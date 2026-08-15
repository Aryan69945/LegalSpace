from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.rag import answer_question
from app.agents.conflict_agent import (
    analyze_conflict,
    retrieve_legal_context,
)


app = FastAPI(
    title="LegalSpace API",
    description="Retrieval-grounded legal information system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ConflictRequest(BaseModel):
    bill_a: str
    bill_b: str


@app.get("/")
def health_check():

    return {
        "status": "ok",
        "service": "LegalSpace",
    }


@app.post("/api/chat")
def chat(request: ChatRequest):

    return answer_question(
        request.question
    )



@app.post("/api/conflict")
def conflict(request: ConflictRequest):

    legal_context = retrieve_legal_context(
        request.bill_a,
        request.bill_b,
    )

    result = analyze_conflict(
        request.bill_a,
        request.bill_b,
        legal_context,
    )

    return result
