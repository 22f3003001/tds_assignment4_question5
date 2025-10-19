from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
from typing import List
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = "https://aipipe.org/openrouter/v1"
API_KEY = os.getenv("OPENAI_API_KEY")

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embeddings(texts: List[str]):
    """Call OpenRouter embeddings endpoint and return list of embeddings."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }
    response = requests.post(f"{BASE_URL}/embeddings", json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    # OpenRouter returns embeddings under data[i].embedding
    return [np.array(item["embedding"]) for item in data["data"]]

@app.post("/similarity")
async def similarity_search(request: SimilarityRequest):
    all_texts = request.docs + [request.query]
    embeddings = get_embeddings(all_texts)

    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    similarities = [
        (cosine_similarity(query_embedding, doc_emb), doc)
        for doc_emb, doc in zip(doc_embeddings, request.docs)
    ]

    similarities.sort(reverse=True, key=lambda x: x[0])
    top_3 = [doc for _, doc in similarities[:3]]

    return {"matches": top_3}

@app.get("/")
async def root():
    return {"status": "running"}

