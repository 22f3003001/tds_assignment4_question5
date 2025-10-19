from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
from typing import List
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use AI Pipe's OpenAI endpoint for embeddings
client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
async def similarity_search(request: SimilarityRequest):
    # Get embeddings for all docs and query
    all_texts = request.docs + [request.query]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=all_texts
    )
    
    # Extract embeddings
    embeddings = [np.array(item.embedding) for item in response.data]
    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]
    
    # Calculate similarities
    similarities = [
        (cosine_similarity(query_embedding, doc_emb), doc)
        for doc_emb, doc in zip(doc_embeddings, request.docs)
    ]
    
    # Sort by similarity (highest first) and get top 3
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_3 = [doc for _, doc in similarities[:3]]
    
    return {"matches": top_3}

@app.get("/")
async def root():
    return {"status": "running"}

