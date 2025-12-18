from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI(title="SHL Assessment Recommendation API")

# ---------- Request / Response Schemas ----------

class QueryRequest(BaseModel):
    query: str

class Recommendation(BaseModel):
    name: str
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Recommendation]

# ---------- Load Dataset ----------

df = pd.read_csv("shl_catalog.csv")

# Rename columns to standard names
df = df.rename(columns={
    "Assessment_url": "url",
    "Query": "query_text"
})

# Derive NAME from URL (IMPORTANT)
df["name"] = df["url"].apply(
    lambda x: x.split("/")[-2].replace("-", " ").title()
)

# Use query text as description proxy
df["description"] = df["query_text"]

# Simple heuristic for test type
def infer_test_type(text):
    text = text.lower()
    if "coding" in text or "python" in text or "java" in text:
        return ["K"]
    return ["P"]

df["test_type"] = df["query_text"].apply(infer_test_type)

# ---------- Embeddings (RAG) ----------

model = SentenceTransformer("all-MiniLM-L6-v2")
corpus = (df["name"] + " " + df["description"]).tolist()
embeddings = model.encode(corpus, normalize_embeddings=True)

# ---------- API ----------

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: QueryRequest):
    query_embedding = model.encode(
        [req.query], normalize_embeddings=True
    )

    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = scores.argsort()[::-1][:10]

    results = []
    for i in top_indices:
        results.append({
            "name": df.iloc[i]["name"],
            "url": df.iloc[i]["url"],
            "adaptive_support": "No",
            "description": df.iloc[i]["description"],
            "duration": 60,
            "remote_support": "Yes",
            "test_type": df.iloc[i]["test_type"]
        })

    return {"recommended_assessments": results}
