from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="SHL Assessment Recommendation API")

class QueryRequest(BaseModel):
    query: str

# Load catalog
df = pd.read_csv("shl_catalog.csv")

# Prepare text corpus
df["combined"] = (
    df["name"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["test_type"].fillna("")
)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined"].tolist())

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/recommend")
def recommend(req: QueryRequest):
    query_vec = vectorizer.transform([req.query])
    scores = cosine_similarity(query_vec, X)[0]

    top_indices = scores.argsort()[::-1][:10]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "name": row.get("name", ""),
            "url": row.get("url", ""),
            "adaptive_support": row.get("adaptive_support", "No"),
            "description": row.get("description", ""),
            "duration": int(row.get("duration", 60)),
            "remote_support": row.get("remote_support", "Yes"),
            "test_type": [row.get("test_type", "")]
        })

    return {"recommended_assessments": results}
