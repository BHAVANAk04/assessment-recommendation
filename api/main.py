from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import re

app = FastAPI(title="SHL Assessment Recommendation API")

class QueryRequest(BaseModel):
    query: str

# Load catalog
df = pd.read_csv("shl_catalog.csv").fillna("")

def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/recommend")
def recommend(req: QueryRequest):
    query_tokens = tokenize(req.query)

    scores = []
    for idx, row in df.iterrows():
        text = f"{row['name']} {row['description']} {row['test_type']}"
        tokens = tokenize(text)
        score = len(query_tokens & tokens)
        scores.append((score, idx))

    scores.sort(reverse=True)
    top = scores[:10]

    results = []
    for _, idx in top:
        row = df.iloc[idx]
        results.append({
            "name": row["name"],
            "url": row["url"],
            "adaptive_support": row.get("adaptive_support", "No"),
            "description": row["description"],
            "duration": int(row.get("duration", 60)),
            "remote_support": row.get("remote_support", "Yes"),
            "test_type": [row["test_type"]]
        })

    return {"recommended_assessments": results}
