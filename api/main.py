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
        text = f"{row.to_string()}"
        tokens = tokenize(text)
        score = len(query_tokens & tokens)
        scores.append((score, idx))

    scores.sort(reverse=True)
    top = scores[:10]

    results = []
    for _, idx in top:
        row = df.iloc[idx]

        results.append({
            "name": str(row.get("Assessment Name", row.get("name", ""))),
            "url": str(row.get("URL", row.get("url", ""))),
            "adaptive_support": str(row.get("Adaptive Support", "No")),
            "description": str(row.get("Assessment Description", row.get("description", ""))),
            "duration": int(row.get("Duration", 60)),
            "remote_support": str(row.get("Remote Support", "Yes")),
            "test_type": [str(row.get("Test Type", row.get("test_type", "")))]
        })

    return {"recommended_assessments": results}

