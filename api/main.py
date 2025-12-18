from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import re

app = FastAPI(title="SHL Assessment Recommendation API")

class QueryRequest(BaseModel):
    query: str

# Load dataset
df = pd.read_csv("shl_catalog.csv").fillna("")

def tokenize(text: str):
    return set(re.findall(r"\w+", text.lower()))

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/recommend")
def recommend(req: QueryRequest):
    query_tokens = tokenize(req.query)

    scored_rows = []
    for idx, row in df.iterrows():
        # Use entire row text → avoids KeyError completely
        row_text = " ".join(map(str, row.values))
        tokens = tokenize(row_text)
        score = len(query_tokens & tokens)
        scored_rows.append((score, idx))

    # Sort by relevance
    scored_rows.sort(reverse=True)
    top = scored_rows[:10]

    results = []
    for _, idx in top:
        row = df.iloc[idx]

        results.append({
            "name": str(row.get("Assessment Name", row.get("assessment_name", row.get("name", "")))),
            "url": str(row.get("URL", row.get("url", ""))),
            "adaptive_support": str(row.get("Adaptive Support", "No")),
            "description": str(row.get("Assessment Description", row.get("description", ""))),
            "duration": int(row.get("Duration", 60)),
            "remote_support": str(row.get("Remote Support", "Yes")),
            "test_type": [str(row.get("Test Type", row.get("test_type", "")))]
        })

    return {"recommended_assessments": results}
