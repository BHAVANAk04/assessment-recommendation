from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load data once (important for memory)
df = pd.read_csv("shl_catalog.csv")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Query"])

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/recommend")
def recommend(req: QueryRequest):
    user_vec = vectorizer.transform([req.query])
    similarities = cosine_similarity(user_vec, tfidf_matrix)[0]

    top_indices = similarities.argsort()[-5:][::-1]

    results = []
    for idx in top_indices:
        url = df.iloc[idx]["Assessment_url"]

        results.append({
            "name": url.split("/")[-2].replace("-", " ").title(),
            "url": url
        })

    return {"recommended_assessments": results}
