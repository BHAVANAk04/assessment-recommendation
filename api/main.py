import pandas as pd
import re
import math
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SHL Assessment Recommendation API")

class QueryRequest(BaseModel):
    query: str

# Load catalog
df = pd.read_csv("shl_catalog.csv")

def tokenize(text):
    if pd.isna(text):
        return []
    return re.findall(r"[a-zA-Z]+", text.lower())

# 🔴 FIX 1: build documents from assessment content, NOT query column
documents = []
for _, row in df.iterrows():
    combined_text = " ".join([
        str(row.get("Assessment_name", "")),
        str(row.get("Description", "")),
        str(row.get("Skills", "")),
    ])
    documents.append(tokenize(combined_text))

# Build vocabulary
vocab = sorted(set(word for doc in documents for word in doc))
word_index = {w: i for i, w in enumerate(vocab)}

# Compute IDF
N = len(documents)
idf = np.zeros(len(vocab))

for word, i in word_index.items():
    df_count = sum(word in doc for doc in documents)
    idf[i] = math.log((N + 1) / (df_count + 1)) + 1

# Compute TF-IDF matrix
tfidf_docs = []
for doc in documents:
    tf = np.zeros(len(vocab))
    for w in doc:
        tf[word_index[w]] += 1
    tfidf_docs.append(tf * idf)

tfidf_docs = np.array(tfidf_docs)

def vectorize_query(query):
    tf = np.zeros(len(vocab))
    for w in tokenize(query):
        if w in word_index:
            tf[word_index[w]] += 1
    return tf * idf

def cosine(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/recommend")
def recommend(req: QueryRequest):
    q_vec = vectorize_query(req.query)
    scores = [cosine(q_vec, doc) for doc in tfidf_docs]

    top_idx = np.argsort(scores)[::-1][:5]

    results = []
    for i in top_idx:
        results.append({
            "name": df.iloc[i].get("Assessment_name", "Unknown Assessment"),
            "url": df.iloc[i].get("Assessment_url", "")
        })

    return {"recommended_assessments": results}
