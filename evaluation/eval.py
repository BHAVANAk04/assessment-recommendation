
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Expected CSVs:
# catalog.csv with columns: name, url, description
# train.csv with columns: query, relevant_urls (comma-separated)

catalog = pd.read_csv("shl_catalog.csv")
train = pd.read_csv("train.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")
cat_emb = model.encode((catalog["name"].astype(str)+" "+catalog["description"].astype(str)).tolist(), normalize_embeddings=True)

def recall_at_k(relevant, ranked, k):
    return len(set(relevant) & set(ranked[:k])) / max(1,len(relevant))

scores = []
for _, row in train.iterrows():
    qemb = model.encode([row["query"]], normalize_embeddings=True)
    sims = cosine_similarity(qemb, cat_emb)[0]
    order = sims.argsort()[::-1]
    ranked_urls = catalog.iloc[order]["url"].tolist()
    relevant = [u.strip() for u in str(row["relevant_urls"]).split(",")]
    scores.append(recall_at_k(relevant, ranked_urls, 10))

print("Mean Recall@10:", sum(scores)/len(scores))
