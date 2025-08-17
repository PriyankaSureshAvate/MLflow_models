# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import scipy.sparse as sp

# -----------------------------
# Load trained model & mappings
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

user_map = mappings.get("user_map", {})
item_map = mappings.get("item_map", {})
inv_item_map = {v: k for k, v in item_map.items()}

# -----------------------------
# Build train matrix
# -----------------------------
df = pd.read_csv("data/interactions.csv")
df['user_idx'] = df['user_id'].map(user_map)
df['item_idx'] = df['item_id'].map(item_map)
train_matrix = sp.coo_matrix(
    (df['interaction'], (df['item_idx'], df['user_idx']))
).tocsr()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="ALS Recommendation API")

# -----------------------------
# Pydantic model
# -----------------------------
class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Recommendation API!"}

@app.get("/users/")
def get_all_users():
    """Return all valid user IDs."""
    return {"user_ids": [int(uid) for uid in user_map.keys()]}

@app.post("/recommend/")
def recommend(req: RecommendationRequest):
    """Return top-N recommendations for a specific user."""
    if req.user_id not in user_map:
        raise HTTPException(status_code=404, detail="User ID not found")
    
    user_idx = user_map[req.user_id]
    
    recommended_items, scores = model.recommend(
        user_idx,
        train_matrix,
        N=req.top_n,
        filter_already_liked_items=False
    )
    
    # Convert numpy arrays to native Python types
    recommended_item_ids = [int(inv_item_map[i]) for i in recommended_items.astype(int)]
    scores_list = scores.tolist()
    
    return {
        "user_id": req.user_id,
        "top_n": req.top_n,
        "recommended_items": recommended_item_ids,
        "scores": scores_list
    }

@app.get("/all_recommendations/")
def all_recommendations(top_n: int = 5):
    """Return top-N recommendations for all users."""
    all_recs = {}
    for user_id, user_idx in user_map.items():
        recommended_items, scores = model.recommend(
            user_idx,
            train_matrix,
            N=top_n,
            filter_already_liked_items=False
        )
        recommended_item_ids = [int(inv_item_map[i]) for i in recommended_items.astype(int)]
        scores_list = scores.tolist()
        
        all_recs[user_id] = {
            "recommended_items": recommended_item_ids,
            "scores": scores_list
        }

    return {"top_n": top_n, "recommendations": all_recs}