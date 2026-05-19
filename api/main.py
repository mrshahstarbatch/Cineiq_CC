from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

# Import our ML Engine components
from ml_engine.collaborative import CollaborativeModel
from ml_engine.content import ContentModel
from ml_engine.ensemble import HybridEnsemble
from ml_engine.data_loader import load_movie_metadata, load_links
from ml_engine.nlp_reranker import NLPReranker

app = FastAPI(title="CINEIQ API", description="Hybrid ML Movie Recommendation Engine")

# Define global variables for models
ensemble = None
nlp_reranker = None

@app.on_event("startup")
def load_models():
    """Loads all models and data into memory when the API boots up."""
    global ensemble, nlp_reranker
    print("Loading ML Models into memory...")
    
    collab_model = CollaborativeModel()
    collab_model.load_model("svd_model.pkl")
    
    content_model = ContentModel()
    content_model.load_model("content_model.pkl")
    
    movies_df = load_movie_metadata()
    links_df = load_links()
    
    ensemble = HybridEnsemble(collab_model, content_model, movies_df, links_df)
    nlp_reranker = NLPReranker()
    print("✅ CINEIQ Engine is ready to serve!")

class RecommendationRequest(BaseModel):
    user_id: int
    movie_title: str
    top_n: int = 5

class RecommendationResponse(BaseModel):
    movie: str
    explanation: str

@app.post("/recommend", response_model=List[RecommendationResponse])
def get_recommendations(req: RecommendationRequest):
    # 1. Get Hybrid Scores from the Ensemble
    base_recommendations = ensemble.recommend(req.user_id, req.movie_title, top_n=20)
    
    if not base_recommendations:
        raise HTTPException(status_code=404, detail="Movie not found or not enough data.")
        
    # Convert list of dicts to list of tuples for the NLP reranker
    hybrid_tuples = [(item["title"], item["collab_pred"]) for item in base_recommendations]
    
    # 2. Re-Rank with NLP (Using an empty dict for reviews since we aren't live-scraping IMDB right now)
    final_movies = nlp_reranker.rerank_recommendations(hybrid_tuples, movie_reviews_dict={})
    
    # 3. Generate Explainability Layer
    final_response = []
    
    # Look up the original predicted ratings to build the explanations
    pred_lookup = {item["title"]: item["collab_pred"] for item in base_recommendations}
    
    for title in final_movies[:req.top_n]:
        pred_rating = pred_lookup.get(title, 3.0)
        
        reason = f"Recommended based on your interest in '{req.movie_title}'. "
        if pred_rating >= 4.0:
            reason += f"Our collaborative filter strongly predicts a match (Score: {pred_rating:.1f}/5.0)."
        else:
            reason += f"It shares similar plot mechanics and themes (Score: {pred_rating:.1f}/5.0)."
            
        final_response.append({
            "movie": title,
            "explanation": reason
        })
        
    return final_response