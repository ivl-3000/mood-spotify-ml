"""
Recommendation API for Hybrid Recommender System
FastAPI-based REST API for music recommendations.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from db.engine import get_session
from ml.recommender_system import HybridRecommender
from ml.context_aware_recommender import ContextAwareRecommender, ContextLearningRecommender

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Music Recommendation API",
    description="Hybrid music recommendation system with mood and context awareness",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
hybrid_model: Optional[HybridRecommender] = None
context_aware_model: Optional[ContextAwareRecommender] = None
learning_model: Optional[ContextLearningRecommender] = None


# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    user_tracks: List[str] = Field(default=[], description="User's track history")
    target_mood: Optional[str] = Field(None, description="Target mood for recommendations")
    context: Optional[Dict[str, Any]] = Field(None, description="Context information")
    n_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    use_context: bool = Field(default=True, description="Whether to use context awareness")
    use_learning: bool = Field(default=False, description="Whether to use learning from feedback")


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    context_used: Dict[str, Any]
    model_info: Dict[str, str]
    timestamp: str


class FeedbackRequest(BaseModel):
    user_id: str
    track_id: str
    context: Dict[str, Any]
    feedback: str = Field(..., pattern="^(positive|negative|skip)$")
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)


class TrackInfo(BaseModel):
    track_id: str
    name: str
    artist: str
    album: str
    duration_ms: int
    popularity: int
    audio_features: Dict[str, float]
    mood_info: Optional[Dict[str, Any]]


class ContextSuggestionResponse(BaseModel):
    suggested_activities: List[str]
    recommended_moods: List[str]
    optimal_context: Dict[str, Any]


# Dependency to get database session
def get_db_session():
    return get_session()


# Model loading functions
def load_models():
    """Load trained models."""
    global hybrid_model, context_aware_model, learning_model
    
    try:
        models_dir = Path("models/recommender")
        
        if (models_dir / "hybrid_model.pkl").exists():
            import pickle
            with open(models_dir / "hybrid_model.pkl", 'rb') as f:
                hybrid_model = pickle.load(f)
            
            context_aware_model = ContextAwareRecommender(hybrid_model)
            learning_model = ContextLearningRecommender(hybrid_model)
            
            logger.info("Models loaded successfully")
        else:
            logger.warning("No trained models found. Please train models first.")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": hybrid_model is not None
    }


# Main recommendation endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    db_session = Depends(get_db_session)
):
    """Get music recommendations."""
    try:
        if not hybrid_model:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Auto-detect context if not provided and context awareness is enabled
        if request.use_context and not request.context:
            if context_aware_model:
                request.context = context_aware_model._auto_detect_context()
        
        # Get recommendations
        if request.use_learning and learning_model:
            recommendations = learning_model.recommend_with_learning(
                user_id=request.user_id,
                user_tracks=request.user_tracks,
                context=request.context,
                target_mood=request.target_mood,
                n_recommendations=request.n_recommendations
            )
        elif request.use_context and context_aware_model:
            recommendations = context_aware_model.recommend_with_context(
                user_id=request.user_id,
                user_tracks=request.user_tracks,
                context=request.context,
                target_mood=request.target_mood,
                n_recommendations=request.n_recommendations
            )
        else:
            recommendations = hybrid_model.recommend(
                user_id=request.user_id,
                user_tracks=request.user_tracks,
                target_mood=request.target_mood,
                context=request.context,
                n_recommendations=request.n_recommendations
            )
        
        # Enrich recommendations with track information
        enriched_recommendations = []
        for track_id, score in recommendations:
            track_info = await get_track_info(track_id, db_session)
            if track_info:
                enriched_recommendations.append({
                    "track_id": track_id,
                    "score": float(score),
                    "track_info": track_info
                })
        
        return RecommendationResponse(
            recommendations=enriched_recommendations,
            context_used=request.context or {},
            model_info={
                "model_type": "learning" if request.use_learning else "context_aware" if request.use_context else "hybrid",
                "version": "1.0.0"
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Track information endpoint
@app.get("/track/{track_id}", response_model=TrackInfo)
async def get_track_details(track_id: str, db_session = Depends(get_db_session)):
    """Get detailed track information."""
    try:
        track_info = await get_track_info(track_id, db_session)
        if not track_info:
            raise HTTPException(status_code=404, detail="Track not found")
        
        return TrackInfo(**track_info)
        
    except Exception as e:
        logger.error(f"Error getting track details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Feedback endpoint
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for learning."""
    try:
        if not learning_model:
            raise HTTPException(status_code=503, detail="Learning model not available")
        
        learning_model.record_feedback(
            user_id=feedback.user_id,
            track_id=feedback.track_id,
            context=feedback.context,
            feedback=feedback.feedback,
            rating=feedback.rating
        )
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Context suggestions endpoint
@app.get("/context/suggestions", response_model=ContextSuggestionResponse)
async def get_context_suggestions(
    time_of_day: Optional[str] = Query(None, description="Time of day"),
    activity: Optional[str] = Query(None, description="Current activity"),
    season: Optional[str] = Query(None, description="Current season")
):
    """Get context-based suggestions."""
    try:
        if not context_aware_model:
            raise HTTPException(status_code=503, detail="Context-aware model not available")
        
        context = {}
        if time_of_day:
            context['time_of_day'] = time_of_day
        if activity:
            context['activity'] = activity
        if season:
            context['season'] = season
        
        if not context:
            context = context_aware_model._auto_detect_context()
        
        suggestions = context_aware_model.get_context_suggestions(context)
        
        return ContextSuggestionResponse(
            suggested_activities=suggestions.get('recommended_activities', []),
            recommended_moods=suggestions.get('mood_suggestions', []),
            optimal_context=context
        )
        
    except Exception as e:
        logger.error(f"Error getting context suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Similar tracks endpoint
@app.get("/similar/{track_id}")
async def get_similar_tracks(
    track_id: str,
    n_similar: int = Query(default=10, ge=1, le=20),
    db_session = Depends(get_db_session)
):
    """Get tracks similar to the given track."""
    try:
        if not hybrid_model:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Get similar tracks from content-based model
        similar_tracks = hybrid_model.content_model.recommend_similar(
            track_id, n_similar
        )
        
        # Enrich with track information
        enriched_tracks = []
        for sim_track_id, similarity in similar_tracks:
            track_info = await get_track_info(sim_track_id, db_session)
            if track_info:
                enriched_tracks.append({
                    "track_id": sim_track_id,
                    "similarity": float(similarity),
                    "track_info": track_info
                })
        
        return {"similar_tracks": enriched_tracks}
        
    except Exception as e:
        logger.error(f"Error getting similar tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mood-based recommendations endpoint
@app.get("/mood/{mood}")
async def get_mood_recommendations(
    mood: str,
    n_recommendations: int = Query(default=10, ge=1, le=20),
    db_session = Depends(get_db_session)
):
    """Get recommendations based on mood."""
    try:
        if not hybrid_model:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Get mood-based recommendations
        mood_recommendations = hybrid_model.mood_model.recommend_by_mood(
            mood, n_recommendations
        )
        
        # Enrich with track information
        enriched_recommendations = []
        for track_id, score in mood_recommendations:
            track_info = await get_track_info(track_id, db_session)
            if track_info:
                enriched_recommendations.append({
                    "track_id": track_id,
                    "mood_score": float(score),
                    "track_info": track_info
                })
        
        return {"mood_recommendations": enriched_recommendations}
        
    except Exception as e:
        logger.error(f"Error getting mood recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper function to get track information
async def get_track_info(track_id: str, db_session) -> Optional[Dict[str, Any]]:
    """Get comprehensive track information from database."""
    try:
        from sqlalchemy import text
        
        query = text("""
            SELECT 
                t.id, t.name, t.duration_ms, t.popularity,
                af.danceability, af.energy, af.valence, af.tempo, af.loudness,
                a.name as album_name,
                ar.name as artist_name,
                lnl.sentiment_score, lnl.dominant_emotion, lnl.emotion_probs
            FROM tracks t
            LEFT JOIN audio_features af ON t.id = af.track_id
            LEFT JOIN albums a ON t.album_id = a.id
            LEFT JOIN artists ar ON a.artist_id = ar.id
            LEFT JOIN lyrics_nlp lnl ON t.id = lnl.track_id
            WHERE t.id = :track_id
        """)
        
        result = db_session.execute(query, {"track_id": track_id}).fetchone()
        
        if not result:
            return None
        
        return {
            "track_id": result.id,
            "name": result.name,
            "artist": result.artist_name or "Unknown",
            "album": result.album_name or "Unknown",
            "duration_ms": result.duration_ms or 0,
            "popularity": result.popularity or 0,
            "audio_features": {
                "danceability": float(result.danceability or 0),
                "energy": float(result.energy or 0),
                "valence": float(result.valence or 0),
                "tempo": float(result.tempo or 0),
                "loudness": float(result.loudness or 0)
            },
            "mood_info": {
                "sentiment_score": float(result.sentiment_score or 0),
                "dominant_emotion": result.dominant_emotion,
                "emotion_probs": json.loads(result.emotion_probs) if result.emotion_probs else {}
            } if result.sentiment_score else None
        }
        
    except Exception as e:
        logger.error(f"Error getting track info for {track_id}: {e}")
        return None


# API documentation endpoint
@app.get("/docs")
async def get_api_docs():
    """Get API documentation."""
    return {
        "title": "Music Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /recommend": "Get music recommendations",
            "GET /track/{track_id}": "Get track details",
            "POST /feedback": "Submit user feedback",
            "GET /context/suggestions": "Get context suggestions",
            "GET /similar/{track_id}": "Get similar tracks",
            "GET /mood/{mood}": "Get mood-based recommendations",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "recommendation_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
