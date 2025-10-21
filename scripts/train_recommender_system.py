"""
Training script for Hybrid Recommender System
Implements Phase 4: Hybrid Recommender System training and evaluation.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from db.engine import get_session
from db.models import Track, AudioFeatures, PlaylistTrack, Lyrics, LyricsNLP
from ml.recommender_system import (
    HybridRecommender, 
    RecommenderEvaluator,
    CollaborativeFilteringRecommender,
    ContentBasedRecommender,
    MoodAwareRecommender
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommender_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_interaction_data(session: Session) -> pd.DataFrame:
    """Load playlist-track interactions for collaborative filtering."""
    logger.info("Loading interaction data...")
    
    query = text("""
        SELECT 
            pt.playlist_id,
            pt.track_id,
            pt.added_at,
            pt.position
        FROM playlist_tracks pt
        JOIN playlists p ON pt.playlist_id = p.id
        WHERE p.public = 1
    """)
    
    df = pd.read_sql(query, session.bind)
    logger.info(f"Loaded {len(df)} interactions from {df['playlist_id'].nunique()} playlists")
    return df


def load_track_data(session: Session) -> pd.DataFrame:
    """Load comprehensive track data for content-based filtering."""
    logger.info("Loading track data...")
    
    query = text("""
        SELECT 
            t.id as track_id,
            t.name as track_name,
            t.duration_ms,
            t.popularity,
            t.explicit,
            
            -- Audio features
            af.danceability,
            af.energy,
            af.valence,
            af.tempo,
            af.key,
            af.mode,
            af.loudness,
            
            -- Lyrics
            l.text as lyrics_text,
            l.language as lyrics_language,
            
            -- NLP features
            lnl.sentiment_score,
            lnl.dominant_emotion,
            lnl.emotion_probs,
            
            -- Album and artist info
            a.name as album_name,
            a.release_date,
            ar.name as artist_name
            
        FROM tracks t
        LEFT JOIN audio_features af ON t.id = af.track_id
        LEFT JOIN lyrics l ON t.id = l.track_id
        LEFT JOIN lyrics_nlp lnl ON t.id = lnl.track_id
        LEFT JOIN albums a ON t.album_id = a.id
        LEFT JOIN artists ar ON a.artist_id = ar.id
        WHERE t.id IS NOT NULL
    """)
    
    df = pd.read_sql(query, session.bind)
    logger.info(f"Loaded {len(df)} tracks with features")
    return df


def load_mood_data(session: Session) -> pd.DataFrame:
    """Load mood and sentiment data."""
    logger.info("Loading mood data...")
    
    query = text("""
        SELECT 
            t.id as track_id,
            lnl.sentiment_score,
            lnl.dominant_emotion,
            lnl.emotion_probs
        FROM tracks t
        JOIN lyrics_nlp lnl ON t.id = lnl.track_id
        WHERE lnl.sentiment_score IS NOT NULL
    """)
    
    df = pd.read_sql(query, session.bind)
    logger.info(f"Loaded {len(df)} tracks with mood data")
    return df


def create_test_data(interactions: pd.DataFrame, test_ratio: float = 0.2) -> pd.DataFrame:
    """Create test data by splitting user interactions."""
    logger.info("Creating test data...")
    
    test_data = []
    
    # For each playlist, create a test case
    for playlist_id in interactions['playlist_id'].unique():
        playlist_tracks = interactions[interactions['playlist_id'] == playlist_id]['track_id'].tolist()
        
        if len(playlist_tracks) < 3:  # Need at least 3 tracks for meaningful test
            continue
        
        # Split tracks into user history and relevant tracks
        np.random.seed(42)
        np.random.shuffle(playlist_tracks)
        
        split_point = max(1, int(len(playlist_tracks) * (1 - test_ratio)))
        user_tracks = playlist_tracks[:split_point]
        relevant_tracks = playlist_tracks[split_point:]
        
        if len(relevant_tracks) > 0:
            test_data.append({
                'user_id': playlist_id,
                'user_tracks': user_tracks,
                'relevant_tracks': relevant_tracks
            })
    
    test_df = pd.DataFrame(test_data)
    logger.info(f"Created {len(test_df)} test cases")
    return test_df


def train_individual_models(interactions: pd.DataFrame, 
                          track_data: pd.DataFrame, 
                          mood_data: pd.DataFrame) -> dict:
    """Train individual recommender models."""
    logger.info("Training individual models...")
    
    models = {}
    
    # Train Collaborative Filtering
    logger.info("Training Collaborative Filtering model...")
    cf_model = CollaborativeFilteringRecommender(factors=50, regularization=0.01, iterations=50)
    cf_model.fit(interactions)
    models['collaborative'] = cf_model
    
    # Train Content-Based
    logger.info("Training Content-Based model...")
    content_model = ContentBasedRecommender(max_features=1000, n_components=50)
    content_model.fit(track_data)
    models['content'] = content_model
    
    # Train Mood-Aware
    logger.info("Training Mood-Aware model...")
    mood_model = MoodAwareRecommender()
    mood_model.fit(mood_data)
    models['mood'] = mood_model
    
    return models


def train_hybrid_model(interactions: pd.DataFrame,
                      track_data: pd.DataFrame,
                      mood_data: pd.DataFrame) -> HybridRecommender:
    """Train the hybrid recommender system."""
    logger.info("Training hybrid recommender system...")
    
    # Initialize hybrid model with different weight configurations
    weight_configs = [
        {'cf_weight': 0.5, 'content_weight': 0.3, 'mood_weight': 0.2},
        {'cf_weight': 0.4, 'content_weight': 0.4, 'mood_weight': 0.2},
        {'cf_weight': 0.3, 'content_weight': 0.3, 'mood_weight': 0.4},
        {'cf_weight': 0.4, 'content_weight': 0.3, 'mood_weight': 0.3}
    ]
    
    best_model = None
    best_score = 0
    
    for config in weight_configs:
        logger.info(f"Testing weight configuration: {config}")
        
        model = HybridRecommender(**config)
        model.fit(interactions, track_data, mood_data)
        
        # Quick evaluation on a subset
        test_subset = create_test_data(interactions, test_ratio=0.1)
        if len(test_subset) > 0:
            # Simple evaluation - just check if we can generate recommendations
            try:
                sample_test = test_subset.iloc[0]
                recs = model.recommend(
                    user_id=sample_test['user_id'],
                    user_tracks=sample_test['user_tracks'],
                    n_recommendations=10
                )
                
                if len(recs) > 0:
                    score = len(recs)  # Simple scoring based on number of recommendations
                    if score > best_score:
                        best_score = score
                        best_model = model
                        logger.info(f"New best configuration with score: {score}")
            except Exception as e:
                logger.warning(f"Error testing configuration {config}: {e}")
                continue
    
    if best_model is None:
        logger.warning("No valid configuration found, using default weights")
        best_model = HybridRecommender()
        best_model.fit(interactions, track_data, mood_data)
    
    return best_model


def evaluate_models(models: dict, test_data: pd.DataFrame) -> dict:
    """Evaluate individual models and hybrid model."""
    logger.info("Evaluating models...")
    
    results = {}
    
    # Evaluate individual models
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name} model...")
        
        if model_name == 'collaborative':
            # Evaluate CF model
            cf_scores = []
            for _, row in test_data.iterrows():
                try:
                    recs = model.recommend_for_user(row['user_id'], 10)
                    cf_scores.append(len(recs))
                except:
                    cf_scores.append(0)
            results[model_name] = {'avg_recommendations': np.mean(cf_scores)}
        
        elif model_name == 'content':
            # Evaluate content model
            content_scores = []
            for _, row in test_data.iterrows():
                try:
                    recs = model.recommend_for_user_profile(row['user_tracks'], 10)
                    content_scores.append(len(recs))
                except:
                    content_scores.append(0)
            results[model_name] = {'avg_recommendations': np.mean(content_scores)}
        
        elif model_name == 'mood':
            # Evaluate mood model
            mood_scores = []
            for _, row in test_data.iterrows():
                try:
                    recs = model.recommend_mood_similar(row['user_tracks'][0] if row['user_tracks'] else '', 10)
                    mood_scores.append(len(recs))
                except:
                    mood_scores.append(0)
            results[model_name] = {'avg_recommendations': np.mean(mood_scores)}
    
    return results


def save_models(models: dict, hybrid_model: HybridRecommender, output_dir: Path):
    """Save trained models."""
    logger.info("Saving models...")
    
    output_dir.mkdir(exist_ok=True)
    
    # Save individual models
    for model_name, model in models.items():
        model_path = output_dir / f"{model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            import pickle
            pickle.dump(model, f)
        logger.info(f"Saved {model_name} model to {model_path}")
    
    # Save hybrid model
    hybrid_path = output_dir / "hybrid_model.pkl"
    with open(hybrid_path, 'wb') as f:
        import pickle
        pickle.dump(hybrid_model, f)
    logger.info(f"Saved hybrid model to {hybrid_path}")


def main():
    """Main training pipeline."""
    logger.info("Starting recommender system training...")
    
    # Get database session
    session = get_session()
    
    try:
        # Load data
        interactions = load_interaction_data(session)
        track_data = load_track_data(session)
        mood_data = load_mood_data(session)
        
        # Check data availability
        if len(interactions) == 0:
            logger.error("No interaction data available. Please collect playlist data first.")
            return
        
        if len(track_data) == 0:
            logger.error("No track data available. Please collect track data first.")
            return
        
        # Create test data
        test_data = create_test_data(interactions)
        
        # Train individual models
        individual_models = train_individual_models(interactions, track_data, mood_data)
        
        # Train hybrid model
        hybrid_model = train_hybrid_model(interactions, track_data, mood_data)
        
        # Evaluate models
        evaluation_results = evaluate_models(individual_models, test_data)
        
        # Print results
        logger.info("Evaluation Results:")
        for model_name, results in evaluation_results.items():
            logger.info(f"{model_name}: {results}")
        
        # Save models
        output_dir = Path("models/recommender")
        save_models(individual_models, hybrid_model, output_dir)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
