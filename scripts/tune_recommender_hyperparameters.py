"""
Hyperparameter tuning for Hybrid Recommender System
Implements grid search for optimal hyperparameters.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
from sqlalchemy.orm import Session

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from db.engine import get_session
from ml.recommender_system import HybridRecommender, RecommenderEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparameter_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(session: Session) -> tuple:
    """Load all required data."""
    logger.info("Loading data for hyperparameter tuning...")
    
    # Load interactions
    interactions_query = """
        SELECT playlist_id, track_id, added_at, position
        FROM playlist_tracks pt
        JOIN playlists p ON pt.playlist_id = p.id
        WHERE p.public = 1
    """
    interactions = pd.read_sql(interactions_query, session.bind)
    
    # Load track data
    track_query = """
        SELECT 
            t.id as track_id, t.name, t.duration_ms, t.popularity,
            af.danceability, af.energy, af.valence, af.tempo, af.loudness,
            l.text as lyrics_text,
            lnl.sentiment_score, lnl.dominant_emotion, lnl.emotion_probs
        FROM tracks t
        LEFT JOIN audio_features af ON t.id = af.track_id
        LEFT JOIN lyrics l ON t.id = l.track_id
        LEFT JOIN lyrics_nlp lnl ON t.id = lnl.track_id
    """
    track_data = pd.read_sql(track_query, session.bind)
    
    # Load mood data
    mood_query = """
        SELECT t.id as track_id, lnl.sentiment_score, lnl.dominant_emotion, lnl.emotion_probs
        FROM tracks t
        JOIN lyrics_nlp lnl ON t.id = lnl.track_id
        WHERE lnl.sentiment_score IS NOT NULL
    """
    mood_data = pd.read_sql(mood_query, session.bind)
    
    logger.info(f"Loaded {len(interactions)} interactions, {len(track_data)} tracks, {len(mood_data)} mood records")
    return interactions, track_data, mood_data


def create_cross_validation_splits(interactions: pd.DataFrame, n_splits: int = 3) -> list:
    """Create cross-validation splits."""
    logger.info(f"Creating {n_splits} cross-validation splits...")
    
    splits = []
    playlist_ids = interactions['playlist_id'].unique()
    np.random.seed(42)
    np.random.shuffle(playlist_ids)
    
    split_size = len(playlist_ids) // n_splits
    
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < n_splits - 1 else len(playlist_ids)
        
        test_playlists = playlist_ids[start_idx:end_idx]
        train_playlists = np.concatenate([playlist_ids[:start_idx], playlist_ids[end_idx:]])
        
        train_interactions = interactions[interactions['playlist_id'].isin(train_playlists)]
        test_interactions = interactions[interactions['playlist_id'].isin(test_playlists)]
        
        splits.append((train_interactions, test_interactions))
    
    return splits


def evaluate_configuration(interactions: pd.DataFrame, 
                          track_data: pd.DataFrame, 
                          mood_data: pd.DataFrame,
                          config: dict) -> float:
    """Evaluate a specific hyperparameter configuration."""
    try:
        # Create model with configuration
        model = HybridRecommender(**config)
        model.fit(interactions, track_data, mood_data)
        
        # Create test data
        test_cases = []
        for playlist_id in interactions['playlist_id'].unique()[:5]:  # Limit for speed
            playlist_tracks = interactions[interactions['playlist_id'] == playlist_id]['track_id'].tolist()
            if len(playlist_tracks) >= 3:
                np.random.shuffle(playlist_tracks)
                user_tracks = playlist_tracks[:len(playlist_tracks)//2]
                relevant_tracks = playlist_tracks[len(playlist_tracks)//2:]
                
                if len(relevant_tracks) > 0:
                    test_cases.append({
                        'user_id': playlist_id,
                        'user_tracks': user_tracks,
                        'relevant_tracks': relevant_tracks
                    })
        
        if len(test_cases) == 0:
            return 0.0
        
        # Evaluate on test cases
        scores = []
        for test_case in test_cases:
            try:
                recommendations = model.recommend(
                    user_id=test_case['user_id'],
                    user_tracks=test_case['user_tracks'],
                    n_recommendations=10
                )
                
                if len(recommendations) > 0:
                    # Calculate simple metrics
                    recommended_tracks = [track_id for track_id, _ in recommendations]
                    precision = RecommenderEvaluator.precision_at_k(
                        recommended_tracks, 
                        test_case['relevant_tracks'], 
                        10
                    )
                    recall = RecommenderEvaluator.recall_at_k(
                        recommended_tracks, 
                        test_case['relevant_tracks'], 
                        10
                    )
                    
                    # Combined score (F1-like)
                    if precision + recall > 0:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                        scores.append(f1_score)
                    else:
                        scores.append(0.0)
                else:
                    scores.append(0.0)
            except Exception as e:
                logger.warning(f"Error evaluating test case: {e}")
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
        
    except Exception as e:
        logger.warning(f"Error evaluating configuration {config}: {e}")
        return 0.0


def grid_search_hyperparameters(interactions: pd.DataFrame,
                               track_data: pd.DataFrame,
                               mood_data: pd.DataFrame) -> dict:
    """Perform grid search for optimal hyperparameters."""
    logger.info("Starting grid search for hyperparameters...")
    
    # Define parameter grids
    param_grids = {
        'cf_weight': [0.2, 0.3, 0.4, 0.5, 0.6],
        'content_weight': [0.2, 0.3, 0.4, 0.5, 0.6],
        'mood_weight': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    # Filter combinations where weights sum to 1.0 (within tolerance)
    valid_configs = []
    for cf_w, content_w, mood_w in product(
        param_grids['cf_weight'],
        param_grids['content_weight'],
        param_grids['mood_weight']
    ):
        if abs(cf_w + content_w + mood_w - 1.0) < 0.01:  # Allow small tolerance
            valid_configs.append({
                'cf_weight': cf_w,
                'content_weight': content_w,
                'mood_weight': mood_w
            })
    
    logger.info(f"Testing {len(valid_configs)} valid configurations...")
    
    best_score = 0.0
    best_config = None
    results = []
    
    for i, config in enumerate(valid_configs):
        logger.info(f"Evaluating configuration {i+1}/{len(valid_configs)}: {config}")
        
        score = evaluate_configuration(interactions, track_data, mood_data, config)
        results.append((config, score))
        
        logger.info(f"Configuration {config} scored: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_config = config
            logger.info(f"New best configuration with score: {score:.4f}")
    
    # Sort results by score
    results.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("Top 5 configurations:")
    for i, (config, score) in enumerate(results[:5]):
        logger.info(f"{i+1}. {config}: {score:.4f}")
    
    return {
        'best_config': best_config,
        'best_score': best_score,
        'all_results': results
    }


def tune_collaborative_filtering_params(interactions: pd.DataFrame) -> dict:
    """Tune collaborative filtering specific parameters."""
    logger.info("Tuning collaborative filtering parameters...")
    
    # Define parameter grid for CF
    cf_params = {
        'factors': [20, 50, 100],
        'regularization': [0.01, 0.1, 0.5],
        'iterations': [20, 50, 100]
    }
    
    best_score = 0.0
    best_params = None
    
    for factors, reg, iters in product(
        cf_params['factors'],
        cf_params['regularization'],
        cf_params['iterations']
    ):
        try:
            from ml.recommender_system import CollaborativeFilteringRecommender
            
            model = CollaborativeFilteringRecommender(
                factors=factors,
                regularization=reg,
                iterations=iters
            )
            model.fit(interactions)
            
            # Simple evaluation - check if model can generate recommendations
            test_playlists = interactions['playlist_id'].unique()[:3]
            total_recs = 0
            
            for playlist_id in test_playlists:
                try:
                    recs = model.recommend_for_user(playlist_id, 10)
                    total_recs += len(recs)
                except:
                    pass
            
            score = total_recs / len(test_playlists) if test_playlists else 0
            
            logger.info(f"CF params {factors, reg, iters}: {score:.2f} avg recommendations")
            
            if score > best_score:
                best_score = score
                best_params = {'factors': factors, 'regularization': reg, 'iterations': iters}
                
        except Exception as e:
            logger.warning(f"Error with CF params {factors, reg, iters}: {e}")
            continue
    
    logger.info(f"Best CF parameters: {best_params} with score: {best_score:.2f}")
    return best_params


def tune_content_based_params(track_data: pd.DataFrame) -> dict:
    """Tune content-based filtering parameters."""
    logger.info("Tuning content-based parameters...")
    
    # Define parameter grid for content-based
    content_params = {
        'max_features': [500, 1000, 2000],
        'n_components': [20, 50, 100]
    }
    
    best_score = 0.0
    best_params = None
    
    for max_feat, n_comp in product(
        content_params['max_features'],
        content_params['n_components']
    ):
        try:
            from ml.recommender_system import ContentBasedRecommender
            
            model = ContentBasedRecommender(
                max_features=max_feat,
                n_components=n_comp
            )
            model.fit(track_data)
            
            # Simple evaluation - check if model can generate recommendations
            test_tracks = track_data['track_id'].unique()[:3]
            total_recs = 0
            
            for track_id in test_tracks:
                try:
                    recs = model.recommend_similar(track_id, 10)
                    total_recs += len(recs)
                except:
                    pass
            
            score = total_recs / len(test_tracks) if test_tracks else 0
            
            logger.info(f"Content params {max_feat, n_comp}: {score:.2f} avg recommendations")
            
            if score > best_score:
                best_score = score
                best_params = {'max_features': max_feat, 'n_components': n_comp}
                
        except Exception as e:
            logger.warning(f"Error with content params {max_feat, n_comp}: {e}")
            continue
    
    logger.info(f"Best content parameters: {best_params} with score: {best_score:.2f}")
    return best_params


def main():
    """Main hyperparameter tuning pipeline."""
    logger.info("Starting hyperparameter tuning...")
    
    # Get database session
    session = get_session()
    
    try:
        # Load data
        interactions, track_data, mood_data = load_data(session)
        
        if len(interactions) == 0:
            logger.error("No interaction data available.")
            return
        
        # Tune individual component parameters
        logger.info("=== Tuning Individual Components ===")
        
        cf_best_params = tune_collaborative_filtering_params(interactions)
        content_best_params = tune_content_based_params(track_data)
        
        # Tune hybrid model weights
        logger.info("=== Tuning Hybrid Model Weights ===")
        
        grid_results = grid_search_hyperparameters(interactions, track_data, mood_data)
        
        # Print final results
        logger.info("=== HYPERPARAMETER TUNING RESULTS ===")
        logger.info(f"Best Collaborative Filtering params: {cf_best_params}")
        logger.info(f"Best Content-Based params: {content_best_params}")
        logger.info(f"Best Hybrid weights: {grid_results['best_config']}")
        logger.info(f"Best Hybrid score: {grid_results['best_score']:.4f}")
        
        # Save results
        results = {
            'collaborative_filtering': cf_best_params,
            'content_based': content_best_params,
            'hybrid_weights': grid_results['best_config'],
            'hybrid_score': grid_results['best_score'],
            'all_hybrid_results': grid_results['all_results']
        }
        
        import json
        with open('hyperparameter_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Results saved to hyperparameter_results.json")
        logger.info("Hyperparameter tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
