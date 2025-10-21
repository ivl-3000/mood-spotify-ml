"""
Hybrid Recommender System for Phase 4
Implements collaborative filtering, content-based, and mood-aware recommenders.
"""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import BM25Recommender
import warnings
warnings.filterwarnings('ignore')

from db.models import Track, AudioFeatures, PlaylistTrack, Lyrics, LyricsNLP, SocialNLP
from db.engine import get_session

logger = logging.getLogger(__name__)

class CollaborativeFilteringRecommender:
    """Collaborative filtering using ALS and Matrix Factorization."""
    
    def __init__(self, factors: int = 50, regularization: float = 0.01, iterations: int = 50):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.model = None
        self.user_item_matrix = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def _build_user_item_matrix(self, interactions: pd.DataFrame) -> csr_matrix:
        """Build user-item interaction matrix from playlist data."""
        logger.info("Building user-item matrix from playlist interactions...")
        
        # Create mappings
        unique_users = interactions['playlist_id'].unique()
        unique_items = interactions['track_id'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Build sparse matrix
        rows = [self.user_mapping[user] for user in interactions['playlist_id']]
        cols = [self.item_mapping[item] for item in interactions['track_id']]
        data = np.ones(len(interactions))  # Binary interactions
        
        matrix = csr_matrix((data, (rows, cols)), 
                           shape=(len(unique_users), len(unique_items)))
        
        logger.info(f"Built matrix: {matrix.shape[0]} users, {matrix.shape[1]} items")
        return matrix
    
    def fit(self, interactions: pd.DataFrame) -> None:
        """Train the collaborative filtering model."""
        logger.info("Training collaborative filtering model...")
        
        self.user_item_matrix = self._build_user_item_matrix(interactions)
        
        # Train ALS model
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42
        )
        
        self.model.fit(self.user_item_matrix)
        logger.info("Collaborative filtering model trained successfully")
    
    def recommend_for_user(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Get recommendations for a specific user."""
        if user_id not in self.user_mapping:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_mapping[user_id]
        recommendations = self.model.recommend(
            user_idx, 
            self.user_item_matrix[user_idx], 
            N=n_recommendations,
            filter_already_liked_items=False
        )
        
        # Convert back to track IDs
        results = []
        for recommendation in recommendations:
            if len(recommendation) == 2:
                item_idx, score = recommendation
                track_id = self.reverse_item_mapping[item_idx]
                results.append((track_id, score))
            else:
                # Handle case where recommendation format is different
                item_idx = recommendation[0] if len(recommendation) > 0 else None
                score = recommendation[1] if len(recommendation) > 1 else 0.0
                if item_idx is not None and item_idx in self.reverse_item_mapping:
                    track_id = self.reverse_item_mapping[item_idx]
                    results.append((track_id, score))
        
        return results
    
    def get_similar_items(self, item_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get similar items to a given item."""
        if item_id not in self.item_mapping:
            logger.warning(f"Item {item_id} not found in training data")
            return []
        
        item_idx = self.item_mapping[item_id]
        similar_items = self.model.similar_items(item_idx, n_similar)
        
        results = []
        for similar_item in similar_items:
            if len(similar_item) == 2:
                similar_idx, score = similar_item
                track_id = self.reverse_item_mapping[similar_idx]
                results.append((track_id, score))
            else:
                # Handle case where format is different
                similar_idx = similar_item[0] if len(similar_item) > 0 else None
                score = similar_item[1] if len(similar_item) > 1 else 0.0
                if similar_idx is not None and similar_idx in self.reverse_item_mapping:
                    track_id = self.reverse_item_mapping[similar_idx]
                    results.append((track_id, score))
        
        return results


class ContentBasedRecommender:
    """Content-based recommender using lyrics and audio features."""
    
    def __init__(self, max_features: int = 1000, n_components: int = 50):
        self.max_features = max_features
        self.n_components = n_components
        self.tfidf_vectorizer = None
        self.svd = None
        self.audio_scaler = StandardScaler()
        self.track_features = None
        self.track_mapping = {}
        self.reverse_track_mapping = {}
        
    def _extract_lyrics_features(self, lyrics_data: pd.DataFrame) -> np.ndarray:
        """Extract TF-IDF features from lyrics."""
        logger.info("Extracting lyrics features using TF-IDF...")
        
        # Clean and preprocess lyrics
        lyrics_texts = lyrics_data['lyrics_text'].fillna('').astype(str)
        
        # TF-IDF vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(lyrics_texts)
        
        # Dimensionality reduction (adjust n_components based on actual features)
        actual_features = min(self.n_components, tfidf_matrix.shape[1])
        self.svd = TruncatedSVD(n_components=actual_features, random_state=42)
        lyrics_features = self.svd.fit_transform(tfidf_matrix)
        
        logger.info(f"Extracted lyrics features: {lyrics_features.shape}")
        return lyrics_features
    
    def _extract_audio_features(self, audio_data: pd.DataFrame) -> np.ndarray:
        """Extract and normalize audio features."""
        logger.info("Extracting audio features...")
        
        audio_columns = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
        audio_features = audio_data[audio_columns].fillna(0)
        
        # Normalize audio features
        audio_features_scaled = self.audio_scaler.fit_transform(audio_features)
        
        logger.info(f"Extracted audio features: {audio_features_scaled.shape}")
        return audio_features_scaled
    
    def _combine_features(self, lyrics_features: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        """Combine lyrics and audio features."""
        # Weighted combination (lyrics: 0.7, audio: 0.3)
        combined = np.hstack([
            lyrics_features * 0.7,
            audio_features * 0.3
        ])
        
        logger.info(f"Combined features shape: {combined.shape}")
        return combined
    
    def fit(self, track_data: pd.DataFrame) -> None:
        """Train the content-based model."""
        logger.info("Training content-based model...")
        
        # Create track mapping
        unique_tracks = track_data['track_id'].unique()
        self.track_mapping = {track: idx for idx, track in enumerate(unique_tracks)}
        self.reverse_track_mapping = {idx: track for track, idx in self.track_mapping.items()}
        
        # Extract features
        lyrics_features = self._extract_lyrics_features(track_data)
        audio_features = self._extract_audio_features(track_data)
        
        # Combine features
        self.track_features = self._combine_features(lyrics_features, audio_features)
        
        logger.info("Content-based model trained successfully")
    
    def recommend_similar(self, track_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Get content-based recommendations for a track."""
        if track_id not in self.track_mapping:
            logger.warning(f"Track {track_id} not found in training data")
            return []
        
        track_idx = self.track_mapping[track_id]
        track_vector = self.track_features[track_idx].reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(track_vector, self.track_features)[0]
        
        # Get top similar tracks (excluding the input track)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        results = []
        for idx in similar_indices:
            track_id = self.reverse_track_mapping[idx]
            similarity = similarities[idx]
            results.append((track_id, similarity))
        
        return results
    
    def recommend_for_user_profile(self, user_tracks: List[str], n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Get recommendations based on user's track history."""
        if not user_tracks:
            return []
        
        # Calculate user profile as average of track features
        user_profile = np.zeros(self.track_features.shape[1])
        valid_tracks = 0
        
        for track_id in user_tracks:
            if track_id in self.track_mapping:
                track_idx = self.track_mapping[track_id]
                user_profile += self.track_features[track_idx]
                valid_tracks += 1
        
        if valid_tracks == 0:
            return []
        
        user_profile /= valid_tracks
        
        # Calculate similarities
        similarities = cosine_similarity([user_profile], self.track_features)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        results = []
        for idx in top_indices:
            track_id = self.reverse_track_mapping[idx]
            similarity = similarities[idx]
            results.append((track_id, similarity))
        
        return results


class MoodAwareRecommender:
    """Mood-aware recommender using mood tags and sentiment analysis."""
    
    def __init__(self, mood_weights: Optional[Dict[str, float]] = None):
        self.mood_weights = mood_weights or {
            'happy': 1.0, 'sad': 1.0, 'angry': 1.0, 'calm': 1.0,
            'energetic': 1.0, 'romantic': 1.0, 'melancholic': 1.0, 'neutral': 1.0
        }
        self.mood_features = None
        self.track_mood_mapping = {}
        
    def _extract_mood_features(self, mood_data: pd.DataFrame) -> np.ndarray:
        """Extract mood features from sentiment and emotion data."""
        logger.info("Extracting mood features...")
        
        # Combine sentiment scores and emotion probabilities
        mood_features = []
        
        for _, row in mood_data.iterrows():
            features = []
            
            # Sentiment score
            sentiment = row.get('sentiment_score', 0.0)
            features.append(sentiment)
            
            # Emotion probabilities (handle both dict and string)
            emotion_probs = row.get('emotion_probs', {})
            if isinstance(emotion_probs, str):
                try:
                    import json
                    emotion_probs = json.loads(emotion_probs)
                except:
                    emotion_probs = {}
            
            for mood in self.mood_weights.keys():
                prob = emotion_probs.get(mood, 0.0) if isinstance(emotion_probs, dict) else 0.0
                features.append(prob)
            
            # Dominant emotion (one-hot encoded)
            dominant_emotion = row.get('dominant_emotion', 'neutral')
            for mood in self.mood_weights.keys():
                features.append(1.0 if mood == dominant_emotion else 0.0)
            
            mood_features.append(features)
        
        mood_array = np.array(mood_features)
        logger.info(f"Extracted mood features: {mood_array.shape}")
        return mood_array
    
    def fit(self, mood_data: pd.DataFrame) -> None:
        """Train the mood-aware model."""
        logger.info("Training mood-aware model...")
        
        # Create track mapping
        unique_tracks = mood_data['track_id'].unique()
        self.track_mood_mapping = {track: idx for idx, track in enumerate(unique_tracks)}
        
        # Extract mood features
        self.mood_features = self._extract_mood_features(mood_data)
        
        logger.info("Mood-aware model trained successfully")
    
    def recommend_by_mood(self, target_mood: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Get recommendations based on target mood."""
        if target_mood not in self.mood_weights:
            logger.warning(f"Unknown mood: {target_mood}")
            return []
        
        # Find tracks with high probability for target mood
        mood_idx = list(self.mood_weights.keys()).index(target_mood)
        mood_scores = self.mood_features[:, mood_idx + 1]  # +1 for sentiment score
        
        # Get top tracks
        top_indices = np.argsort(mood_scores)[::-1][:n_recommendations]
        
        results = []
        for idx in top_indices:
            track_id = list(self.track_mood_mapping.keys())[idx]
            score = mood_scores[idx]
            results.append((track_id, score))
        
        return results
    
    def recommend_mood_similar(self, track_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Get mood-based similar tracks."""
        if track_id not in self.track_mood_mapping:
            logger.warning(f"Track {track_id} not found in mood data")
            return []
        
        track_idx = self.track_mood_mapping[track_id]
        track_mood_vector = self.mood_features[track_idx].reshape(1, -1)
        
        # Calculate mood similarities
        similarities = cosine_similarity(track_mood_vector, self.mood_features)[0]
        
        # Get top similar tracks
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        results = []
        for idx in similar_indices:
            track_id = list(self.track_mood_mapping.keys())[idx]
            similarity = similarities[idx]
            results.append((track_id, similarity))
        
        return results


class HybridRecommender:
    """Hybrid recommender combining collaborative, content-based, and mood-aware approaches."""
    
    def __init__(self, 
                 cf_weight: float = 0.4,
                 content_weight: float = 0.3,
                 mood_weight: float = 0.3,
                 context_weights: Optional[Dict[str, float]] = None):
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.mood_weight = mood_weight
        self.context_weights = context_weights or {
            'time_of_day': 0.1,
            'activity': 0.1,
            'season': 0.05
        }
        
        self.cf_model = CollaborativeFilteringRecommender()
        self.content_model = ContentBasedRecommender()
        self.mood_model = MoodAwareRecommender()
        
        self.track_metadata = {}
        
    def fit(self, 
            interactions: pd.DataFrame,
            track_data: pd.DataFrame,
            mood_data: pd.DataFrame) -> None:
        """Train all components of the hybrid model."""
        logger.info("Training hybrid recommender system...")
        
        # Train individual models
        self.cf_model.fit(interactions)
        self.content_model.fit(track_data)
        self.mood_model.fit(mood_data)
        
        # Store track metadata for context
        self.track_metadata = track_data.set_index('track_id').to_dict('index')
        
        logger.info("Hybrid recommender system trained successfully")
    
    def recommend(self, 
                  user_id: str,
                  user_tracks: List[str],
                  target_mood: Optional[str] = None,
                  context: Optional[Dict[str, Any]] = None,
                  n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Get hybrid recommendations."""
        logger.info(f"Generating hybrid recommendations for user {user_id}")
        
        # Get recommendations from each model
        cf_recs = self.cf_model.recommend_for_user(user_id, n_recommendations * 2)
        content_recs = self.content_model.recommend_for_user_profile(user_tracks, n_recommendations * 2)
        
        if target_mood:
            mood_recs = self.mood_model.recommend_by_mood(target_mood, n_recommendations * 2)
        else:
            mood_recs = []
            if user_tracks:
                # Use mood of user's tracks
                for track_id in user_tracks[:3]:  # Use first 3 tracks
                    track_mood_recs = self.mood_model.recommend_mood_similar(track_id, 5)
                    mood_recs.extend(track_mood_recs)
        
        # Combine recommendations with weights
        combined_scores = {}
        
        # Add collaborative filtering scores
        for track_id, score in cf_recs:
            combined_scores[track_id] = combined_scores.get(track_id, 0) + score * self.cf_weight
        
        # Add content-based scores
        for track_id, score in content_recs:
            combined_scores[track_id] = combined_scores.get(track_id, 0) + score * self.content_weight
        
        # Add mood-based scores
        for track_id, score in mood_recs:
            combined_scores[track_id] = combined_scores.get(track_id, 0) + score * self.mood_weight
        
        # Apply context adjustments
        if context:
            combined_scores = self._apply_context_weights(combined_scores, context)
        
        # Sort by combined score and return top recommendations
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_recs[:n_recommendations]
    
    def _apply_context_weights(self, scores: Dict[str, float], context: Dict[str, Any]) -> Dict[str, float]:
        """Apply context-based adjustments to scores."""
        adjusted_scores = scores.copy()
        
        # Time of day adjustments
        if 'time_of_day' in context:
            time_weight = self.context_weights['time_of_day']
            # Example: boost energetic songs in morning, calm songs at night
            for track_id in adjusted_scores.keys():
                if track_id in self.track_metadata:
                    metadata = self.track_metadata[track_id]
                    energy = metadata.get('energy', 0.5)
                    
                    if context['time_of_day'] == 'morning' and energy > 0.7:
                        adjusted_scores[track_id] *= (1 + time_weight)
                    elif context['time_of_day'] == 'night' and energy < 0.4:
                        adjusted_scores[track_id] *= (1 + time_weight)
        
        # Activity-based adjustments
        if 'activity' in context:
            activity_weight = self.context_weights['activity']
            for track_id in adjusted_scores.keys():
                if track_id in self.track_metadata:
                    metadata = self.track_metadata[track_id]
                    danceability = metadata.get('danceability', 0.5)
                    
                    if context['activity'] == 'workout' and danceability > 0.7:
                        adjusted_scores[track_id] *= (1 + activity_weight)
                    elif context['activity'] == 'study' and danceability < 0.4:
                        adjusted_scores[track_id] *= (1 + activity_weight)
        
        return adjusted_scores


class RecommenderEvaluator:
    """Evaluation metrics for recommender systems."""
    
    @staticmethod
    def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        if len(recommended_k) == 0:
            return 0.0
        
        return len(recommended_k.intersection(relevant_set)) / len(recommended_k)
    
    @staticmethod
    def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
        """Calculate Recall@K."""
        if len(relevant) == 0:
            return 0.0
        
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        return len(recommended_k.intersection(relevant_set)) / len(relevant_set)
    
    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
        """Calculate NDCG@K."""
        if k == 0 or len(relevant) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def evaluate_recommender(recommender: HybridRecommender,
                           test_data: pd.DataFrame,
                           k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[str, float]]:
        """Evaluate recommender system on test data."""
        logger.info("Evaluating recommender system...")
        
        results = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for _, row in test_data.iterrows():
                user_id = row['user_id']
                user_tracks = row['user_tracks']
                relevant_tracks = row['relevant_tracks']
                
                # Get recommendations
                recommendations = recommender.recommend(
                    user_id=user_id,
                    user_tracks=user_tracks,
                    n_recommendations=k
                )
                
                recommended_tracks = [track_id for track_id, _ in recommendations]
                
                # Calculate metrics
                precision = RecommenderEvaluator.precision_at_k(recommended_tracks, relevant_tracks, k)
                recall = RecommenderEvaluator.recall_at_k(recommended_tracks, relevant_tracks, k)
                ndcg = RecommenderEvaluator.ndcg_at_k(recommended_tracks, relevant_tracks, k)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
            
            results[f'k={k}'] = {
                'precision': np.mean(precision_scores),
                'recall': np.mean(recall_scores),
                'ndcg': np.mean(ndcg_scores)
            }
        
        logger.info("Evaluation completed")
        return results
