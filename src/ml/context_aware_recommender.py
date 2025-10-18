"""
Context-Aware Recommender System
Implements time-of-day, activity, and seasonal context awareness.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, time
from enum import Enum
import json

from ml.recommender_system import HybridRecommender

logger = logging.getLogger(__name__)


class TimeOfDay(Enum):
    """Time of day categories."""
    EARLY_MORNING = "early_morning"  # 5-8 AM
    MORNING = "morning"              # 8-12 PM
    AFTERNOON = "afternoon"         # 12-5 PM
    EVENING = "evening"             # 5-8 PM
    NIGHT = "night"                 # 8 PM-5 AM


class Activity(Enum):
    """Activity categories."""
    WORKOUT = "workout"
    STUDY = "study"
    WORK = "work"
    RELAX = "relax"
    PARTY = "party"
    COMMUTE = "commute"
    SLEEP = "sleep"


class Season(Enum):
    """Season categories."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


class ContextAwareRecommender:
    """Context-aware recommender that considers time, activity, and season."""
    
    def __init__(self, base_recommender: HybridRecommender):
        self.base_recommender = base_recommender
        self.context_rules = self._initialize_context_rules()
        
    def _initialize_context_rules(self) -> Dict[str, Dict[str, float]]:
        """Initialize context-based adjustment rules."""
        return {
            'time_of_day': {
                'early_morning': {
                    'energy_boost': 0.2,      # Boost energetic songs
                    'calm_penalty': -0.1,     # Penalize calm songs
                    'tempo_boost': 0.15       # Boost high tempo
                },
                'morning': {
                    'energy_boost': 0.15,
                    'calm_penalty': -0.05,
                    'tempo_boost': 0.1
                },
                'afternoon': {
                    'energy_boost': 0.1,
                    'calm_penalty': 0.0,
                    'tempo_boost': 0.05
                },
                'evening': {
                    'energy_boost': 0.0,
                    'calm_penalty': 0.0,
                    'tempo_boost': 0.0
                },
                'night': {
                    'energy_boost': -0.1,     # Reduce energetic songs
                    'calm_boost': 0.15,       # Boost calm songs
                    'tempo_penalty': -0.1     # Penalize high tempo
                }
            },
            'activity': {
                'workout': {
                    'energy_boost': 0.3,
                    'danceability_boost': 0.2,
                    'tempo_boost': 0.25,
                    'calm_penalty': -0.2
                },
                'study': {
                    'energy_penalty': -0.2,
                    'calm_boost': 0.2,
                    'tempo_penalty': -0.15,
                    'instrumental_boost': 0.1
                },
                'work': {
                    'energy_penalty': -0.1,
                    'calm_boost': 0.1,
                    'tempo_penalty': -0.1
                },
                'relax': {
                    'calm_boost': 0.2,
                    'energy_penalty': -0.15,
                    'tempo_penalty': -0.1
                },
                'party': {
                    'energy_boost': 0.3,
                    'danceability_boost': 0.3,
                    'tempo_boost': 0.2
                },
                'commute': {
                    'energy_boost': 0.1,
                    'tempo_boost': 0.1
                },
                'sleep': {
                    'calm_boost': 0.3,
                    'energy_penalty': -0.3,
                    'tempo_penalty': -0.2
                }
            },
            'season': {
                'spring': {
                    'valence_boost': 0.1,     # Boost positive mood
                    'energy_boost': 0.05
                },
                'summer': {
                    'energy_boost': 0.1,
                    'danceability_boost': 0.1,
                    'tempo_boost': 0.05
                },
                'fall': {
                    'calm_boost': 0.05,
                    'energy_penalty': -0.05
                },
                'winter': {
                    'calm_boost': 0.1,
                    'energy_penalty': -0.1
                }
            }
        }
    
    def _get_time_of_day(self, current_time: Optional[datetime] = None) -> TimeOfDay:
        """Determine current time of day."""
        if current_time is None:
            current_time = datetime.now()
        
        hour = current_time.hour
        
        if 5 <= hour < 8:
            return TimeOfDay.EARLY_MORNING
        elif 8 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 20:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT
    
    def _get_season(self, current_date: Optional[datetime] = None) -> Season:
        """Determine current season."""
        if current_date is None:
            current_date = datetime.now()
        
        month = current_date.month
        
        if month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        elif month in [9, 10, 11]:
            return Season.FALL
        else:
            return Season.WINTER
    
    def _apply_context_adjustments(self, 
                                  recommendations: List[Tuple[str, float]],
                                  context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Apply context-based adjustments to recommendations."""
        if not recommendations:
            return recommendations
        
        adjusted_recs = []
        
        for track_id, score in recommendations:
            adjusted_score = score
            
            # Get track metadata for context adjustments
            track_metadata = self._get_track_metadata(track_id)
            if not track_metadata:
                adjusted_recs.append((track_id, score))
                continue
            
            # Apply time of day adjustments
            if 'time_of_day' in context:
                time_adjustments = self.context_rules['time_of_day'].get(
                    context['time_of_day'], {}
                )
                adjusted_score = self._apply_feature_adjustments(
                    adjusted_score, track_metadata, time_adjustments
                )
            
            # Apply activity adjustments
            if 'activity' in context:
                activity_adjustments = self.context_rules['activity'].get(
                    context['activity'], {}
                )
                adjusted_score = self._apply_feature_adjustments(
                    adjusted_score, track_metadata, activity_adjustments
                )
            
            # Apply seasonal adjustments
            if 'season' in context:
                season_adjustments = self.context_rules['season'].get(
                    context['season'], {}
                )
                adjusted_score = self._apply_feature_adjustments(
                    adjusted_score, track_metadata, season_adjustments
                )
            
            # Apply weather adjustments if available
            if 'weather' in context:
                weather_adjustments = self._get_weather_adjustments(context['weather'])
                adjusted_score = self._apply_feature_adjustments(
                    adjusted_score, track_metadata, weather_adjustments
                )
            
            adjusted_recs.append((track_id, max(0.0, adjusted_score)))  # Ensure non-negative
        
        # Re-sort by adjusted scores
        adjusted_recs.sort(key=lambda x: x[1], reverse=True)
        return adjusted_recs
    
    def _apply_feature_adjustments(self, 
                                  score: float,
                                  track_metadata: Dict[str, float],
                                  adjustments: Dict[str, float]) -> float:
        """Apply feature-based score adjustments."""
        adjusted_score = score
        
        for adjustment_type, adjustment_value in adjustments.items():
            if adjustment_type == 'energy_boost' and 'energy' in track_metadata:
                if track_metadata['energy'] > 0.6:  # High energy track
                    adjusted_score *= (1 + adjustment_value)
            elif adjustment_type == 'energy_penalty' and 'energy' in track_metadata:
                if track_metadata['energy'] > 0.6:  # High energy track
                    adjusted_score *= (1 + adjustment_value)
            elif adjustment_type == 'calm_boost' and 'energy' in track_metadata:
                if track_metadata['energy'] < 0.4:  # Low energy track
                    adjusted_score *= (1 + adjustment_value)
            elif adjustment_type == 'calm_penalty' and 'energy' in track_metadata:
                if track_metadata['energy'] < 0.4:  # Low energy track
                    adjusted_score *= (1 + adjustment_value)
            elif adjustment_type == 'danceability_boost' and 'danceability' in track_metadata:
                if track_metadata['danceability'] > 0.6:
                    adjusted_score *= (1 + adjustment_value)
            elif adjustment_type == 'tempo_boost' and 'tempo' in track_metadata:
                if track_metadata['tempo'] > 120:  # High tempo
                    adjusted_score *= (1 + adjustment_value)
            elif adjustment_type == 'tempo_penalty' and 'tempo' in track_metadata:
                if track_metadata['tempo'] > 120:  # High tempo
                    adjusted_score *= (1 + adjustment_value)
            elif adjustment_type == 'valence_boost' and 'valence' in track_metadata:
                if track_metadata['valence'] > 0.6:  # Positive valence
                    adjusted_score *= (1 + adjustment_value)
        
        return adjusted_score
    
    def _get_track_metadata(self, track_id: str) -> Optional[Dict[str, float]]:
        """Get track metadata for context adjustments."""
        # This would typically query the database
        # For now, return a placeholder
        return {
            'energy': 0.5,
            'danceability': 0.5,
            'valence': 0.5,
            'tempo': 120.0
        }
    
    def _get_weather_adjustments(self, weather: str) -> Dict[str, float]:
        """Get weather-based adjustments."""
        weather_rules = {
            'sunny': {'energy_boost': 0.1, 'valence_boost': 0.1},
            'rainy': {'calm_boost': 0.1, 'energy_penalty': -0.1},
            'cloudy': {'calm_boost': 0.05},
            'snowy': {'calm_boost': 0.1, 'energy_penalty': -0.05}
        }
        return weather_rules.get(weather, {})
    
    def recommend_with_context(self,
                              user_id: str,
                              user_tracks: List[str],
                              context: Optional[Dict[str, Any]] = None,
                              target_mood: Optional[str] = None,
                              n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Get context-aware recommendations."""
        logger.info(f"Generating context-aware recommendations for user {user_id}")
        
        # Auto-detect context if not provided
        if context is None:
            context = self._auto_detect_context()
        
        # Get base recommendations
        base_recommendations = self.base_recommender.recommend(
            user_id=user_id,
            user_tracks=user_tracks,
            target_mood=target_mood,
            context=context,
            n_recommendations=n_recommendations * 2  # Get more for filtering
        )
        
        # Apply context adjustments
        context_aware_recs = self._apply_context_adjustments(base_recommendations, context)
        
        # Return top recommendations
        return context_aware_recs[:n_recommendations]
    
    def _auto_detect_context(self) -> Dict[str, Any]:
        """Auto-detect context from current time and date."""
        current_time = datetime.now()
        
        context = {
            'time_of_day': self._get_time_of_day(current_time).value,
            'season': self._get_season(current_time).value,
            'day_of_week': current_time.strftime('%A').lower(),
            'hour': current_time.hour
        }
        
        # Suggest activity based on time
        hour = current_time.hour
        if 6 <= hour < 9:
            context['suggested_activity'] = 'commute'
        elif 9 <= hour < 17:
            context['suggested_activity'] = 'work'
        elif 17 <= hour < 20:
            context['suggested_activity'] = 'relax'
        elif 20 <= hour < 23:
            context['suggested_activity'] = 'party'
        else:
            context['suggested_activity'] = 'sleep'
        
        return context
    
    def get_context_suggestions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get suggestions for optimal context settings."""
        suggestions = {}
        
        # Time-based suggestions
        if 'time_of_day' in context:
            time_suggestions = {
                'early_morning': {
                    'recommended_activities': ['workout', 'commute'],
                    'mood_suggestions': ['energetic', 'happy'],
                    'avoid_moods': ['calm', 'melancholic']
                },
                'morning': {
                    'recommended_activities': ['work', 'study'],
                    'mood_suggestions': ['happy', 'energetic'],
                    'avoid_moods': ['sad', 'melancholic']
                },
                'afternoon': {
                    'recommended_activities': ['work', 'study'],
                    'mood_suggestions': ['happy', 'neutral'],
                    'avoid_moods': ['angry']
                },
                'evening': {
                    'recommended_activities': ['relax', 'party'],
                    'mood_suggestions': ['happy', 'romantic'],
                    'avoid_moods': ['angry']
                },
                'night': {
                    'recommended_activities': ['relax', 'sleep'],
                    'mood_suggestions': ['calm', 'romantic'],
                    'avoid_moods': ['energetic', 'angry']
                }
            }
            suggestions.update(time_suggestions.get(context['time_of_day'], {}))
        
        # Activity-based suggestions
        if 'activity' in context:
            activity_suggestions = {
                'workout': {
                    'recommended_moods': ['energetic', 'happy'],
                    'energy_range': [0.7, 1.0],
                    'tempo_range': [120, 200]
                },
                'study': {
                    'recommended_moods': ['calm', 'neutral'],
                    'energy_range': [0.2, 0.5],
                    'tempo_range': [60, 120]
                },
                'relax': {
                    'recommended_moods': ['calm', 'romantic'],
                    'energy_range': [0.1, 0.4],
                    'tempo_range': [60, 100]
                }
            }
            suggestions.update(activity_suggestions.get(context['activity'], {}))
        
        return suggestions


class ContextLearningRecommender(ContextAwareRecommender):
    """Context-aware recommender that learns from user feedback."""
    
    def __init__(self, base_recommender: HybridRecommender):
        super().__init__(base_recommender)
        self.user_context_preferences = {}
        self.context_feedback_history = []
    
    def record_feedback(self, 
                       user_id: str,
                       track_id: str,
                       context: Dict[str, Any],
                       feedback: str,  # 'positive', 'negative', 'skip'
                       rating: Optional[float] = None):
        """Record user feedback for context learning."""
        feedback_record = {
            'user_id': user_id,
            'track_id': track_id,
            'context': context,
            'feedback': feedback,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }
        
        self.context_feedback_history.append(feedback_record)
        
        # Update user context preferences
        if user_id not in self.user_context_preferences:
            self.user_context_preferences[user_id] = {}
        
        # Learn from positive feedback
        if feedback == 'positive' and rating and rating > 3:
            for context_key, context_value in context.items():
                if context_key not in self.user_context_preferences[user_id]:
                    self.user_context_preferences[user_id][context_key] = {}
                
                if context_value not in self.user_context_preferences[user_id][context_key]:
                    self.user_context_preferences[user_id][context_key][context_value] = 0
                
                self.user_context_preferences[user_id][context_key][context_value] += 1
    
    def get_personalized_context_weights(self, user_id: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Get personalized context weights based on user history."""
        if user_id not in self.user_context_preferences:
            return {}
        
        user_prefs = self.user_context_preferences[user_id]
        personalized_weights = {}
        
        for context_key, context_value in context.items():
            if context_key in user_prefs and context_value in user_prefs[context_key]:
                # Calculate weight based on user preference frequency
                total_preferences = sum(user_prefs[context_key].values())
                preference_count = user_prefs[context_key][context_value]
                weight = preference_count / total_preferences if total_preferences > 0 else 0
                personalized_weights[context_key] = weight
        
        return personalized_weights
    
    def recommend_with_learning(self,
                              user_id: str,
                              user_tracks: List[str],
                              context: Optional[Dict[str, Any]] = None,
                              target_mood: Optional[str] = None,
                              n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Get recommendations with learned context preferences."""
        # Get personalized context weights
        if context:
            personalized_weights = self.get_personalized_context_weights(user_id, context)
            
            # Apply personalized weights to context rules
            for context_key, weight in personalized_weights.items():
                if context_key in self.context_rules:
                    for rule_key, rule_value in self.context_rules[context_key].items():
                        if context_key in context and rule_key == context[context_key]:
                            # Boost rules that user has positive history with
                            self.context_rules[context_key][rule_key] *= (1 + weight * 0.5)
        
        # Get context-aware recommendations
        return self.recommend_with_context(
            user_id=user_id,
            user_tracks=user_tracks,
            context=context,
            target_mood=target_mood,
            n_recommendations=n_recommendations
        )
