"""
User Journey Simulation for Recommender System
Simulates realistic user journeys with mood-based playlist creation.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json
import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from db.engine import get_session
from ml.recommender_system import HybridRecommender
from ml.context_aware_recommender import ContextAwareRecommender

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('user_journey_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UserJourneySimulator:
    """Simulates realistic user journeys with mood-based playlist creation."""
    
    def __init__(self, recommender_model: HybridRecommender):
        self.recommender = recommender_model
        self.context_aware_model = ContextAwareRecommender(recommender_model)
        self.session = get_session()
        
        # User journey patterns
        self.mood_patterns = {
            'morning': ['energetic', 'happy', 'motivated'],
            'afternoon': ['focused', 'neutral', 'productive'],
            'evening': ['relaxed', 'romantic', 'calm'],
            'night': ['melancholic', 'calm', 'sleepy']
        }
        
        self.activity_patterns = {
            'workout': ['energetic', 'motivated', 'pumped'],
            'study': ['focused', 'calm', 'concentrated'],
            'work': ['productive', 'focused', 'neutral'],
            'relax': ['calm', 'peaceful', 'relaxed'],
            'party': ['energetic', 'happy', 'excited'],
            'commute': ['neutral', 'energetic', 'focused']
        }
        
        self.seasonal_moods = {
            'spring': ['happy', 'energetic', 'optimistic'],
            'summer': ['energetic', 'happy', 'excited'],
            'fall': ['melancholic', 'calm', 'reflective'],
            'winter': ['calm', 'cozy', 'melancholic']
        }
    
    def generate_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Generate a realistic user profile."""
        # Simulate user preferences based on demographics
        age_groups = ['teen', 'young_adult', 'adult', 'senior']
        age_group = random.choice(age_groups)
        
        # Age-based preferences
        age_preferences = {
            'teen': {'genres': ['pop', 'hip-hop', 'electronic'], 'energy': 0.8},
            'young_adult': {'genres': ['pop', 'rock', 'indie'], 'energy': 0.7},
            'adult': {'genres': ['rock', 'classical', 'jazz'], 'energy': 0.5},
            'senior': {'genres': ['classical', 'jazz', 'folk'], 'energy': 0.3}
        }
        
        profile = {
            'user_id': user_id,
            'age_group': age_group,
            'preferred_genres': age_preferences[age_group]['genres'],
            'energy_preference': age_preferences[age_group]['energy'],
            'mood_sensitivity': random.uniform(0.3, 0.9),
            'context_awareness': random.uniform(0.5, 1.0),
            'discovery_tendency': random.uniform(0.2, 0.8)
        }
        
        return profile
    
    def simulate_daily_journey(self, user_profile: Dict[str, Any], date: datetime) -> List[Dict[str, Any]]:
        """Simulate a single day's user journey."""
        journey = []
        
        # Define time periods and typical activities
        time_periods = [
            {'start': 6, 'end': 9, 'activity': 'morning_routine', 'context': 'morning'},
            {'start': 9, 'end': 12, 'activity': 'work', 'context': 'morning'},
            {'start': 12, 'end': 14, 'activity': 'lunch', 'context': 'afternoon'},
            {'start': 14, 'end': 17, 'activity': 'work', 'context': 'afternoon'},
            {'start': 17, 'end': 19, 'activity': 'commute', 'context': 'evening'},
            {'start': 19, 'end': 22, 'activity': 'evening', 'context': 'evening'},
            {'start': 22, 'end': 24, 'activity': 'night', 'context': 'night'}
        ]
        
        for period in time_periods:
            # Determine if user is active during this period
            activity_probability = self._get_activity_probability(period, user_profile)
            
            if random.random() < activity_probability:
                # Generate mood based on time and activity
                mood = self._generate_mood_for_context(
                    period['context'], 
                    period['activity'], 
                    user_profile
                )
                
                # Create playlist session
                session = self._create_playlist_session(
                    user_profile, 
                    period, 
                    mood, 
                    date
                )
                
                if session:
                    journey.append(session)
        
        return journey
    
    def _get_activity_probability(self, period: Dict, user_profile: Dict) -> float:
        """Calculate probability of user being active during a time period."""
        base_probabilities = {
            'morning_routine': 0.7,
            'work': 0.9,
            'lunch': 0.6,
            'commute': 0.8,
            'evening': 0.8,
            'night': 0.4
        }
        
        base_prob = base_probabilities.get(period['activity'], 0.5)
        
        # Adjust based on user profile
        if user_profile['age_group'] == 'teen':
            base_prob *= 1.2 if period['activity'] in ['evening', 'night'] else 0.8
        elif user_profile['age_group'] == 'senior':
            base_prob *= 0.8 if period['activity'] in ['night'] else 1.1
        
        return min(base_prob, 1.0)
    
    def _generate_mood_for_context(self, time_context: str, activity: str, user_profile: Dict) -> str:
        """Generate mood based on context and user profile."""
        # Get base mood from time context
        time_moods = self.mood_patterns.get(time_context, ['neutral'])
        
        # Get activity-specific moods
        activity_moods = self.activity_patterns.get(activity, ['neutral'])
        
        # Combine and weight based on user profile
        all_moods = time_moods + activity_moods
        
        # Apply user preferences
        if user_profile['energy_preference'] > 0.7:
            energetic_moods = ['energetic', 'happy', 'excited', 'motivated']
            all_moods.extend(energetic_moods)
        elif user_profile['energy_preference'] < 0.4:
            calm_moods = ['calm', 'relaxed', 'peaceful', 'melancholic']
            all_moods.extend(calm_moods)
        
        # Add some randomness
        mood_weights = {}
        for mood in set(all_moods):
            weight = 1.0
            if mood in ['energetic', 'happy'] and user_profile['energy_preference'] > 0.6:
                weight *= 2.0
            elif mood in ['calm', 'relaxed'] and user_profile['energy_preference'] < 0.5:
                weight *= 2.0
            
            mood_weights[mood] = weight
        
        # Select mood based on weights
        moods = list(mood_weights.keys())
        weights = list(mood_weights.values())
        selected_mood = np.random.choice(moods, p=np.array(weights)/sum(weights))
        
        return selected_mood
    
    def _create_playlist_session(self, user_profile: Dict, period: Dict, mood: str, date: datetime) -> Dict[str, Any]:
        """Create a playlist session for the user."""
        try:
            # Get user's existing tracks (simulate from database)
            user_tracks = self._get_user_tracks(user_profile['user_id'])
            
            # Generate context
            context = {
                'time_of_day': period['context'],
                'activity': period['activity'],
                'mood': mood,
                'season': self._get_season(date)
            }
            
            # Get recommendations
            if len(user_tracks) > 0:
                recommendations = self.context_aware_model.recommend_with_context(
                    user_id=user_profile['user_id'],
                    user_tracks=user_tracks,
                    context=context,
                    target_mood=mood,
                    n_recommendations=random.randint(5, 15)
                )
            else:
                # New user - get mood-based recommendations
                recommendations = self.recommender.mood_model.recommend_by_mood(
                    mood, random.randint(5, 15)
                )
            
            if not recommendations:
                return None
            
            # Simulate user interaction
            playlist_session = {
                'user_id': user_profile['user_id'],
                'session_id': f"{user_profile['user_id']}_{date.strftime('%Y%m%d_%H%M')}",
                'timestamp': date,
                'context': context,
                'recommendations': recommendations,
                'user_interactions': self._simulate_user_interactions(
                    recommendations, 
                    user_profile, 
                    mood
                ),
                'playlist_created': self._create_playlist_from_session(
                    recommendations, 
                    user_profile, 
                    context
                )
            }
            
            return playlist_session
            
        except Exception as e:
            logger.warning(f"Error creating playlist session: {e}")
            return None
    
    def _get_user_tracks(self, user_id: str) -> List[str]:
        """Get user's existing tracks (simulate from database)."""
        try:
            # Query user's playlist tracks
            query = """
                SELECT DISTINCT pt.track_id
                FROM playlist_tracks pt
                JOIN playlists p ON pt.playlist_id = p.id
                WHERE p.owner_id = :user_id
                LIMIT 20
            """
            result = self.session.execute(query, {"user_id": user_id}).fetchall()
            return [row[0] for row in result]
        except:
            # Return empty list if no tracks found
            return []
    
    def _get_season(self, date: datetime) -> str:
        """Determine season from date."""
        month = date.month
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'fall'
        else:
            return 'winter'
    
    def _simulate_user_interactions(self, recommendations: List[Tuple[str, float]], 
                                  user_profile: Dict, mood: str) -> Dict[str, Any]:
        """Simulate realistic user interactions with recommendations."""
        interactions = {
            'liked_tracks': [],
            'skipped_tracks': [],
            'added_to_playlist': [],
            'played_full': [],
            'skipped_early': []
        }
        
        for track_id, score in recommendations:
            # Simulate interaction based on score and user profile
            interaction_prob = self._calculate_interaction_probability(
                score, user_profile, mood
            )
            
            if random.random() < interaction_prob:
                # User liked the track
                interactions['liked_tracks'].append(track_id)
                
                # Simulate further interactions
                if random.random() < 0.7:  # 70% chance to add to playlist
                    interactions['added_to_playlist'].append(track_id)
                
                if random.random() < 0.8:  # 80% chance to play full track
                    interactions['played_full'].append(track_id)
                else:
                    interactions['skipped_early'].append(track_id)
            else:
                # User skipped the track
                interactions['skipped_tracks'].append(track_id)
        
        return interactions
    
    def _calculate_interaction_probability(self, score: float, user_profile: Dict, mood: str) -> float:
        """Calculate probability of user interacting with a recommendation."""
        base_prob = min(score, 1.0)  # Base probability from recommendation score
        
        # Adjust based on user profile
        if user_profile['mood_sensitivity'] > 0.7:
            base_prob *= 1.2  # More sensitive to mood-based recommendations
        
        if user_profile['discovery_tendency'] > 0.6:
            base_prob *= 1.1  # More likely to try new tracks
        
        # Add some randomness
        noise = random.uniform(-0.1, 0.1)
        final_prob = max(0.0, min(1.0, base_prob + noise))
        
        return final_prob
    
    def _create_playlist_from_session(self, recommendations: List[Tuple[str, float]], 
                                    user_profile: Dict, context: Dict) -> Dict[str, Any]:
        """Create a playlist from the session."""
        # Select tracks that were added to playlist
        playlist_tracks = []
        for track_id, score in recommendations:
            if random.random() < 0.6:  # 60% chance to add to playlist
                playlist_tracks.append({
                    'track_id': track_id,
                    'score': score,
                    'added_at': datetime.now()
                })
        
        if not playlist_tracks:
            return None
        
        playlist = {
            'playlist_id': f"sim_{user_profile['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'name': f"{context['mood'].title()} {context['activity'].title()} Playlist",
            'description': f"Auto-generated playlist for {context['mood']} mood during {context['activity']}",
            'owner_id': user_profile['user_id'],
            'tracks': playlist_tracks,
            'created_at': datetime.now(),
            'context': context,
            'mood': context['mood'],
            'activity': context['activity']
        }
        
        return playlist
    
    def simulate_multiple_users(self, num_users: int = 10, days: int = 7) -> Dict[str, Any]:
        """Simulate journeys for multiple users over multiple days."""
        logger.info(f"Simulating journeys for {num_users} users over {days} days...")
        
        all_journeys = []
        all_playlists = []
        
        for user_idx in range(num_users):
            user_id = f"sim_user_{user_idx:03d}"
            user_profile = self.generate_user_profile(user_id)
            
            logger.info(f"Simulating user {user_id} ({user_profile['age_group']})")
            
            for day in range(days):
                date = datetime.now() - timedelta(days=days-day-1)
                daily_journey = self.simulate_daily_journey(user_profile, date)
                
                for session in daily_journey:
                    all_journeys.append(session)
                    if session['playlist_created']:
                        all_playlists.append(session['playlist_created'])
        
        # Generate summary statistics
        summary = self._generate_journey_summary(all_journeys, all_playlists)
        
        return {
            'journeys': all_journeys,
            'playlists': all_playlists,
            'summary': summary
        }
    
    def _generate_journey_summary(self, journeys: List[Dict], playlists: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from simulated journeys."""
        if not journeys:
            return {}
        
        # Mood distribution
        mood_counts = {}
        for journey in journeys:
            mood = journey['context']['mood']
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        # Activity distribution
        activity_counts = {}
        for journey in journeys:
            activity = journey['context']['activity']
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        # Time distribution
        time_counts = {}
        for journey in journeys:
            time_of_day = journey['context']['time_of_day']
            time_counts[time_of_day] = time_counts.get(time_of_day, 0) + 1
        
        # Interaction statistics
        total_recommendations = sum(len(j['recommendations']) for j in journeys)
        total_liked = sum(len(j['user_interactions']['liked_tracks']) for j in journeys)
        total_playlists = len(playlists)
        
        return {
            'total_sessions': len(journeys),
            'total_playlists_created': total_playlists,
            'total_recommendations': total_recommendations,
            'total_liked_tracks': total_liked,
            'like_rate': total_liked / total_recommendations if total_recommendations > 0 else 0,
            'mood_distribution': mood_counts,
            'activity_distribution': activity_counts,
            'time_distribution': time_counts,
            'avg_tracks_per_playlist': np.mean([len(p['tracks']) for p in playlists]) if playlists else 0
        }
    
    def save_simulation_results(self, results: Dict[str, Any], output_dir: Path):
        """Save simulation results to files."""
        output_dir.mkdir(exist_ok=True)
        
        # Save journeys
        journeys_df = pd.DataFrame(results['journeys'])
        journeys_df.to_csv(output_dir / "user_journeys.csv", index=False)
        
        # Save playlists
        playlists_data = []
        for playlist in results['playlists']:
            playlist_data = {
                'playlist_id': playlist['playlist_id'],
                'name': playlist['name'],
                'owner_id': playlist['owner_id'],
                'mood': playlist['mood'],
                'activity': playlist['activity'],
                'created_at': playlist['created_at'],
                'num_tracks': len(playlist['tracks'])
            }
            playlists_data.append(playlist_data)
        
        playlists_df = pd.DataFrame(playlists_data)
        playlists_df.to_csv(output_dir / "simulated_playlists.csv", index=False)
        
        # Save summary
        with open(output_dir / "simulation_summary.json", 'w') as f:
            json.dump(results['summary'], f, indent=2, default=str)
        
        logger.info(f"Simulation results saved to {output_dir}")


def main():
    """Main simulation function."""
    logger.info("Starting user journey simulation...")
    
    try:
        # Load trained recommender model
        models_dir = Path("models/recommender")
        if not (models_dir / "hybrid_model.pkl").exists():
            logger.error("No trained model found. Please train the model first.")
            return
        
        import pickle
        with open(models_dir / "hybrid_model.pkl", 'rb') as f:
            hybrid_model = pickle.load(f)
        
        # Initialize simulator
        simulator = UserJourneySimulator(hybrid_model)
        
        # Run simulation
        results = simulator.simulate_multiple_users(num_users=20, days=7)
        
        # Save results
        output_dir = Path("results/simulation")
        simulator.save_simulation_results(results, output_dir)
        
        # Print summary
        summary = results['summary']
        logger.info("=== SIMULATION SUMMARY ===")
        logger.info(f"Total sessions: {summary['total_sessions']}")
        logger.info(f"Playlists created: {summary['total_playlists_created']}")
        logger.info(f"Recommendations made: {summary['total_recommendations']}")
        logger.info(f"Like rate: {summary['like_rate']:.2%}")
        logger.info(f"Avg tracks per playlist: {summary['avg_tracks_per_playlist']:.1f}")
        
        logger.info("User journey simulation completed successfully!")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()
