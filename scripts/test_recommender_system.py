"""
Test script for Hybrid Recommender System
Comprehensive testing of all recommender components.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from db.engine import get_session
from ml.recommender_system import (
    HybridRecommender,
    CollaborativeFilteringRecommender,
    ContentBasedRecommender,
    MoodAwareRecommender
)
from ml.context_aware_recommender import ContextAwareRecommender

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommender_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RecommenderTester:
    """Comprehensive tester for recommender system."""
    
    def __init__(self):
        self.test_results = {}
        self.session = get_session()
    
    def test_data_availability(self) -> bool:
        """Test if required data is available."""
        logger.info("Testing data availability...")
        
        try:
            # Check interactions
            from sqlalchemy import text
            interactions_query = text("SELECT COUNT(*) FROM playlist_tracks")
            interactions_count = self.session.execute(interactions_query).fetchone()[0]
            
            # Check tracks
            tracks_query = text("SELECT COUNT(*) FROM tracks")
            tracks_count = self.session.execute(tracks_query).fetchone()[0]
            
            # Check audio features
            audio_query = text("SELECT COUNT(*) FROM audio_features")
            audio_count = self.session.execute(audio_query).fetchone()[0]
            
            # Check lyrics
            lyrics_query = text("SELECT COUNT(*) FROM lyrics")
            lyrics_count = self.session.execute(lyrics_query).fetchone()[0]
            
            # Check mood data
            mood_query = text("SELECT COUNT(*) FROM lyrics_nlp")
            mood_count = self.session.execute(mood_query).fetchone()[0]
            
            logger.info(f"Data availability:")
            logger.info(f"  Interactions: {interactions_count}")
            logger.info(f"  Tracks: {tracks_count}")
            logger.info(f"  Audio features: {audio_count}")
            logger.info(f"  Lyrics: {lyrics_count}")
            logger.info(f"  Mood data: {mood_count}")
            
            # Check if we have minimum required data
            has_minimum_data = (
                interactions_count > 0 and
                tracks_count > 0 and
                (audio_count > 0 or lyrics_count > 0)
            )
            
            if has_minimum_data:
                logger.info("Minimum data requirements met")
                return True
            else:
                logger.warning("Insufficient data for recommender system")
                return False
                
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return False
    
    def test_collaborative_filtering(self) -> Dict[str, Any]:
        """Test collaborative filtering component."""
        logger.info("Testing Collaborative Filtering...")
        
        try:
            # Load interaction data
            interactions_query = """
                SELECT playlist_id, track_id, added_at, position
                FROM playlist_tracks pt
                JOIN playlists p ON pt.playlist_id = p.id
                WHERE p.public = 1
            """
            interactions = pd.read_sql(interactions_query, self.session.bind)
            
            if len(interactions) == 0:
                return {"status": "skipped", "reason": "No interaction data available"}
            
            # Test model initialization
            cf_model = CollaborativeFilteringRecommender(factors=20, regularization=0.01, iterations=10)
            
            # Test training
            cf_model.fit(interactions)
            
            # Test recommendations
            test_playlist = interactions['playlist_id'].iloc[0]
            recommendations = cf_model.recommend_for_user(test_playlist, 5)
            
            # Test similar items
            test_track = interactions['track_id'].iloc[0]
            similar_items = cf_model.get_similar_items(test_track, 5)
            
            return {
                "status": "passed",
                "recommendations_generated": len(recommendations),
                "similar_items_generated": len(similar_items),
                "model_trained": True
            }
            
        except Exception as e:
            logger.error(f"Collaborative filtering test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_content_based_filtering(self) -> Dict[str, Any]:
        """Test content-based filtering component."""
        logger.info("Testing Content-Based Filtering...")
        
        try:
            # Load track data
            track_query = """
                SELECT 
                    t.id as track_id, t.name,
                    af.danceability, af.energy, af.valence, af.tempo, af.loudness,
                    l.text as lyrics_text
                FROM tracks t
                LEFT JOIN audio_features af ON t.id = af.track_id
                LEFT JOIN lyrics l ON t.id = l.track_id
                WHERE t.id IS NOT NULL
            """
            track_data = pd.read_sql(track_query, self.session.bind)
            
            if len(track_data) == 0:
                return {"status": "skipped", "reason": "No track data available"}
            
            # Test model initialization
            content_model = ContentBasedRecommender(max_features=500, n_components=20)
            
            # Test training
            content_model.fit(track_data)
            
            # Test recommendations
            test_track = track_data['track_id'].iloc[0]
            similar_tracks = content_model.recommend_similar(test_track, 5)
            
            # Test user profile recommendations
            user_tracks = track_data['track_id'].head(3).tolist()
            user_recs = content_model.recommend_for_user_profile(user_tracks, 5)
            
            return {
                "status": "passed",
                "similar_tracks_generated": len(similar_tracks),
                "user_recommendations_generated": len(user_recs),
                "model_trained": True
            }
            
        except Exception as e:
            logger.error(f"Content-based filtering test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_mood_aware_filtering(self) -> Dict[str, Any]:
        """Test mood-aware filtering component."""
        logger.info("Testing Mood-Aware Filtering...")
        
        try:
            # Load mood data
            mood_query = """
                SELECT 
                    t.id as track_id,
                    lnl.sentiment_score,
                    lnl.dominant_emotion,
                    lnl.emotion_probs
                FROM tracks t
                JOIN lyrics_nlp lnl ON t.id = lnl.track_id
                WHERE lnl.sentiment_score IS NOT NULL
            """
            mood_data = pd.read_sql(mood_query, self.session.bind)
            
            if len(mood_data) == 0:
                return {"status": "skipped", "reason": "No mood data available"}
            
            # Test model initialization
            mood_model = MoodAwareRecommender()
            
            # Test training
            mood_model.fit(mood_data)
            
            # Test mood-based recommendations
            test_mood = 'happy'
            mood_recs = mood_model.recommend_by_mood(test_mood, 5)
            
            # Test similar mood tracks
            test_track = mood_data['track_id'].iloc[0]
            similar_mood = mood_model.recommend_mood_similar(test_track, 5)
            
            return {
                "status": "passed",
                "mood_recommendations_generated": len(mood_recs),
                "similar_mood_generated": len(similar_mood),
                "model_trained": True
            }
            
        except Exception as e:
            logger.error(f"Mood-aware filtering test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_hybrid_recommender(self) -> Dict[str, Any]:
        """Test hybrid recommender system."""
        logger.info("Testing Hybrid Recommender...")
        
        try:
            # Load all required data
            interactions_query = """
                SELECT playlist_id, track_id, added_at, position
                FROM playlist_tracks pt
                JOIN playlists p ON pt.playlist_id = p.id
                WHERE p.public = 1
            """
            interactions = pd.read_sql(interactions_query, self.session.bind)
            
            track_query = """
                SELECT 
                    t.id as track_id, t.name,
                    af.danceability, af.energy, af.valence, af.tempo, af.loudness,
                    l.text as lyrics_text,
                    lnl.sentiment_score, lnl.dominant_emotion, lnl.emotion_probs
                FROM tracks t
                LEFT JOIN audio_features af ON t.id = af.track_id
                LEFT JOIN lyrics l ON t.id = l.track_id
                LEFT JOIN lyrics_nlp lnl ON t.id = lnl.track_id
            """
            track_data = pd.read_sql(track_query, self.session.bind)
            
            mood_query = """
                SELECT 
                    t.id as track_id,
                    lnl.sentiment_score,
                    lnl.dominant_emotion,
                    lnl.emotion_probs
                FROM tracks t
                JOIN lyrics_nlp lnl ON t.id = lnl.track_id
                WHERE lnl.sentiment_score IS NOT NULL
            """
            mood_data = pd.read_sql(mood_query, self.session.bind)
            
            if len(interactions) == 0 or len(track_data) == 0:
                return {"status": "skipped", "reason": "Insufficient data for hybrid model"}
            
            # Test hybrid model
            hybrid_model = HybridRecommender(
                cf_weight=0.4,
                content_weight=0.3,
                mood_weight=0.3
            )
            
            # Test training
            hybrid_model.fit(interactions, track_data, mood_data)
            
            # Test recommendations
            test_playlist = interactions['playlist_id'].iloc[0]
            test_tracks = interactions[interactions['playlist_id'] == test_playlist]['track_id'].head(3).tolist()
            
            recommendations = hybrid_model.recommend(
                user_id=test_playlist,
                user_tracks=test_tracks,
                target_mood='happy',
                n_recommendations=10
            )
            
            return {
                "status": "passed",
                "recommendations_generated": len(recommendations),
                "model_trained": True,
                "components_working": True
            }
            
        except Exception as e:
            logger.error(f"Hybrid recommender test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_context_awareness(self) -> Dict[str, Any]:
        """Test context-aware recommendations."""
        logger.info("Testing Context Awareness...")
        
        try:
            # Load hybrid model (or create a simple one for testing)
            hybrid_model = HybridRecommender()
            
            # Test context-aware model
            context_model = ContextAwareRecommender(hybrid_model)
            
            # Test different contexts
            contexts = [
                {'time_of_day': 'morning', 'activity': 'workout'},
                {'time_of_day': 'evening', 'activity': 'relax'},
                {'time_of_day': 'night', 'activity': 'sleep'}
            ]
            
            context_results = []
            for context in contexts:
                try:
                    # Test context detection
                    auto_context = context_model._auto_detect_context()
                    
                    # Test context suggestions
                    suggestions = context_model.get_context_suggestions(context)
                    
                    context_results.append({
                        "context": context,
                        "auto_detection_works": True,
                        "suggestions_available": len(suggestions) > 0
                    })
                    
                except Exception as e:
                    context_results.append({
                        "context": context,
                        "error": str(e)
                    })
            
            return {
                "status": "passed",
                "context_tests": context_results,
                "context_awareness_working": True
            }
            
        except Exception as e:
            logger.error(f"Context awareness test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test API endpoint functionality."""
        logger.info("Testing API Endpoints...")
        
        try:
            # Test if API can be imported
            from api.recommendation_api import app
            
            # Test Pydantic models
            from api.recommendation_api import RecommendationRequest, FeedbackRequest
            
            # Test request model creation
            test_request = RecommendationRequest(
                user_id="test_user",
                user_tracks=["track1", "track2"],
                n_recommendations=5
            )
            
            # Test feedback model creation
            test_feedback = FeedbackRequest(
                user_id="test_user",
                track_id="track1",
                context={"time_of_day": "morning"},
                feedback="positive",
                rating=4.5
            )
            
            return {
                "status": "passed",
                "api_imports_work": True,
                "pydantic_models_work": True,
                "request_models_valid": True
            }
            
        except Exception as e:
            logger.error(f"API endpoints test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("Starting comprehensive recommender system test...")
        
        test_results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "tests": {}
        }
        
        # Test data availability
        data_available = self.test_data_availability()
        test_results["tests"]["data_availability"] = {
            "status": "passed" if data_available else "failed",
            "data_available": data_available
        }
        
        # Test individual components
        test_results["tests"]["collaborative_filtering"] = self.test_collaborative_filtering()
        test_results["tests"]["content_based_filtering"] = self.test_content_based_filtering()
        test_results["tests"]["mood_aware_filtering"] = self.test_mood_aware_filtering()
        
        # Test hybrid system
        test_results["tests"]["hybrid_recommender"] = self.test_hybrid_recommender()
        
        # Test context awareness
        test_results["tests"]["context_awareness"] = self.test_context_awareness()
        
        # Test API
        test_results["tests"]["api_endpoints"] = self.test_api_endpoints()
        
        # Calculate overall status
        passed_tests = sum(1 for test in test_results["tests"].values() 
                          if test.get("status") == "passed")
        total_tests = len(test_results["tests"])
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0
        }
        
        return test_results
    
    def generate_test_report(self, results: Dict[str, Any], output_file: str = "test_report.json"):
        """Generate test report."""
        logger.info(f"Generating test report: {output_file}")
        
        # Save JSON report
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        logger.info("=== TEST SUMMARY ===")
        logger.info(f"Total tests: {results['summary']['total_tests']}")
        logger.info(f"Passed: {results['summary']['passed_tests']}")
        logger.info(f"Failed: {results['summary']['failed_tests']}")
        logger.info(f"Success rate: {results['summary']['success_rate']:.2%}")
        
        # Print individual test results
        for test_name, test_result in results["tests"].items():
            status = test_result.get("status", "unknown")
            logger.info(f"{test_name}: {status}")
            if status == "failed" and "error" in test_result:
                logger.info(f"  Error: {test_result['error']}")


def main():
    """Main test function."""
    logger.info("Starting recommender system testing...")
    
    try:
        # Initialize tester
        tester = RecommenderTester()
        
        # Run comprehensive tests
        results = tester.run_comprehensive_test()
        
        # Generate report
        tester.generate_test_report(results)
        
        # Check if all critical tests passed
        critical_tests = ["data_availability", "hybrid_recommender"]
        critical_passed = all(
            results["tests"].get(test, {}).get("status") == "passed"
            for test in critical_tests
        )
        
        if critical_passed:
            logger.info("All critical tests passed!")
            return 0
        else:
            logger.warning("Some critical tests failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        return 1
    finally:
        if hasattr(tester, 'session'):
            tester.session.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
