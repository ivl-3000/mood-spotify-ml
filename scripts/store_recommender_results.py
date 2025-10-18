"""
Store Recommender Results in SQL Database
Saves evaluation metrics, recommendations, and user interactions to database.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from db.engine import get_session, init_db
from db.recommender_results import RecommenderResultsManager
from ml.recommender_system import HybridRecommender, RecommenderEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('store_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ResultsStorageManager:
    """Manages storage of recommender system results."""
    
    def __init__(self):
        self.session = get_session()
        self.results_manager = RecommenderResultsManager(self.session)
        self.evaluator = RecommenderEvaluator()
    
    def store_evaluation_results(self, model_name: str, results: Dict[str, Any]) -> None:
        """Store evaluation results for a model."""
        logger.info(f"Storing evaluation results for {model_name}")
        
        try:
            # Store main evaluation metrics
            self.results_manager.store_evaluation_results(
                model_name=model_name,
                results=results,
                model_config={'timestamp': datetime.now().isoformat()}
            )
            
            # Store performance metrics
            if 'performance' in results:
                self.results_manager.store_model_performance(
                    model_name=model_name,
                    metrics=results['performance'],
                    evaluation_period='single',
                    period_start=datetime.now(),
                    period_end=datetime.now(),
                    num_samples=results.get('num_test_cases', 0)
                )
            
            logger.info(f"Successfully stored evaluation results for {model_name}")
            
        except Exception as e:
            logger.error(f"Error storing evaluation results: {e}")
            raise
    
    def store_recommendation_batch(self, recommendations_data: List[Dict[str, Any]]) -> None:
        """Store a batch of recommendation results."""
        logger.info(f"Storing {len(recommendations_data)} recommendation batches")
        
        try:
            for rec_data in recommendations_data:
                self.results_manager.store_recommendations(
                    user_id=rec_data['user_id'],
                    model_name=rec_data['model_name'],
                    recommendations=rec_data['recommendations'],
                    context=rec_data.get('context')
                )
            
            logger.info("Successfully stored recommendation batch")
            
        except Exception as e:
            logger.error(f"Error storing recommendations: {e}")
            raise
    
    def store_user_interactions(self, interactions_data: List[Dict[str, Any]]) -> None:
        """Store user interaction data."""
        logger.info(f"Storing {len(interactions_data)} user interactions")
        
        try:
            for interaction in interactions_data:
                self.results_manager.store_user_interaction(
                    user_id=interaction['user_id'],
                    track_id=interaction['track_id'],
                    interaction_type=interaction['interaction_type'],
                    rating=interaction.get('rating'),
                    context=interaction.get('context'),
                    session_id=interaction.get('session_id')
                )
            
            logger.info("Successfully stored user interactions")
            
        except Exception as e:
            logger.error(f"Error storing user interactions: {e}")
            raise
    
    def store_ab_test_results(self, ab_test_data: Dict[str, Any]) -> None:
        """Store A/B test results."""
        logger.info("Storing A/B test results")
        
        try:
            self.results_manager.store_ab_test_result(
                test_name=ab_test_data['test_name'],
                variant_a=ab_test_data['variant_a'],
                variant_b=ab_test_data['variant_b'],
                metric_name=ab_test_data['metric_name'],
                variant_a_value=ab_test_data['variant_a_value'],
                variant_b_value=ab_test_data['variant_b_value'],
                statistical_significance=ab_test_data.get('statistical_significance'),
                confidence_interval=ab_test_data.get('confidence_interval'),
                test_duration_days=ab_test_data.get('test_duration_days', 0),
                num_users_a=ab_test_data.get('num_users_a', 0),
                num_users_b=ab_test_data.get('num_users_b', 0)
            )
            
            logger.info("Successfully stored A/B test results")
            
        except Exception as e:
            logger.error(f"Error storing A/B test results: {e}")
            raise
    
    def generate_and_store_performance_report(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """Generate and store performance report."""
        logger.info(f"Generating performance report for {model_name}")
        
        try:
            report = self.results_manager.generate_performance_report(model_name, days)
            
            # Store report as JSON
            report_data = {
                'model_name': model_name,
                'report_date': datetime.now().isoformat(),
                'period_days': days,
                'metrics': report
            }
            
            # Save to file
            output_dir = Path("results/reports")
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / f"{model_name}_performance_report.json", 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Performance report saved for {model_name}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            raise
    
    def load_simulation_results(self, results_file: str) -> Dict[str, Any]:
        """Load simulation results from file."""
        logger.info(f"Loading simulation results from {results_file}")
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading simulation results: {e}")
            raise
    
    def store_simulation_results(self, simulation_results: Dict[str, Any]) -> None:
        """Store simulation results in database."""
        logger.info("Storing simulation results")
        
        try:
            # Store user journeys as interactions
            journeys = simulation_results.get('journeys', [])
            interactions_data = []
            
            for journey in journeys:
                user_id = journey['user_id']
                session_id = journey['session_id']
                context = journey['context']
                
                # Store user interactions from the journey
                user_interactions = journey.get('user_interactions', {})
                
                # Store liked tracks
                for track_id in user_interactions.get('liked_tracks', []):
                    interactions_data.append({
                        'user_id': user_id,
                        'track_id': track_id,
                        'interaction_type': 'like',
                        'rating': 4.0,  # Default rating for liked tracks
                        'context': context,
                        'session_id': session_id
                    })
                
                # Store skipped tracks
                for track_id in user_interactions.get('skipped_tracks', []):
                    interactions_data.append({
                        'user_id': user_id,
                        'track_id': track_id,
                        'interaction_type': 'skip',
                        'rating': 1.0,  # Low rating for skipped tracks
                        'context': context,
                        'session_id': session_id
                    })
                
                # Store playlist additions
                for track_id in user_interactions.get('added_to_playlist', []):
                    interactions_data.append({
                        'user_id': user_id,
                        'track_id': track_id,
                        'interaction_type': 'add_to_playlist',
                        'rating': 5.0,  # High rating for playlist additions
                        'context': context,
                        'session_id': session_id
                    })
            
            # Store interactions in batches
            batch_size = 100
            for i in range(0, len(interactions_data), batch_size):
                batch = interactions_data[i:i + batch_size]
                self.store_user_interactions(batch)
            
            # Store playlists as recommendation results
            playlists = simulation_results.get('playlists', [])
            recommendations_data = []
            
            for playlist in playlists:
                user_id = playlist['owner_id']
                tracks = playlist['tracks']
                
                # Store each track as a recommendation
                for rank, track_info in enumerate(tracks, 1):
                    recommendations_data.append({
                        'user_id': user_id,
                        'model_name': 'simulation_hybrid',
                        'recommendations': [(track_info['track_id'], track_info['score'])],
                        'context': {
                            'mood': playlist['mood'],
                            'activity': playlist['activity'],
                            'playlist_id': playlist['playlist_id']
                        }
                    })
            
            # Store recommendations in batches
            for i in range(0, len(recommendations_data), batch_size):
                batch = recommendations_data[i:i + batch_size]
                self.store_recommendation_batch(batch)
            
            logger.info("Successfully stored simulation results")
            
        except Exception as e:
            logger.error(f"Error storing simulation results: {e}")
            raise
    
    def create_sample_data(self) -> None:
        """Create sample data for demonstration."""
        logger.info("Creating sample data")
        
        try:
            # Sample evaluation results
            sample_evaluation = {
                'k=5': {
                    'precision': 0.234,
                    'recall': 0.156,
                    'ndcg': 0.189,
                    'coverage': 0.445
                },
                'k=10': {
                    'precision': 0.267,
                    'recall': 0.178,
                    'ndcg': 0.201,
                    'coverage': 0.523
                },
                'k=20': {
                    'precision': 0.289,
                    'recall': 0.201,
                    'ndcg': 0.223,
                    'coverage': 0.612
                }
            }
            
            self.store_evaluation_results('hybrid_model', sample_evaluation)
            
            # Sample recommendation results
            sample_recommendations = [
                {
                    'user_id': 'user_001',
                    'model_name': 'hybrid_model',
                    'recommendations': [
                        ('track_001', 0.95),
                        ('track_002', 0.87),
                        ('track_003', 0.82)
                    ],
                    'context': {'mood': 'happy', 'activity': 'workout'}
                }
            ]
            
            self.store_recommendation_batch(sample_recommendations)
            
            # Sample user interactions
            sample_interactions = [
                {
                    'user_id': 'user_001',
                    'track_id': 'track_001',
                    'interaction_type': 'like',
                    'rating': 4.5,
                    'context': {'mood': 'happy'},
                    'session_id': 'session_001'
                }
            ]
            
            self.store_user_interactions(sample_interactions)
            
            # Sample A/B test results
            sample_ab_test = {
                'test_name': 'hybrid_vs_baseline',
                'variant_a': 'hybrid_model',
                'variant_b': 'popularity_baseline',
                'metric_name': 'precision@10',
                'variant_a_value': 0.267,
                'variant_b_value': 0.123,
                'statistical_significance': 0.95,
                'confidence_interval': {'lower': 0.12, 'upper': 0.16},
                'test_duration_days': 7,
                'num_users_a': 100,
                'num_users_b': 100
            }
            
            self.store_ab_test_results(sample_ab_test)
            
            logger.info("Sample data created successfully")
            
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            raise
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all stored results."""
        logger.info("Generating summary report")
        
        try:
            # Get all evaluation results
            evaluations = self.results_manager.get_evaluation_results()
            
            # Get all recommendations
            recommendations = self.session.query(RecommendationResult).count()
            
            # Get all interactions
            interactions = self.session.query(UserInteraction).count()
            
            # Get all A/B tests
            ab_tests = self.results_manager.get_ab_test_results()
            
            summary = {
                'total_evaluations': len(evaluations),
                'total_recommendations': recommendations,
                'total_interactions': interactions,
                'total_ab_tests': len(ab_tests),
                'models_evaluated': list(set([e.model_name for e in evaluations])),
                'evaluation_metrics': list(set([e.evaluation_type for e in evaluations])),
                'date_range': {
                    'earliest': min([e.evaluation_date for e in evaluations]) if evaluations else None,
                    'latest': max([e.evaluation_date for e in evaluations]) if evaluations else None
                }
            }
            
            # Save summary report
            output_dir = Path("results/reports")
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "summary_report.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Summary report generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            raise


def main():
    """Main function to store recommender results."""
    logger.info("Starting recommender results storage...")
    
    try:
        # Initialize database
        init_db()
        
        # Initialize storage manager
        storage_manager = ResultsStorageManager()
        
        # Create sample data
        storage_manager.create_sample_data()
        
        # Generate summary report
        summary = storage_manager.generate_summary_report()
        
        # Print summary
        logger.info("=== STORAGE SUMMARY ===")
        logger.info(f"Total evaluations: {summary['total_evaluations']}")
        logger.info(f"Total recommendations: {summary['total_recommendations']}")
        logger.info(f"Total interactions: {summary['total_interactions']}")
        logger.info(f"Total A/B tests: {summary['total_ab_tests']}")
        logger.info(f"Models evaluated: {summary['models_evaluated']}")
        
        # Check if simulation results exist
        simulation_file = Path("results/simulation/simulation_summary.json")
        if simulation_file.exists():
            logger.info("Loading and storing simulation results...")
            simulation_results = storage_manager.load_simulation_results(str(simulation_file))
            storage_manager.store_simulation_results(simulation_results)
        
        logger.info("Recommender results storage completed successfully!")
        
    except Exception as e:
        logger.error(f"Storage failed: {e}")
        raise
    finally:
        storage_manager.session.close()


if __name__ == "__main__":
    main()
