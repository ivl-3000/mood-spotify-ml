"""
Database storage for recommender system results
Stores evaluation metrics, recommendations, and user interactions.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, DateTime, JSON, Text, ForeignKey

from db.models import Base

logger = logging.getLogger(__name__)


class RecommenderEvaluation(Base):
    """Storage for recommender evaluation results."""
    __tablename__ = "recommender_evaluations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String)
    evaluation_type: Mapped[str] = mapped_column(String)  # 'precision', 'recall', 'ndcg', etc.
    k_value: Mapped[int] = mapped_column(Integer)
    metric_value: Mapped[float] = mapped_column(Float)
    std_deviation: Mapped[Optional[float]] = mapped_column(Float)
    num_test_cases: Mapped[int] = mapped_column(Integer)
    evaluation_date: Mapped[datetime] = mapped_column(DateTime)
    model_config: Mapped[Optional[Dict]] = mapped_column(JSON)
    notes: Mapped[Optional[str]] = mapped_column(Text)


class RecommendationResult(Base):
    """Storage for individual recommendation results."""
    __tablename__ = "recommendation_results"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String)
    track_id: Mapped[str] = mapped_column(String, ForeignKey("tracks.id"))
    recommendation_score: Mapped[float] = mapped_column(Float)
    rank_position: Mapped[int] = mapped_column(Integer)
    context: Mapped[Optional[Dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    
    # Relationships
    track = relationship("Track", back_populates="recommendations")


class UserInteraction(Base):
    """Storage for user interaction feedback."""
    __tablename__ = "user_interactions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String)
    track_id: Mapped[str] = mapped_column(String, ForeignKey("tracks.id"))
    interaction_type: Mapped[str] = mapped_column(String)  # 'like', 'skip', 'play', 'add_to_playlist'
    rating: Mapped[Optional[float]] = mapped_column(Float)
    context: Mapped[Optional[Dict]] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    session_id: Mapped[Optional[str]] = mapped_column(String)
    
    # Relationships
    track = relationship("Track", back_populates="interactions")


class ModelPerformance(Base):
    """Storage for model performance metrics."""
    __tablename__ = "model_performance"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String)
    metric_name: Mapped[str] = mapped_column(String)
    metric_value: Mapped[float] = mapped_column(Float)
    evaluation_period: Mapped[str] = mapped_column(String)  # 'daily', 'weekly', 'monthly'
    period_start: Mapped[datetime] = mapped_column(DateTime)
    period_end: Mapped[datetime] = mapped_column(DateTime)
    num_samples: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class A/BTestResult(Base):
    """Storage for A/B testing results."""
    __tablename__ = "ab_test_results"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_name: Mapped[str] = mapped_column(String)
    variant_a: Mapped[str] = mapped_column(String)
    variant_b: Mapped[str] = mapped_column(String)
    metric_name: Mapped[str] = mapped_column(String)
    variant_a_value: Mapped[float] = mapped_column(Float)
    variant_b_value: Mapped[float] = mapped_column(Float)
    statistical_significance: Mapped[Optional[float]] = mapped_column(Float)
    confidence_interval: Mapped[Optional[Dict]] = mapped_column(JSON)
    test_duration_days: Mapped[int] = mapped_column(Integer)
    num_users_a: Mapped[int] = mapped_column(Integer)
    num_users_b: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class RecommenderResultsManager:
    """Manager for storing and retrieving recommender results."""
    
    def __init__(self, session):
        self.session = session
    
    def store_evaluation_results(self, model_name: str, results: Dict[str, Any], 
                                model_config: Optional[Dict] = None) -> None:
        """Store evaluation results in database."""
        logger.info(f"Storing evaluation results for {model_name}")
        
        evaluation_date = datetime.now()
        
        for metric_name, metric_data in results.items():
            if isinstance(metric_data, dict) and 'k=' in metric_name:
                k_value = int(metric_name.split('=')[1])
                
                # Store individual metrics
                for metric_type in ['precision', 'recall', 'ndcg', 'coverage']:
                    if metric_type in metric_data:
                        evaluation = RecommenderEvaluation(
                            model_name=model_name,
                            evaluation_type=metric_type,
                            k_value=k_value,
                            metric_value=metric_data[metric_type],
                            std_deviation=metric_data.get(f'std_{metric_type}'),
                            num_test_cases=metric_data.get('num_test_cases', 0),
                            evaluation_date=evaluation_date,
                            model_config=model_config
                        )
                        self.session.add(evaluation)
        
        self.session.commit()
        logger.info(f"Stored evaluation results for {model_name}")
    
    def store_recommendations(self, user_id: str, model_name: str, 
                             recommendations: List[Tuple[str, float]], 
                             context: Optional[Dict] = None) -> None:
        """Store recommendation results."""
        logger.info(f"Storing {len(recommendations)} recommendations for user {user_id}")
        
        created_at = datetime.now()
        
        for rank, (track_id, score) in enumerate(recommendations, 1):
            result = RecommendationResult(
                user_id=user_id,
                model_name=model_name,
                track_id=track_id,
                recommendation_score=score,
                rank_position=rank,
                context=context,
                created_at=created_at
            )
            self.session.add(result)
        
        self.session.commit()
        logger.info(f"Stored recommendations for user {user_id}")
    
    def store_user_interaction(self, user_id: str, track_id: str, 
                              interaction_type: str, rating: Optional[float] = None,
                              context: Optional[Dict] = None, session_id: Optional[str] = None) -> None:
        """Store user interaction feedback."""
        interaction = UserInteraction(
            user_id=user_id,
            track_id=track_id,
            interaction_type=interaction_type,
            rating=rating,
            context=context,
            timestamp=datetime.now(),
            session_id=session_id
        )
        self.session.add(interaction)
        self.session.commit()
    
    def store_model_performance(self, model_name: str, metrics: Dict[str, float],
                               evaluation_period: str, period_start: datetime,
                               period_end: datetime, num_samples: int,
                               model_version: Optional[str] = None) -> None:
        """Store model performance metrics."""
        for metric_name, metric_value in metrics.items():
            performance = ModelPerformance(
                model_name=model_name,
                metric_name=metric_name,
                metric_value=metric_value,
                evaluation_period=evaluation_period,
                period_start=period_start,
                period_end=period_end,
                num_samples=num_samples,
                model_version=model_version,
                created_at=datetime.now()
            )
            self.session.add(performance)
        
        self.session.commit()
    
    def store_ab_test_result(self, test_name: str, variant_a: str, variant_b: str,
                            metric_name: str, variant_a_value: float, variant_b_value: float,
                            statistical_significance: Optional[float] = None,
                            confidence_interval: Optional[Dict] = None,
                            test_duration_days: int = 0,
                            num_users_a: int = 0, num_users_b: int = 0) -> None:
        """Store A/B test results."""
        ab_result = A/BTestResult(
            test_name=test_name,
            variant_a=variant_a,
            variant_b=variant_b,
            metric_name=metric_name,
            variant_a_value=variant_a_value,
            variant_b_value=variant_b_value,
            statistical_significance=statistical_significance,
            confidence_interval=confidence_interval,
            test_duration_days=test_duration_days,
            num_users_a=num_users_a,
            num_users_b=num_users_b,
            created_at=datetime.now()
        )
        self.session.add(ab_result)
        self.session.commit()
    
    def get_evaluation_results(self, model_name: Optional[str] = None, 
                             metric_type: Optional[str] = None,
                             k_value: Optional[int] = None) -> List[RecommenderEvaluation]:
        """Retrieve evaluation results."""
        query = self.session.query(RecommenderEvaluation)
        
        if model_name:
            query = query.filter(RecommenderEvaluation.model_name == model_name)
        if metric_type:
            query = query.filter(RecommenderEvaluation.evaluation_type == metric_type)
        if k_value:
            query = query.filter(RecommenderEvaluation.k_value == k_value)
        
        return query.order_by(RecommenderEvaluation.evaluation_date.desc()).all()
    
    def get_recommendation_history(self, user_id: str, model_name: Optional[str] = None,
                                 limit: int = 100) -> List[RecommendationResult]:
        """Get recommendation history for a user."""
        query = self.session.query(RecommendationResult).filter(
            RecommendationResult.user_id == user_id
        )
        
        if model_name:
            query = query.filter(RecommendationResult.model_name == model_name)
        
        return query.order_by(RecommendationResult.created_at.desc()).limit(limit).all()
    
    def get_user_interactions(self, user_id: str, interaction_type: Optional[str] = None,
                             limit: int = 100) -> List[UserInteraction]:
        """Get user interaction history."""
        query = self.session.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        )
        
        if interaction_type:
            query = query.filter(UserInteraction.interaction_type == interaction_type)
        
        return query.order_by(UserInteraction.timestamp.desc()).limit(limit).all()
    
    def get_model_performance_trends(self, model_name: str, metric_name: str,
                                    evaluation_period: str = 'daily') -> List[ModelPerformance]:
        """Get performance trends for a model."""
        return self.session.query(ModelPerformance).filter(
            ModelPerformance.model_name == model_name,
            ModelPerformance.metric_name == metric_name,
            ModelPerformance.evaluation_period == evaluation_period
        ).order_by(ModelPerformance.period_start).all()
    
    def get_ab_test_results(self, test_name: Optional[str] = None) -> List[A/BTestResult]:
        """Get A/B test results."""
        query = self.session.query(A/BTestResult)
        
        if test_name:
            query = query.filter(A/BTestResult.test_name == test_name)
        
        return query.order_by(A/BTestResult.created_at.desc()).all()
    
    def generate_performance_report(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """Generate performance report for a model."""
        from datetime import timedelta
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Get recent evaluations
        evaluations = self.session.query(RecommenderEvaluation).filter(
            RecommenderEvaluation.model_name == model_name,
            RecommenderEvaluation.evaluation_date >= start_date
        ).all()
        
        # Get performance metrics
        performance = self.session.query(ModelPerformance).filter(
            ModelPerformance.model_name == model_name,
            ModelPerformance.period_start >= start_date
        ).all()
        
        # Get recommendation counts
        recommendations = self.session.query(RecommendationResult).filter(
            RecommendationResult.model_name == model_name,
            RecommendationResult.created_at >= start_date
        ).count()
        
        # Get user interactions
        interactions = self.session.query(UserInteraction).filter(
            UserInteraction.timestamp >= start_date
        ).count()
        
        return {
            'model_name': model_name,
            'period_days': days,
            'total_evaluations': len(evaluations),
            'total_recommendations': recommendations,
            'total_interactions': interactions,
            'performance_metrics': [p.metric_value for p in performance],
            'evaluation_metrics': [e.metric_value for e in evaluations]
        }
