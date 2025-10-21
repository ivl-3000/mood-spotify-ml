"""
Power BI Visualization Data Preparation
Creates datasets and visualizations for recommender system evaluation metrics.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from db.engine import get_session
from db.recommender_results import RecommenderResultsManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('powerbi_visualization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PowerBIDataExporter:
    """Exports data for Power BI visualizations."""
    
    def __init__(self):
        self.session = get_session()
        self.results_manager = RecommenderResultsManager(self.session)
    
    def export_evaluation_metrics(self, output_dir: Path) -> None:
        """Export evaluation metrics for Power BI."""
        logger.info("Exporting evaluation metrics...")
        
        try:
            # Get all evaluation results
            evaluations = self.results_manager.get_evaluation_results()
            
            if not evaluations:
                logger.warning("No evaluation results found")
                return
            
            # Convert to DataFrame
            eval_data = []
            for eval_result in evaluations:
                eval_data.append({
                    'Model_Name': eval_result.model_name,
                    'Metric_Type': eval_result.evaluation_type,
                    'K_Value': eval_result.k_value,
                    'Metric_Value': eval_result.metric_value,
                    'Std_Deviation': eval_result.std_deviation,
                    'Num_Test_Cases': eval_result.num_test_cases,
                    'Evaluation_Date': eval_result.evaluation_date,
                    'Model_Config': json.dumps(eval_result.model_config) if eval_result.model_config else None
                })
            
            eval_df = pd.DataFrame(eval_data)
            eval_df.to_csv(output_dir / "evaluation_metrics.csv", index=False)
            
            # Create pivot table for metrics comparison
            pivot_df = eval_df.pivot_table(
                index=['Model_Name', 'K_Value'],
                columns='Metric_Type',
                values='Metric_Value',
                aggfunc='mean'
            ).reset_index()
            
            pivot_df.to_csv(output_dir / "evaluation_metrics_pivot.csv", index=False)
            
            logger.info(f"Exported {len(eval_data)} evaluation metrics")
            
        except Exception as e:
            logger.error(f"Error exporting evaluation metrics: {e}")
            raise
    
    def export_recommendation_data(self, output_dir: Path) -> None:
        """Export recommendation data for Power BI."""
        logger.info("Exporting recommendation data...")
        
        try:
            # Get recommendation results
            query = """
                SELECT 
                    rr.user_id,
                    rr.model_name,
                    rr.track_id,
                    rr.recommendation_score,
                    rr.rank_position,
                    rr.context,
                    rr.created_at,
                    t.name as track_name,
                    t.popularity as track_popularity
                FROM recommendation_results rr
                LEFT JOIN tracks t ON rr.track_id = t.id
                ORDER BY rr.created_at DESC
            """
            
            rec_df = pd.read_sql(query, self.session.bind)
            
            if len(rec_df) > 0:
                # Parse context JSON
                rec_df['context_mood'] = rec_df['context'].apply(
                    lambda x: json.loads(x).get('mood') if x else None
                )
                rec_df['context_activity'] = rec_df['context'].apply(
                    lambda x: json.loads(x).get('activity') if x else None
                )
                
                rec_df.to_csv(output_dir / "recommendation_data.csv", index=False)
                
                # Create aggregated views
                self._create_recommendation_aggregates(rec_df, output_dir)
            
            logger.info(f"Exported {len(rec_df)} recommendation records")
            
        except Exception as e:
            logger.error(f"Error exporting recommendation data: {e}")
            raise
    
    def export_user_interactions(self, output_dir: Path) -> None:
        """Export user interaction data for Power BI."""
        logger.info("Exporting user interaction data...")
        
        try:
            # Get user interactions
            query = """
                SELECT 
                    ui.user_id,
                    ui.track_id,
                    ui.interaction_type,
                    ui.rating,
                    ui.context,
                    ui.timestamp,
                    ui.session_id,
                    t.name as track_name,
                    t.popularity as track_popularity
                FROM user_interactions ui
                LEFT JOIN tracks t ON ui.track_id = t.id
                ORDER BY ui.timestamp DESC
            """
            
            interactions_df = pd.read_sql(query, self.session.bind)
            
            if len(interactions_df) > 0:
                # Parse context JSON
                interactions_df['context_mood'] = interactions_df['context'].apply(
                    lambda x: json.loads(x).get('mood') if x else None
                )
                interactions_df['context_activity'] = interactions_df['context'].apply(
                    lambda x: json.loads(x).get('activity') if x else None
                )
                
                # Add time-based features
                interactions_df['hour'] = pd.to_datetime(interactions_df['timestamp']).dt.hour
                interactions_df['day_of_week'] = pd.to_datetime(interactions_df['timestamp']).dt.day_name()
                interactions_df['date'] = pd.to_datetime(interactions_df['timestamp']).dt.date
                
                interactions_df.to_csv(output_dir / "user_interactions.csv", index=False)
                
                # Create interaction summaries
                self._create_interaction_summaries(interactions_df, output_dir)
            
            logger.info(f"Exported {len(interactions_df)} interaction records")
            
        except Exception as e:
            logger.error(f"Error exporting user interactions: {e}")
            raise
    
    def export_model_performance(self, output_dir: Path) -> None:
        """Export model performance data for Power BI."""
        logger.info("Exporting model performance data...")
        
        try:
            # Get model performance data
            query = """
                SELECT 
                    mp.model_name,
                    mp.metric_name,
                    mp.metric_value,
                    mp.evaluation_period,
                    mp.period_start,
                    mp.period_end,
                    mp.num_samples,
                    mp.model_version,
                    mp.created_at
                FROM model_performance mp
                ORDER BY mp.period_start DESC
            """
            
            perf_df = pd.read_sql(query, self.session.bind)
            
            if len(perf_df) > 0:
                perf_df.to_csv(output_dir / "model_performance.csv", index=False)
                
                # Create performance trends
                self._create_performance_trends(perf_df, output_dir)
            
            logger.info(f"Exported {len(perf_df)} performance records")
            
        except Exception as e:
            logger.error(f"Error exporting model performance: {e}")
            raise
    
    def export_ab_test_results(self, output_dir: Path) -> None:
        """Export A/B test results for Power BI."""
        logger.info("Exporting A/B test results...")
        
        try:
            # Get A/B test results
            ab_tests = self.results_manager.get_ab_test_results()
            
            if ab_tests:
                ab_data = []
                for test in ab_tests:
                    ab_data.append({
                        'Test_Name': test.test_name,
                        'Variant_A': test.variant_a,
                        'Variant_B': test.variant_b,
                        'Metric_Name': test.metric_name,
                        'Variant_A_Value': test.variant_a_value,
                        'Variant_B_Value': test.variant_b_value,
                        'Statistical_Significance': test.statistical_significance,
                        'Confidence_Interval': json.dumps(test.confidence_interval) if test.confidence_interval else None,
                        'Test_Duration_Days': test.test_duration_days,
                        'Num_Users_A': test.num_users_a,
                        'Num_Users_B': test.num_users_b,
                        'Created_At': test.created_at
                    })
                
                ab_df = pd.DataFrame(ab_data)
                ab_df.to_csv(output_dir / "ab_test_results.csv", index=False)
                
                # Create A/B test analysis
                self._create_ab_test_analysis(ab_df, output_dir)
            
            logger.info(f"Exported {len(ab_tests)} A/B test results")
            
        except Exception as e:
            logger.error(f"Error exporting A/B test results: {e}")
            raise
    
    def _create_recommendation_aggregates(self, rec_df: pd.DataFrame, output_dir: Path) -> None:
        """Create recommendation aggregates for Power BI."""
        logger.info("Creating recommendation aggregates...")
        
        # Model performance by K value
        model_k_performance = rec_df.groupby(['model_name', 'rank_position']).agg({
            'recommendation_score': ['mean', 'std', 'count']
        }).round(4)
        
        model_k_performance.columns = ['avg_score', 'std_score', 'count']
        model_k_performance.to_csv(output_dir / "model_k_performance.csv")
        
        # Context-based performance
        if 'context_mood' in rec_df.columns:
            mood_performance = rec_df.groupby(['model_name', 'context_mood']).agg({
                'recommendation_score': 'mean',
                'rank_position': 'mean'
            }).round(4)
            
            mood_performance.to_csv(output_dir / "mood_performance.csv")
        
        # User engagement by model
        user_engagement = rec_df.groupby('model_name').agg({
            'user_id': 'nunique',
            'track_id': 'nunique',
            'recommendation_score': 'mean'
        }).round(4)
        
        user_engagement.columns = ['unique_users', 'unique_tracks', 'avg_score']
        user_engagement.to_csv(output_dir / "user_engagement.csv")
    
    def _create_interaction_summaries(self, interactions_df: pd.DataFrame, output_dir: Path) -> None:
        """Create interaction summaries for Power BI."""
        logger.info("Creating interaction summaries...")
        
        # Interaction types distribution
        interaction_dist = interactions_df['interaction_type'].value_counts().reset_index()
        interaction_dist.columns = ['interaction_type', 'count']
        interaction_dist.to_csv(output_dir / "interaction_distribution.csv", index=False)
        
        # Hourly interaction patterns
        hourly_patterns = interactions_df.groupby('hour').size().reset_index()
        hourly_patterns.columns = ['hour', 'interaction_count']
        hourly_patterns.to_csv(output_dir / "hourly_patterns.csv", index=False)
        
        # Day of week patterns
        dow_patterns = interactions_df.groupby('day_of_week').size().reset_index()
        dow_patterns.columns = ['day_of_week', 'interaction_count']
        dow_patterns.to_csv(output_dir / "day_of_week_patterns.csv", index=False)
        
        # Rating distribution
        if 'rating' in interactions_df.columns:
            rating_dist = interactions_df['rating'].value_counts().sort_index().reset_index()
            rating_dist.columns = ['rating', 'count']
            rating_dist.to_csv(output_dir / "rating_distribution.csv", index=False)
    
    def _create_performance_trends(self, perf_df: pd.DataFrame, output_dir: Path) -> None:
        """Create performance trends for Power BI."""
        logger.info("Creating performance trends...")
        
        # Daily performance trends
        daily_trends = perf_df.groupby(['model_name', 'metric_name', 'period_start']).agg({
            'metric_value': 'mean'
        }).reset_index()
        
        daily_trends.to_csv(output_dir / "daily_performance_trends.csv", index=False)
        
        # Model comparison
        model_comparison = perf_df.groupby(['model_name', 'metric_name']).agg({
            'metric_value': ['mean', 'std', 'count']
        }).round(4)
        
        model_comparison.columns = ['avg_value', 'std_value', 'count']
        model_comparison.to_csv(output_dir / "model_comparison.csv")
    
    def _create_ab_test_analysis(self, ab_df: pd.DataFrame, output_dir: Path) -> None:
        """Create A/B test analysis for Power BI."""
        logger.info("Creating A/B test analysis...")
        
        # Test results summary
        test_summary = ab_df.groupby('test_name').agg({
            'variant_a_value': 'mean',
            'variant_b_value': 'mean',
            'statistical_significance': 'mean',
            'test_duration_days': 'mean'
        }).round(4)
        
        test_summary.to_csv(output_dir / "ab_test_summary.csv")
        
        # Statistical significance analysis
        significance_analysis = ab_df[['test_name', 'metric_name', 'statistical_significance']].copy()
        significance_analysis['is_significant'] = significance_analysis['statistical_significance'] > 0.95
        significance_analysis.to_csv(output_dir / "significance_analysis.csv", index=False)
    
    def create_visualization_guide(self, output_dir: Path) -> None:
        """Create Power BI visualization guide."""
        logger.info("Creating Power BI visualization guide...")
        
        guide_content = """# Power BI Visualization Guide for Recommender System

## Dataset Overview

### 1. Evaluation Metrics (evaluation_metrics.csv)
- **Purpose**: Compare model performance across different metrics
- **Key Fields**: Model_Name, Metric_Type, K_Value, Metric_Value
- **Visualizations**:
  - Bar chart: Model performance by metric type
  - Line chart: Performance trends over time
  - Scatter plot: Precision vs Recall by model

### 2. Recommendation Data (recommendation_data.csv)
- **Purpose**: Analyze recommendation patterns and user engagement
- **Key Fields**: User_ID, Model_Name, Track_ID, Recommendation_Score, Rank_Position
- **Visualizations**:
  - Heatmap: Recommendation scores by model and rank
  - Distribution: Score distribution by model
  - Top tracks: Most recommended tracks

### 3. User Interactions (user_interactions.csv)
- **Purpose**: Understand user behavior and engagement patterns
- **Key Fields**: User_ID, Interaction_Type, Rating, Timestamp, Context
- **Visualizations**:
  - Timeline: Interaction patterns over time
  - Funnel: User engagement funnel
  - Cohort: User retention analysis

### 4. Model Performance (model_performance.csv)
- **Purpose**: Track model performance over time
- **Key Fields**: Model_Name, Metric_Name, Metric_Value, Period_Start
- **Visualizations**:
  - Time series: Performance trends
  - Comparison: Model performance comparison
  - KPI cards: Key performance indicators

## Recommended Dashboard Layout

### Page 1: Executive Summary
- KPI cards for key metrics
- Model performance comparison chart
- Top performing models table

### Page 2: Detailed Analysis
- Precision/Recall scatter plot
- NDCG trends over time
- User engagement metrics

### Page 3: User Behavior
- Interaction patterns by time
- Rating distribution
- Context-based performance

### Page 4: A/B Testing
- Test results comparison
- Statistical significance indicators
- Test duration analysis

## Power BI Setup Instructions

1. **Import Data**:
   - Connect to CSV files in the data directory
   - Set up relationships between tables
   - Configure data types and formats

2. **Create Measures**:
   - Average Precision = AVERAGE(evaluation_metrics[Metric_Value])
   - Total Interactions = COUNT(user_interactions[User_ID])
   - Model Performance = CALCULATE(AVERAGE(model_performance[Metric_Value]))

3. **Set Up Filters**:
   - Date range filter for time-based analysis
   - Model selection filter
   - Metric type filter

4. **Create Visualizations**:
   - Use the guide above for specific chart types
   - Apply consistent color scheme
   - Add tooltips and drill-through capabilities

## Data Refresh Schedule

- **Daily**: Model performance metrics
- **Weekly**: User interaction summaries
- **Monthly**: A/B test results and evaluation metrics

## Notes

- All timestamps are in UTC
- Context fields contain JSON data that may need parsing
- Some aggregations are pre-calculated for performance
- Use DAX measures for complex calculations
"""
        
        with open(output_dir / "powerbi_guide.md", 'w') as f:
            f.write(guide_content)
        
        logger.info("Power BI visualization guide created")
    
    def export_all_data(self, output_dir: Path) -> None:
        """Export all data for Power BI."""
        logger.info("Starting Power BI data export...")
        
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Export all datasets
            self.export_evaluation_metrics(output_dir)
            self.export_recommendation_data(output_dir)
            self.export_user_interactions(output_dir)
            self.export_model_performance(output_dir)
            self.export_ab_test_results(output_dir)
            
            # Create visualization guide
            self.create_visualization_guide(output_dir)
            
            # Create data summary
            self._create_data_summary(output_dir)
            
            logger.info(f"All data exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise
    
    def _create_data_summary(self, output_dir: Path) -> None:
        """Create data summary for Power BI."""
        logger.info("Creating data summary...")
        
        summary = {
            'export_date': datetime.now().isoformat(),
            'datasets': [
                'evaluation_metrics.csv',
                'recommendation_data.csv',
                'user_interactions.csv',
                'model_performance.csv',
                'ab_test_results.csv'
            ],
            'aggregates': [
                'evaluation_metrics_pivot.csv',
                'model_k_performance.csv',
                'mood_performance.csv',
                'user_engagement.csv',
                'interaction_distribution.csv',
                'hourly_patterns.csv',
                'day_of_week_patterns.csv',
                'rating_distribution.csv',
                'daily_performance_trends.csv',
                'model_comparison.csv',
                'ab_test_summary.csv',
                'significance_analysis.csv'
            ],
            'guide': 'powerbi_guide.md'
        }
        
        with open(output_dir / "data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Data summary created")


def main():
    """Main function to create Power BI visualizations."""
    logger.info("Starting Power BI data export...")
    
    try:
        # Initialize exporter
        exporter = PowerBIDataExporter()
        
        # Export all data
        output_dir = Path("results/powerbi_data")
        exporter.export_all_data(output_dir)
        
        logger.info("Power BI data export completed successfully!")
        
    except Exception as e:
        logger.error(f"Power BI export failed: {e}")
        raise
    finally:
        exporter.session.close()


if __name__ == "__main__":
    main()
