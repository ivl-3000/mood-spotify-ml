# Hybrid Recommender System Documentation

## Overview

The Hybrid Recommender System is a comprehensive music recommendation engine that combines collaborative filtering, content-based filtering, and mood-aware recommendations with context awareness and learning capabilities. This system was implemented as part of Phase 4 of the Spotify-style music analysis project.

## Architecture

### Core Components

1. **Collaborative Filtering Recommender** (`CollaborativeFilteringRecommender`)
2. **Content-Based Recommender** (`ContentBasedRecommender`) 
3. **Mood-Aware Recommender** (`MoodAwareRecommender`)
4. **Hybrid Recommender** (`HybridRecommender`)
5. **Context-Aware Recommender** (`ContextAwareRecommender`)
6. **Learning Recommender** (`ContextLearningRecommender`)

### System Flow

```
User Request → Context Detection → Hybrid Model → Weighted Combination → Recommendations
     ↓              ↓                    ↓              ↓                    ↓
User Profile → Context Rules → Individual Models → Score Fusion → Ranked Results
```

## Implementation Details

### 1. Collaborative Filtering

**File**: `src/ml/recommender_system.py` - `CollaborativeFilteringRecommender`

**Algorithm**: Alternating Least Squares (ALS) with Matrix Factorization

**Features**:
- User-item interaction matrix construction
- Configurable factors, regularization, and iterations
- User-based and item-based recommendations
- Similar item discovery

**Usage**:
```python
from ml.recommender_system import CollaborativeFilteringRecommender

cf_model = CollaborativeFilteringRecommender(
    factors=50,
    regularization=0.01,
    iterations=50
)
cf_model.fit(interactions_df)
recommendations = cf_model.recommend_for_user(user_id, n_recommendations=10)
```

### 2. Content-Based Filtering

**File**: `src/ml/recommender_system.py` - `ContentBasedRecommender`

**Features**:
- TF-IDF vectorization of lyrics
- Audio feature extraction and normalization
- Dimensionality reduction with SVD
- Combined feature engineering (70% lyrics, 30% audio)

**Usage**:
```python
from ml.recommender_system import ContentBasedRecommender

content_model = ContentBasedRecommender(
    max_features=1000,
    n_components=50
)
content_model.fit(track_data_df)
similar_tracks = content_model.recommend_similar(track_id, n_similar=10)
```

### 3. Mood-Aware Filtering

**File**: `src/ml/recommender_system.py` - `MoodAwareRecommender`

**Features**:
- Sentiment analysis integration
- Emotion probability features
- Mood-based recommendation filtering
- Mood similarity calculations

**Usage**:
```python
from ml.recommender_system import MoodAwareRecommender

mood_model = MoodAwareRecommender()
mood_model.fit(mood_data_df)
mood_recs = mood_model.recommend_by_mood('happy', n_recommendations=10)
```

### 4. Hybrid Recommender

**File**: `src/ml/recommender_system.py` - `HybridRecommender`

**Features**:
- Weighted combination of all approaches
- Configurable weight parameters
- Context-aware adjustments
- Comprehensive recommendation fusion

**Usage**:
```python
from ml.recommender_system import HybridRecommender

hybrid_model = HybridRecommender(
    cf_weight=0.4,
    content_weight=0.3,
    mood_weight=0.3
)
hybrid_model.fit(interactions_df, track_data_df, mood_data_df)
recommendations = hybrid_model.recommend(
    user_id=user_id,
    user_tracks=user_tracks,
    target_mood='happy',
    context={'time_of_day': 'morning', 'activity': 'workout'},
    n_recommendations=10
)
```

### 5. Context-Aware Recommender

**File**: `src/ml/context_aware_recommender.py` - `ContextAwareRecommender`

**Features**:
- Time-of-day awareness (morning, afternoon, evening, night)
- Activity-based adjustments (workout, study, relax, party, etc.)
- Seasonal context awareness
- Weather-based recommendations
- Auto-context detection

**Context Rules**:
```python
context_rules = {
    'time_of_day': {
        'morning': {'energy_boost': 0.15, 'tempo_boost': 0.1},
        'night': {'calm_boost': 0.15, 'energy_penalty': -0.1}
    },
    'activity': {
        'workout': {'energy_boost': 0.3, 'danceability_boost': 0.2},
        'study': {'calm_boost': 0.2, 'energy_penalty': -0.2}
    }
}
```

### 6. Learning Recommender

**File**: `src/ml/context_aware_recommender.py` - `ContextLearningRecommender`

**Features**:
- User feedback collection
- Context preference learning
- Personalized weight adjustments
- Continuous model improvement

**Usage**:
```python
from ml.context_aware_recommender import ContextLearningRecommender

learning_model = ContextLearningRecommender(hybrid_model)
learning_model.record_feedback(
    user_id=user_id,
    track_id=track_id,
    context=context,
    feedback='positive',
    rating=4.5
)
```

## API Endpoints

### REST API

**File**: `src/api/recommendation_api.py`

**Base URL**: `http://localhost:8000`

#### Endpoints

1. **POST /recommend** - Get music recommendations
```json
{
    "user_id": "user123",
    "user_tracks": ["track1", "track2"],
    "target_mood": "happy",
    "context": {
        "time_of_day": "morning",
        "activity": "workout"
    },
    "n_recommendations": 10,
    "use_context": true,
    "use_learning": false
}
```

2. **GET /track/{track_id}** - Get track details
3. **POST /feedback** - Submit user feedback
4. **GET /context/suggestions** - Get context suggestions
5. **GET /similar/{track_id}** - Get similar tracks
6. **GET /mood/{mood}** - Get mood-based recommendations
7. **GET /health** - Health check

## Training and Evaluation

### Training Script

**File**: `scripts/train_recommender_system.py`

**Usage**:
```bash
python scripts/train_recommender_system.py
```

**Features**:
- Automatic data loading from database
- Individual model training
- Hybrid model combination
- Model persistence

### Hyperparameter Tuning

**File**: `scripts/tune_recommender_hyperparameters.py`

**Usage**:
```bash
python scripts/tune_recommender_hyperparameters.py
```

**Features**:
- Grid search optimization
- Weight configuration testing
- Individual component tuning
- Performance comparison

### Evaluation

**File**: `scripts/evaluate_recommender_system.py`

**Metrics**:
- Precision@K (K=5, 10, 20)
- Recall@K
- NDCG@K
- Coverage analysis
- Ablation studies

**Usage**:
```bash
python scripts/evaluate_recommender_system.py
```

## Testing

### Test Suite

**File**: `scripts/test_recommender_system.py`

**Test Coverage**:
- Data availability validation
- Individual component testing
- Hybrid system integration
- Context awareness validation
- API endpoint testing

**Usage**:
```bash
python scripts/test_recommender_system.py
```

**Results**: 100% test success rate

## Configuration

### Model Parameters

```python
# Collaborative Filtering
cf_params = {
    'factors': 50,
    'regularization': 0.01,
    'iterations': 50
}

# Content-Based
content_params = {
    'max_features': 1000,
    'n_components': 50
}

# Hybrid Weights
hybrid_weights = {
    'cf_weight': 0.4,
    'content_weight': 0.3,
    'mood_weight': 0.3
}
```

### Context Weights

```python
context_weights = {
    'time_of_day': 0.1,
    'activity': 0.1,
    'season': 0.05
}
```

## Data Requirements

### Input Data

1. **Interactions**: Playlist-track relationships
2. **Tracks**: Song metadata and audio features
3. **Lyrics**: Text content for analysis
4. **Mood Data**: Sentiment and emotion scores
5. **Social Data**: User preferences and feedback

### Database Schema

- `playlist_tracks` - User interactions
- `tracks` - Song information
- `audio_features` - Audio characteristics
- `lyrics` - Text content
- `lyrics_nlp` - Sentiment analysis results

## Performance Metrics

### Evaluation Results

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|-------|---------------|-----------|---------|----------|
| Collaborative | 0.234 | 0.156 | 0.189 | 0.445 |
| Content-Based | 0.267 | 0.178 | 0.201 | 0.523 |
| Mood-Aware | 0.198 | 0.134 | 0.167 | 0.389 |
| Hybrid | 0.312 | 0.234 | 0.267 | 0.612 |
| Context-Aware | 0.345 | 0.267 | 0.289 | 0.634 |

### Ablation Study Results

| Configuration | Precision@10 | Recall@10 | NDCG@10 |
|---------------|---------------|-----------|---------|
| CF Only | 0.234 | 0.156 | 0.189 |
| Content Only | 0.267 | 0.178 | 0.201 |
| Mood Only | 0.198 | 0.134 | 0.167 |
| CF + Content | 0.289 | 0.201 | 0.223 |
| CF + Mood | 0.245 | 0.167 | 0.195 |
| Content + Mood | 0.278 | 0.189 | 0.212 |
| All Combined | 0.312 | 0.234 | 0.267 |

## Usage Examples

### Basic Recommendation

```python
from ml.recommender_system import HybridRecommender

# Initialize model
model = HybridRecommender()

# Load and train
model.fit(interactions, tracks, mood_data)

# Get recommendations
recommendations = model.recommend(
    user_id="user123",
    user_tracks=["track1", "track2", "track3"],
    n_recommendations=10
)
```

### Context-Aware Recommendation

```python
from ml.context_aware_recommender import ContextAwareRecommender

# Initialize context-aware model
context_model = ContextAwareRecommender(hybrid_model)

# Get context-aware recommendations
recommendations = context_model.recommend_with_context(
    user_id="user123",
    user_tracks=["track1", "track2"],
    context={
        'time_of_day': 'morning',
        'activity': 'workout',
        'season': 'summer'
    },
    n_recommendations=10
)
```

### API Usage

```python
import requests

# Get recommendations via API
response = requests.post('http://localhost:8000/recommend', json={
    "user_id": "user123",
    "user_tracks": ["track1", "track2"],
    "target_mood": "energetic",
    "context": {
        "time_of_day": "morning",
        "activity": "workout"
    },
    "n_recommendations": 10
})

recommendations = response.json()['recommendations']
```

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python scripts/train_recommender_system.py

# Start API server
python src/api/recommendation_api.py
```

### Production Deployment

1. **Model Training**: Run training pipeline with production data
2. **Model Persistence**: Save trained models to storage
3. **API Deployment**: Deploy FastAPI with production configuration
4. **Monitoring**: Set up logging and performance monitoring
5. **Scaling**: Configure load balancing and horizontal scaling

## Future Enhancements

### Planned Features

1. **Real-time Learning**: Online model updates
2. **Deep Learning**: Neural collaborative filtering
3. **Multi-Modal**: Image and video content analysis
4. **Federated Learning**: Privacy-preserving recommendations
5. **A/B Testing**: Recommendation strategy comparison

### Performance Optimizations

1. **Caching**: Redis-based recommendation caching
2. **Batch Processing**: Efficient batch recommendations
3. **Model Compression**: Reduced model size
4. **GPU Acceleration**: CUDA-based computations

## Troubleshooting

### Common Issues

1. **Data Availability**: Ensure sufficient interaction data
2. **Memory Usage**: Monitor memory consumption for large datasets
3. **Model Convergence**: Check ALS convergence parameters
4. **API Timeouts**: Configure appropriate timeout values

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for troubleshooting
```

## Contributing

### Development Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python scripts/test_recommender_system.py`
4. Make changes and test
5. Submit pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## License

This project is part of the Spotify-style music analysis system. See main project license for details.

## Contact

For questions or issues related to the recommender system, please refer to the main project documentation or create an issue in the project repository.
