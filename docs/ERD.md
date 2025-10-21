# ER Diagram and Workflow (Spotify-style)

## Entity-Relationship Diagram
The ERD models Spotify-like entities (users, artists, albums, tracks, playlists) and project-specific tables (lyrics, NLP outputs, social sentiment, interactions).

```mermaid
erDiagram
  USERS ||--o{ PLAYLISTS : owns
  USERS ||--o{ USER_INTERACTIONS : has

  ARTISTS ||--o{ ALBUMS : creates
  ARTISTS ||--o{ TRACK_ARTISTS : appears_on
  TRACKS ||--o{ TRACK_ARTISTS : has

  ALBUMS ||--o{ TRACKS : contains

  PLAYLISTS ||--o{ PLAYLIST_TRACKS : includes
  TRACKS ||--o{ PLAYLIST_TRACKS : appears_in

  TRACKS ||--|| AUDIO_FEATURES : has
  TRACKS ||--o{ LYRICS : has
  TRACKS ||--o{ LYRICS_NLP : has

  SOCIAL_SENTIMENT }o--|| ARTISTS : about
  SOCIAL_SENTIMENT }o--|| TRACKS : about

  USERS {
    string id PK
    string handle
    string country
    string created_at
  }

  ARTISTS {
    string id PK
    string name
    string[] genres
    int followers
  }

  ALBUMS {
    string id PK
    string name
    date release_date
    string label
    string artist_id FK
  }

  TRACKS {
    string id PK
    string name
    string album_id FK
    int duration_ms
    int popularity
    date release_date
  }

  TRACK_ARTISTS {
    string track_id FK
    string artist_id FK
    int position
  }

  AUDIO_FEATURES {
    string track_id PK, FK
    float danceability
    float energy
    float valence
    float tempo
    int key
    int mode
    float loudness
  }

  PLAYLISTS {
    string id PK
    string name
    string owner_user_id FK
    int followers
    string created_at
  }

  PLAYLIST_TRACKS {
    string playlist_id FK
    string track_id FK
    string added_at
    int position
  }

  LYRICS {
    string track_id FK
    string language
    text lyrics_text
    string source
  }

  LYRICS_NLP {
    string track_id FK
    float sentiment_score
    json emotion_probs
    string dominant_emotion
  }

  SOCIAL_SENTIMENT {
    string entity_type
    string entity_id
    string time_window
    float sentiment_score
    int volume
  }

  USER_INTERACTIONS {
    string user_id FK
    string track_id FK
    string interaction_type
    string timestamp
  }
```

PNG (rendered by CI): `docs/diagrams/img/spotify_erd.png`

## Workflow (Spotify-style)
```mermaid
flowchart LR
  U[Users & Clients] --> API[API Gateway]
  API --> PLAY[Playback]
  API --> SEARCH[Search]
  API --> PLAYLIST[Playlist]
  API --> RECO[Recommendations API]

  PLAY & SEARCH & PLAYLIST --> K[(Kafka)]
  K --> SPROC[Stream Proc]
  SPROC --> FEATS[Feature Store]

  DB[(OLTP)] --> FEATS
  FEATS --> TRAIN[Model Training]
  TRAIN --> REG[Model Registry] --> RANK[Online Ranker] --> RECO

  RECO --> DASH[Predictions/Telemetry]
  DASH --> BI[Power BI]
```

PNG (rendered by CI): `docs/diagrams/img/spotify_workflow.png`
