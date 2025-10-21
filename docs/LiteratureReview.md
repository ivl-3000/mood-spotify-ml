# Literature Review: Mood-Aware Music Recommendation

## Themes
- Mood/Sentiment signals: Lyrics-based sentiment/emotion classification can proxy short-term context; multi-label emotions (e.g., joy, sadness, anger) often outperform coarse polarity.
- Hybrid modeling: Combining content (audio/lyrics) with collaborative signals improves coverage and robustness, especially for cold-start and niche moods.
- Context-awareness: Time-of-day, activity, and recent interactions can inform mood priors and dynamic re-ranking.
- Evaluation: Offline ranking metrics should be complemented by mood-coherence checks (e.g., emotion agreement) and diversity/novelty trade-offs.
- Ethics: Avoid sensitive inferences; ensure user control and transparency about mood usage.

## Selected References (representative)
- Oramas et al., “A Deep Multimodal Approach for Cold-start Music Recommendation,” RecSys (audio+text content for cold-start).
- Schedl et al., “Music Recommender Systems: Techniques, Evaluation, and Directions for Future Research,” Foundations and Trends (survey).
- Hu & Downie, “Exploring Mood Metadata: Relationships with Genre, Artist and Usage,” ISMIR (mood taxonomies and relationships).
- Yang & Chen, “Machine Recognition of Music Emotion: A Review,” ACM Computing Surveys (music emotion recognition overview).
- Mohammad & Turney, “Crowdsourcing a Word-Emotion Association Lexicon,” (NRC Emotion Lexicon) (emotion labels useful for mapping).
- Felbo et al., “Using millions of emoji occurrences to learn sentiment and emotion,” EMNLP (transferable emotion features).
- Hutto & Gilbert, “VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text,” (social sentiment baseline).
- Rendle, “Factorization Machines,” IEEE (foundation for hybrid recommenders).

## Implications for This Project
- Use transformer-based classifiers for lyrics emotion; cross-check with lexicon-based baselines (NRC, VADER) for robustness.
- Represent user state with recent mood distribution and apply a re-ranking objective balancing relevance and mood coherence.
- Validate uplift vs. a non-mood baseline using Recall@K/Precision@K, plus mood-consistency metrics.
- Start with CPU-feasible models; modularize to swap in stronger models later.
