# 🎵 Create Spotify-Style Power BI Dashboard (.pbix)

## 🎯 Step-by-Step Instructions

### 1. Open Power BI Desktop
1. Launch Power BI Desktop
2. Create a new report
3. Save as: `Spotify_Music_Analytics_Dashboard.pbix`

### 2. Import Data Sources
1. **Get Data** → **Text/CSV**
2. Import these files in order:
   - `enhanced_tracks_data.csv` (Main data source)
   - `top_10_happy_songs.csv` (Happy songs)
   - `top_10_sad_songs.csv` (Sad songs)

### 3. Create Relationships
1. Go to **Model** view
2. Create these relationships:
   - `enhanced_tracks_data[track_id]` ↔ `top_10_happy_songs[track_id]`
   - `enhanced_tracks_data[track_id]` ↔ `top_10_sad_songs[track_id]`
   - Set both as **Many-to-One** relationships

### 4. Apply Spotify Theme
1. Go to **View** → **Themes** → **Browse for themes**
2. Select `spotify_theme.json`
3. Click **Open** to apply the theme

### 5. Create DAX Measures
Add these measures in the **Fields** pane:

```dax
Total Tracks = COUNTROWS(enhanced_tracks_data)

Average Valence = AVERAGE(enhanced_tracks_data[valence])

Average Energy = AVERAGE(enhanced_tracks_data[energy])

Average Sentiment = AVERAGE(enhanced_tracks_data[sentiment_score])

Happy Songs Count = COUNTROWS(FILTER(enhanced_tracks_data, enhanced_tracks_data[valence] > 0.7))

Sad Songs Count = COUNTROWS(FILTER(enhanced_tracks_data, enhanced_tracks_data[valence] < 0.3))

Happy Songs % = DIVIDE([Happy Songs Count], [Total Tracks])

Sad Songs % = DIVIDE([Sad Songs Count], [Total Tracks])

Happiness Score = enhanced_tracks_data[valence] + enhanced_tracks_data[sentiment_score]

Sadness Score = ABS(enhanced_tracks_data[valence] + enhanced_tracks_data[sentiment_score])
```

### 6. Create Visualizations

#### Page 1: 🎵 Music Overview

**KPI Cards (Top Row)**
1. **Total Tracks Card**:
   - Visual: Card
   - Field: Total Tracks
   - Color: #1DB954 (Spotify Green)

2. **Average Valence Card**:
   - Visual: Card
   - Field: Average Valence
   - Color: #1ED760 (Light Green)

3. **Average Energy Card**:
   - Visual: Card
   - Field: Average Energy
   - Color: #FF6B6B (Coral)

4. **Happy Songs % Card**:
   - Visual: Card
   - Field: Happy Songs %
   - Color: #4ECDC4 (Teal)

5. **Sad Songs % Card**:
   - Visual: Card
   - Field: Sad Songs %
   - Color: #96CEB4 (Mint)

**Top 10 Happy Songs Table (Left Side)**
1. **Visual**: Table
2. **Fields**:
   - Rank (from top_10_happy_songs)
   - track_name
   - artist_name
   - valence
   - sentiment_score
   - happiness_score
3. **Formatting**:
   - Background: #282828
   - Text: #FFFFFF
   - Header: #1DB954

**Top 10 Sad Songs Table (Right Side)**
1. **Visual**: Table
2. **Fields**:
   - Rank (from top_10_sad_songs)
   - track_name
   - artist_name
   - valence
   - sentiment_score
   - sadness_score
3. **Formatting**:
   - Background: #282828
   - Text: #FFFFFF
   - Header: #4ECDC4

**Genre Trends Chart (Bottom Left)**
1. **Visual**: Line Chart
2. **X-axis**: release_date
3. **Y-axis**: Count of track_id
4. **Legend**: genre
5. **Colors**: Use Spotify color palette

**Mood Distribution Chart (Bottom Right)**
1. **Visual**: Pie Chart
2. **Legend**: Create calculated column for mood categories
3. **Values**: Count of track_id
4. **Colors**: Spotify color scheme

#### Page 2: 🎯 Mood Analysis

**Mood Timeline (Top)**
1. **Visual**: Line Chart
2. **X-axis**: release_date
3. **Y-axis**: Average of sentiment_score
4. **Color**: sentiment_score (gradient)

**Sentiment Heatmap (Bottom Left)**
1. **Visual**: Matrix
2. **Rows**: energy (binned)
3. **Columns**: valence (binned)
4. **Values**: Count of track_id
5. **Color**: Use heatmap color scheme

**Audio Features Correlation (Bottom Right)**
1. **Visual**: Scatter Chart
2. **X-axis**: valence
3. **Y-axis**: energy
4. **Color**: sentiment_score
5. **Size**: popularity

#### Page 3: 🎧 Interactive Playlist

**Mood Filter Slicer (Top)**
1. **Visual**: Slicer
2. **Field**: Create mood_category calculated column
3. **Formatting**: Spotify-style buttons

**User Personas Chart (Left)**
1. **Visual**: Pie Chart
2. **Legend**: persona_type (calculated)
3. **Values**: Count of track_id

**Playlist Comparison (Right)**
1. **Visual**: Clustered Column Chart
2. **X-axis**: period (Before/After)
3. **Y-axis**: Average of valence, energy, danceability

#### Page 4: 📊 Analytics Deep Dive

**Precision/Recall Analysis (Top Left)**
1. **Visual**: Line Chart
2. **X-axis**: K values
3. **Y-axis**: Precision, Recall measures

**Coverage Analysis (Top Right)**
1. **Visual**: Gauge
2. **Value**: Coverage percentage
3. **Target**: 80%

**Diversity Analysis (Bottom Left)**
1. **Visual**: Bar Chart
2. **X-axis**: genre
3. **Y-axis**: Count of unique artists

**Novelty Analysis (Bottom Right)**
1. **Visual**: Scatter Chart
2. **X-axis**: popularity
3. **Y-axis**: novelty_score (calculated)

### 7. Apply Spotify Styling

#### Color Scheme
- **Primary**: #1DB954 (Spotify Green)
- **Secondary**: #1ED760 (Light Green)
- **Accent**: #FF6B6B (Coral)
- **Background**: #191414 (Dark)
- **Surface**: #282828 (Dark Gray)
- **Text**: #FFFFFF (White)
- **Subtitle**: #B3B3B3 (Light Gray)

#### Typography
- **Font Family**: Circular, Helvetica, Arial, sans-serif
- **Title Size**: 24px
- **Subtitle Size**: 16px
- **Body Size**: 14px

#### Visual Formatting
- **Border Radius**: 8px for cards, 20px for buttons
- **Shadows**: 0 4px 12px rgba(0,0,0,0.4)
- **Gradients**: Linear gradients for backgrounds
- **Hover Effects**: Transform and shadow changes

### 8. Add Filters and Slicers

**Global Filters**:
1. **Playlist Name**: Slicer
2. **Mood Category**: Slicer
3. **Date Range**: Date slicer
4. **Energy Level**: Slicer

**Formatting**:
- Background: #282828
- Border: #404040
- Selected: #1DB954
- Text: #FFFFFF

### 9. Final Touches

#### Page Backgrounds
- Set each page background to #191414
- Add subtle gradients if desired

#### Visual Consistency
- Ensure all visuals use Spotify colors
- Apply consistent spacing (16px)
- Use proper typography hierarchy

#### Interactive Elements
- Enable cross-filtering
- Add drill-through capabilities
- Set up bookmarks for navigation

### 10. Save and Test

1. **Save**: `Spotify_Music_Analytics_Dashboard.pbix`
2. **Test**: Verify all visualizations work
3. **Format**: Check color consistency
4. **Performance**: Ensure smooth interactions

## 🎨 Spotify Design Checklist

- [ ] Spotify green (#1DB954) used consistently
- [ ] Dark background (#191414) applied
- [ ] Circular font family used
- [ ] Proper spacing and padding
- [ ] Hover effects and animations
- [ ] Gradient backgrounds
- [ ] Professional shadows
- [ ] Consistent color scheme
- [ ] Responsive layout
- [ ] Interactive elements

## 🚀 Ready for Production

Your Spotify-style Power BI dashboard will be ready with:
- ✅ Authentic Spotify design
- ✅ Interactive features
- ✅ Comprehensive analytics
- ✅ Professional appearance
- ✅ Performance optimization

**🎵 Create your Spotify Music Analytics Dashboard with authentic design! 🎵**
