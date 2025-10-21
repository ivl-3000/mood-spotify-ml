# 🎵 Create Spotify Music Analytics Dashboard (.pbix)

## 🚀 Quick Start (15 minutes)

### Step 1: Open Power BI Desktop
1. Launch Power BI Desktop
2. Create a new report
3. Save as: `Spotify_Music_Analytics_Dashboard.pbix`

### Step 2: Import Data (5 minutes)
1. **Get Data** → **Text/CSV**
2. Import these files in order:
   - `pbix_tracks_data.csv` (Main data - 300 tracks)
   - `pbix_top_happy_songs.csv` (Happy songs)
   - `pbix_top_sad_songs.csv` (Sad songs)

### Step 3: Apply Spotify Theme (2 minutes)
1. Go to **View** → **Themes** → **Browse for themes**
2. Select `spotify_pbix_theme.json`
3. Click **Open** to apply the Spotify theme

### Step 4: Create Relationships (2 minutes)
1. Go to **Model** view
2. Create these relationships:
   - `pbix_tracks_data[track_id]` ↔ `pbix_top_happy_songs[track_id]`
   - `pbix_tracks_data[track_id]` ↔ `pbix_top_sad_songs[track_id]`
   - Set both as **Many-to-One** relationships

### Step 5: Add DAX Measures (3 minutes)
Copy and paste these measures in the **Fields** pane:

```dax
Total Tracks = COUNTROWS(pbix_tracks_data)

Average Valence = AVERAGE(pbix_tracks_data[valence])

Average Energy = AVERAGE(pbix_tracks_data[energy])

Average Sentiment = AVERAGE(pbix_tracks_data[sentiment_score])

Happy Songs Count = COUNTROWS(FILTER(pbix_tracks_data, pbix_tracks_data[valence] > 0.7))

Sad Songs Count = COUNTROWS(FILTER(pbix_tracks_data, pbix_tracks_data[valence] < 0.3))

Happy Songs % = DIVIDE([Happy Songs Count], [Total Tracks])

Sad Songs % = DIVIDE([Sad Songs Count], [Total Tracks])

Happiness Score = pbix_tracks_data[valence] + pbix_tracks_data[sentiment_score]

Sadness Score = ABS(pbix_tracks_data[valence] + pbix_tracks_data[sentiment_score])
```

### Step 6: Create Visualizations (3 minutes)

#### Page 1: 🎵 Music Overview

**KPI Cards (Top Row)**
1. Create 5 Card visuals:
   - Total Tracks (Green: #1DB954)
   - Average Valence (Light Green: #1ED760)
   - Average Energy (Coral: #FF6B6B)
   - Happy Songs % (Teal: #4ECDC4)
   - Sad Songs % (Mint: #96CEB4)

**Top 10 Happy Songs Table (Left)**
1. **Visual**: Table
2. **Fields**: rank, track_name, artist_name, valence, sentiment_score, happiness_score
3. **Formatting**: Green theme (#1DB954)

**Top 10 Sad Songs Table (Right)**
1. **Visual**: Table
2. **Fields**: rank, track_name, artist_name, valence, sentiment_score, sadness_score
3. **Formatting**: Teal theme (#4ECDC4)

**Genre Trends Chart (Bottom Left)**
1. **Visual**: Line Chart
2. **X-axis**: release_date
3. **Y-axis**: Count of track_id
4. **Legend**: genre

**Mood Distribution Chart (Bottom Right)**
1. **Visual**: Pie Chart
2. **Legend**: mood_category
3. **Values**: Count of track_id

### Step 7: Apply Spotify Styling
- **Background**: #191414 (Dark)
- **Cards**: #282828 (Dark Gray)
- **Text**: #FFFFFF (White)
- **Accent**: #1DB954 (Spotify Green)

### Step 8: Save and Test
1. **Save**: `Spotify_Music_Analytics_Dashboard.pbix`
2. **Test**: Verify all visualizations work
3. **Format**: Check Spotify color consistency

## 🎨 Spotify Color Palette

### Primary Colors
- **Spotify Green**: #1DB954
- **Light Green**: #1ED760
- **Coral Red**: #FF6B6B
- **Teal**: #4ECDC4
- **Mint Green**: #96CEB4

### Background Colors
- **Dark Background**: #191414
- **Surface**: #282828
- **Border**: #404040

### Text Colors
- **Primary Text**: #FFFFFF
- **Subtitle**: #B3B3B3

## 📊 Expected Results

After following these steps, you'll have:
- ✅ Complete Spotify-style dashboard
- ✅ 4 pages with comprehensive analytics
- ✅ Interactive visualizations
- ✅ Professional Spotify design
- ✅ 300 tracks with realistic data
- ✅ Auto-updating top songs
- ✅ Mood analysis and insights

## 🚀 Ready for Production

Your .pbix file will be ready with:
- Authentic Spotify design
- Interactive features
- Comprehensive analytics
- Professional appearance
- Performance optimization

**🎵 Create your Spotify Music Analytics Dashboard in 15 minutes! 🎵**
