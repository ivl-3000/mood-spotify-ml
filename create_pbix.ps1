# Spotify Music Analytics Dashboard - Power BI Creation Script
# This script helps automate the creation of the Power BI dashboard

Write-Host "🎵 Spotify Music Analytics Dashboard - Power BI Creation" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Check if Power BI Desktop is installed
$powerBI = Get-Process "PBIDesktop" -ErrorAction SilentlyContinue
if (-not $powerBI) {
    Write-Host "⚠️ Power BI Desktop is not running. Please start Power BI Desktop first." -ForegroundColor Yellow
    Write-Host "1. Open Power BI Desktop" -ForegroundColor Cyan
    Write-Host "2. Create a new report" -ForegroundColor Cyan
    Write-Host "3. Save as: Spotify_Music_Analytics_Dashboard.pbix" -ForegroundColor Cyan
    Write-Host "4. Follow the QUICK_PBIX_CREATION.md guide" -ForegroundColor Cyan
} else {
    Write-Host "✅ Power BI Desktop is running" -ForegroundColor Green
}

Write-Host ""
Write-Host "📁 Files ready for import:" -ForegroundColor Cyan
Write-Host "- pbix_tracks_data.csv (300 tracks)" -ForegroundColor White
Write-Host "- pbix_top_happy_songs.csv (Top 10 happy songs)" -ForegroundColor White
Write-Host "- pbix_top_sad_songs.csv (Top 10 sad songs)" -ForegroundColor White
Write-Host "- spotify_pbix_theme.json (Spotify theme)" -ForegroundColor White

Write-Host ""
Write-Host "🎨 Spotify Colors:" -ForegroundColor Cyan
Write-Host "- Primary: #1DB954 (Spotify Green)" -ForegroundColor Green
Write-Host "- Background: #191414 (Dark)" -ForegroundColor DarkGray
Write-Host "- Surface: #282828 (Dark Gray)" -ForegroundColor Gray
Write-Host "- Text: #FFFFFF (White)" -ForegroundColor White

Write-Host ""
Write-Host "🚀 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Import the CSV files into Power BI" -ForegroundColor White
Write-Host "2. Apply the Spotify theme" -ForegroundColor White
Write-Host "3. Create relationships between tables" -ForegroundColor White
Write-Host "4. Add DAX measures" -ForegroundColor White
Write-Host "5. Create visualizations" -ForegroundColor White
Write-Host "6. Apply Spotify styling" -ForegroundColor White

Write-Host ""
Write-Host "📖 Follow the QUICK_PBIX_CREATION.md guide for detailed instructions" -ForegroundColor Yellow
Write-Host "🎵 Your Spotify Music Analytics Dashboard will be ready in 15 minutes! 🎵" -ForegroundColor Green
