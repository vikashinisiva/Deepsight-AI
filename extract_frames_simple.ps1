# Frame Extraction Script for FaceForensics++ Dataset
# Extracts frames at 1 fps from real and fake videos

Write-Host "=== FACEFORENSICS++ FRAME EXTRACTION ===" -ForegroundColor Green
Write-Host ""

# Check if directories exist
if (-not (Test-Path "ffpp_data\real_videos")) {
    Write-Host "Error: ffpp_data\real_videos directory not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "ffpp_data\fake_videos")) {
    Write-Host "Error: ffpp_data\fake_videos directory not found!" -ForegroundColor Red
    exit 1
}

# Create output directories
Write-Host "Creating frame directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "frames\real" -Force | Out-Null
New-Item -ItemType Directory -Path "frames\fake" -Force | Out-Null

# Process REAL videos
Write-Host "Processing REAL videos..." -ForegroundColor Cyan
$realVideos = Get-ChildItem "ffpp_data\real_videos\*.mp4"
$realCount = 0
$realTotal = $realVideos.Count

foreach ($video in $realVideos) {
    $realCount++
    $filename = [IO.Path]::GetFileNameWithoutExtension($video.Name)
    Write-Host "[$realCount/$realTotal] Extracting frames from: $filename" -ForegroundColor White
    
    # Extract frames at 1 fps
    ffmpeg -i $video.FullName -r 1 -q:v 2 "frames\real\${filename}_%03d.jpg" -loglevel error
}

Write-Host ""
Write-Host "Processing FAKE videos..." -ForegroundColor Cyan
$fakeVideos = Get-ChildItem "ffpp_data\fake_videos\*.mp4"
$fakeCount = 0
$fakeTotal = $fakeVideos.Count

foreach ($video in $fakeVideos) {
    $fakeCount++
    $filename = [IO.Path]::GetFileNameWithoutExtension($video.Name)
    Write-Host "[$fakeCount/$fakeTotal] Extracting frames from: $filename" -ForegroundColor White
    
    # Extract frames at 1 fps
    ffmpeg -i $video.FullName -r 1 -q:v 2 "frames\fake\${filename}_%03d.jpg" -loglevel error
}

# Count results
$realFrames = (Get-ChildItem "frames\real\*.jpg" -ErrorAction SilentlyContinue).Count
$fakeFrames = (Get-ChildItem "frames\fake\*.jpg" -ErrorAction SilentlyContinue).Count

Write-Host ""
Write-Host "=== EXTRACTION COMPLETE ===" -ForegroundColor Green
Write-Host "Real frames extracted: $realFrames" -ForegroundColor Green
Write-Host "Fake frames extracted: $fakeFrames" -ForegroundColor Green
Write-Host "Total frames: $($realFrames + $fakeFrames)" -ForegroundColor Green
Write-Host ""
Write-Host "Frames saved to:" -ForegroundColor Yellow
Write-Host "  - frames\real\ (original video frames)" 
Write-Host "  - frames\fake\ (deepfake video frames)"
