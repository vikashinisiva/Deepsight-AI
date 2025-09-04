# Frame Extraction Script for FaceForensics++ Dataset
# This script extracts frames from vide} else {
    Write-Host ""
    Write-Host "Please install FFmpeg first, then run this script again." -ForegroundColor Yellow
}at 1 fps for training deepfake detection models

# Step 1: Install FFmpeg (run this first)
Write-Host "=== STEP 1: INSTALL FFMPEG ===" -ForegroundColor Green
Write-Host "You need to install FFmpeg first. Choose one of these methods:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Method 1 - Using Chocolatey (recommended):" -ForegroundColor Cyan
Write-Host "  1. Run PowerShell as Administrator" 
Write-Host "  2. Run: choco install ffmpeg"
Write-Host ""
Write-Host "Method 2 - Manual Installation:" -ForegroundColor Cyan
Write-Host "  1. Download FFmpeg from https://ffmpeg.org/download.html"
Write-Host "  2. Extract to C:\ffmpeg"
Write-Host "  3. Add C:\ffmpeg\bin to your PATH environment variable"
Write-Host ""
Write-Host "Method 3 - Using winget:" -ForegroundColor Cyan
Write-Host "  Run: winget install --id=Gyan.FFmpeg -e"
Write-Host ""

# Check if FFmpeg is available
try {
    $ffmpegVersion = ffmpeg -version 2>$null
    if ($ffmpegVersion) {
        Write-Host "✓ FFmpeg is installed and ready!" -ForegroundColor Green
        $ffmpegInstalled = $true
    } else {
        $ffmpegInstalled = $false
    }
} catch {
    Write-Host "✗ FFmpeg not found. Please install it first using one of the methods above." -ForegroundColor Red
    $ffmpegInstalled = $false
}

if ($ffmpegInstalled) {
    Write-Host ""
    Write-Host "=== STEP 2: EXTRACT FRAMES ===" -ForegroundColor Green
    
    # Create directories
    Write-Host "Creating frame directories..."
    New-Item -ItemType Directory -Path "frames\real" -Force | Out-Null
    New-Item -ItemType Directory -Path "frames\fake" -Force | Out-Null
    
    # Extract frames from REAL videos
    Write-Host ""
    Write-Host "Extracting frames from REAL videos..." -ForegroundColor Cyan
    $realVideos = Get-ChildItem "ffpp_data\real_videos\*.mp4"
    $totalReal = $realVideos.Count
    $currentReal = 0
    
    foreach ($video in $realVideos) {
        $currentReal++
        $bn = [IO.Path]::GetFileNameWithoutExtension($video.FullName)
        Write-Progress -Activity "Processing Real Videos" -Status "Processing $bn" -PercentComplete (($currentReal / $totalReal) * 100)
        
        # Extract frames at 1 fps
        & ffmpeg -loglevel error -i $video.FullName -r 1 "frames\real\$bn`_%03d.jpg" 2>$null
    }
    Write-Progress -Activity "Processing Real Videos" -Completed
    
    # Extract frames from FAKE videos
    Write-Host ""
    Write-Host "Extracting frames from FAKE videos..." -ForegroundColor Cyan
    $fakeVideos = Get-ChildItem "ffpp_data\fake_videos\*.mp4"
    $totalFake = $fakeVideos.Count
    $currentFake = 0
    
    foreach ($video in $fakeVideos) {
        $currentFake++
        $bn = [IO.Path]::GetFileNameWithoutExtension($video.FullName)
        Write-Progress -Activity "Processing Fake Videos" -Status "Processing $bn" -PercentComplete (($currentFake / $totalFake) * 100)
        
        # Extract frames at 1 fps
        & ffmpeg -loglevel error -i $video.FullName -r 1 "frames\fake\$bn`_%03d.jpg" 2>$null
    }
    Write-Progress -Activity "Processing Fake Videos" -Completed
    
    Write-Host ""
    Write-Host "=== EXTRACTION COMPLETE ===" -ForegroundColor Green
    
    # Count extracted frames
    $realFrames = (Get-ChildItem "frames\real\*.jpg" -ErrorAction SilentlyContinue).Count
    $fakeFrames = (Get-ChildItem "frames\fake\*.jpg" -ErrorAction SilentlyContinue).Count
    
    Write-Host "Real frames extracted: $realFrames" -ForegroundColor Green
    Write-Host "Fake frames extracted: $fakeFrames" -ForegroundColor Green
    Write-Host "Total frames extracted: $($realFrames + $fakeFrames)" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "Frame extraction completed successfully!" -ForegroundColor Green
    Write-Host "Frames are saved in:" -ForegroundColor Yellow
    Write-Host "  - frames\real\ (contains frames from original videos)"
    Write-Host "  - frames\fake\ (contains frames from deepfake videos)"
    
} else {
    Write-Host ""
    Write-Host "Please install FFmpeg first, then run this script again." -ForegroundColor Yellow
}
