@echo off
echo.
echo 🎥 DeepSight AI Visual Demo Launcher
echo ====================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if visual demo app exists
if not exist "visual_demo_app.py" (
    echo ❌ visual_demo_app.py not found in current directory
    echo 📁 Please ensure you're in the DeepSight AI directory
    pause
    exit /b 1
)

echo ✅ Python found
echo 🔍 Launching visual demo...
echo.
echo 🌐 Your browser will open automatically
echo 🎬 Experience the complete deepfake detection pipeline!
echo.
echo 💡 Tips:
echo   - Try the demo videos first
echo   - Upload your own videos for testing
echo   - Watch the animated pipeline stages
echo   - Explore the AI heatmaps
echo.
echo ⏹️  Press Ctrl+C to stop the demo
echo ====================================
echo.

REM Launch the visual demo
python launch_visual_demo.py

pause
