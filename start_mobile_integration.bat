@echo off
REM DeepSight AI - Windows Startup Script for Mobile Integration
REM This script preserves your existing working versions while adding mobile integration

echo 🚀 Starting DeepSight AI - Mobile Integration Mode
echo ==================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check for required packages
echo 📦 Checking Python dependencies...
python -c "import torch, cv2, flask, torchvision" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Missing dependencies. Installing...
    pip install torch torchvision opencv-python flask flask-cors tqdm numpy pillow
)

REM Check if the model weights exist
if not exist "weights\baseline.pth" (
    echo ❌ Model weights not found at weights\baseline.pth
    echo    Please ensure your trained model is available.
    pause
    exit /b 1
)

REM Start the API server
echo 🌐 Starting DeepSight AI API Server...
start "DeepSight API Server" python api_server.py --host 0.0.0.0 --port 5000

REM Wait for server to start
timeout /t 5 /nobreak >nul

REM Check if server is running
curl -s http://localhost:5000/health >nul 2>&1
if not errorlevel 1 (
    echo ✅ API Server is running at http://localhost:5000
    echo.
    echo 📱 Mobile App Integration Ready!
    echo =================================
    echo 🔗 API Endpoint: http://YOUR_IP:5000/api/detect
    echo 📊 Health Check: http://YOUR_IP:5000/health
    echo 📈 Statistics: http://YOUR_IP:5000/api/stats
    echo.
    echo 🏠 Your existing Streamlit app is still available:
    echo    Run: streamlit run app_working.py
    echo.
    echo 📱 To use mobile app:
    echo    1. Update mobile app with your computer's IP address
    echo    2. Make sure both devices are on same network
    echo    3. Start mobile app: cd deepsight-mobile ^&^& npm start
    echo.
    echo 💡 Get your computer's IP: ipconfig
    echo.
    echo Press any key to open mobile app directory...
    pause >nul
    
    REM Open mobile app directory
    cd deepsight-mobile
    echo 📱 Starting mobile app development server...
    npm start
    
) else (
    echo ❌ Failed to start API server
    pause
)

pause
