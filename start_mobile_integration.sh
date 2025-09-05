#!/bin/bash

# DeepSight AI - Optimized Startup Script
# This script preserves your existing working versions while adding mobile integration

echo "🚀 Starting DeepSight AI - Mobile Integration Mode"
echo "=================================================="

# Check if Python environment is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ first."
    exit 1
fi

# Check for required packages
echo "📦 Checking Python dependencies..."
python -c "import torch, cv2, flask, torchvision" 2>/dev/null || {
    echo "⚠️  Missing dependencies. Installing..."
    pip install torch torchvision opencv-python flask flask-cors tqdm numpy pillow
}

# Check if the model weights exist
if [ ! -f "weights/baseline.pth" ]; then
    echo "❌ Model weights not found at weights/baseline.pth"
    echo "   Please ensure your trained model is available."
    exit 1
fi

# Start the API server in background
echo "🌐 Starting DeepSight AI API Server..."
python api_server.py --host 0.0.0.0 --port 5000 &
API_PID=$!

# Wait for server to start
sleep 5

# Check if server is running
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✅ API Server is running at http://localhost:5000"
    
    # Show server information
    echo ""
    echo "📱 Mobile App Integration Ready!"
    echo "================================="
    echo "🔗 API Endpoint: http://YOUR_IP:5000/api/detect"
    echo "📊 Health Check: http://YOUR_IP:5000/health"
    echo "📈 Statistics: http://YOUR_IP:5000/api/stats"
    echo ""
    echo "🏠 Your existing Streamlit app is still available:"
    echo "   Run: streamlit run app_working.py"
    echo ""
    echo "📱 To use mobile app:"
    echo "   1. Update mobile app with your computer's IP address"
    echo "   2. Make sure both devices are on same network"
    echo "   3. Start mobile app: cd deepsight-mobile && npm start"
    echo ""
    echo "Press Ctrl+C to stop the API server"
    echo "🔄 Server PID: $API_PID"
    
    # Keep script running
    wait $API_PID
    
else
    echo "❌ Failed to start API server"
    kill $API_PID 2>/dev/null
    exit 1
fi
