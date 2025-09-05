# 🚀 DeepSight AI - Mobile Integration Guide

This guide shows you how to optimally integrate your existing DeepSight AI deepfake detection with your mobile app **without breaking anything that currently works**.

## 🎯 What This Integration Provides

- ✅ **Preserves all existing functionality** (Streamlit app, inference scripts, etc.)
- ✅ **Production-ready API server** for mobile app communication
- ✅ **Enhanced mobile app** with deepfake detection capabilities
- ✅ **Backward compatibility** with all existing code
- ✅ **Real-time progress tracking** and error handling
- ✅ **Optimized performance** with caching and retry logic

## 📁 New Files Added

```
├── api_server.py                          # NEW: Production API server
├── start_mobile_integration.bat           # NEW: Windows startup script
├── start_mobile_integration.sh            # NEW: Linux/Mac startup script
├── deepsight-mobile/
│   ├── services/
│   │   └── deepSightAIService.ts         # NEW: Enhanced mobile service
│   └── App.tsx                           # ENHANCED: Mobile UI with deepfake detection
└── MOBILE_INTEGRATION_GUIDE.md           # NEW: This guide
```

## 🔧 Quick Setup

### Step 1: Install Additional Dependencies

```bash
pip install flask flask-cors
```

### Step 2: Start the Integration

**Windows:**
```cmd
start_mobile_integration.bat
```

**Linux/Mac:**
```bash
chmod +x start_mobile_integration.sh
./start_mobile_integration.sh
```

### Step 3: Configure Mobile App

1. Find your computer's IP address:
   - Windows: `ipconfig`
   - Mac/Linux: `ifconfig` or `ip addr`

2. Update the mobile app server URL:
   ```typescript
   // In deepsight-mobile/services/deepSightAIService.ts
   const service = new EnhancedAIService('http://YOUR_IP_ADDRESS:5000');
   ```

3. Start the mobile app:
   ```bash
   cd deepsight-mobile
   npm install  # Only needed first time
   npm start
   ```

## 🌐 API Endpoints

### Health Check
```
GET http://your-ip:5000/health
```

### Deepfake Detection
```
POST http://your-ip:5000/api/detect
Content-Type: multipart/form-data

Body:
- video: [video file]
- max_frames: 30 (optional)
- detailed: true (optional)
```

### Analysis History
```
GET http://your-ip:5000/api/history/{session_id}
```

### Statistics
```
GET http://your-ip:5000/api/stats
```

## 📱 Mobile App Features

### Enhanced Capabilities
- **Video Upload**: Choose from gallery or record new video
- **Real-time Progress**: Live progress tracking during analysis
- **Comprehensive Results**: Detailed analysis with confidence levels
- **Error Handling**: Graceful error handling with retry options
- **Server Status**: Real-time server connectivity monitoring
- **Caching**: Smart result caching for improved performance

### UI Components
- **Modern Gradient Design**: Professional-looking interface
- **Animated Progress Bars**: Visual feedback during processing
- **Result Cards**: Clean, organized result presentation
- **Status Indicators**: Clear server and analysis status
- **Error Messages**: User-friendly error reporting

## 🔄 Backward Compatibility

### Your Existing Apps Still Work
- ✅ `streamlit run app_working.py` - Your Streamlit app works as before
- ✅ `python infer_video_final.py video.mp4` - Command line inference works
- ✅ All training scripts work unchanged
- ✅ All existing models and weights work

### What's New
- 🆕 Mobile app can now detect deepfakes
- 🆕 REST API for any client application
- 🆕 Production-ready server with proper error handling
- 🆕 Analysis history and statistics
- 🆕 Progress tracking and caching

## 🛠️ Customization

### Change API Server Port
```python
# In api_server.py, modify the default port
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Change from 5000 to 8080
```

### Adjust Analysis Parameters
```typescript
// In mobile app, modify default parameters
const result = await service.analyzeVideoForDeepfakes({
  videoUri,
  maxFrames: 50,     // Analyze more frames
  detailed: true,    // Get detailed analysis
});
```

### Add Custom Models
```python
# In api_server.py, modify model loading
def load_model():
    # Load your custom model here
    model = YourCustomModel()
    model.load_state_dict(torch.load("path/to/your/model.pth"))
```

## 📊 Performance Optimization

### Server-Side
- **GPU Acceleration**: Automatically uses CUDA if available
- **Efficient Frame Processing**: Smart frame sampling
- **Memory Management**: Automatic cleanup of temporary files
- **Concurrent Processing**: Threaded request handling

### Client-Side
- **Smart Caching**: Results cached to avoid repeated analysis
- **Retry Logic**: Automatic retry on network failures
- **Progress Tracking**: Real-time progress updates
- **Background Processing**: Non-blocking UI during analysis

## 🔧 Troubleshooting

### Common Issues

**1. Server won't start**
```bash
# Check if port is in use
netstat -an | grep 5000
# Kill existing process if needed
taskkill /f /im python.exe  # Windows
pkill -f python             # Linux/Mac
```

**2. Mobile app can't connect**
- Ensure both devices are on the same network
- Check firewall settings
- Verify IP address is correct
- Test with browser: `http://your-ip:5000/health`

**3. Model not found error**
- Ensure `weights/baseline.pth` exists
- Check file permissions
- Verify model path in `infer_video_final.py`

**4. Out of memory errors**
- Reduce `max_frames` parameter
- Close other applications
- Use CPU mode if GPU memory is insufficient

### Network Configuration

**Allow through Windows Firewall:**
```cmd
netsh advfirewall firewall add rule name="DeepSight API" dir=in action=allow protocol=TCP localport=5000
```

**Check connectivity:**
```bash
# From mobile device or another computer
curl http://YOUR_COMPUTER_IP:5000/health
```

## 🎯 Usage Examples

### Command Line Testing
```bash
# Test the API with curl
curl -X POST -F "video=@test_video.mp4" -F "max_frames=20" http://localhost:5000/api/detect

# Check server health
curl http://localhost:5000/health

# Get statistics
curl http://localhost:5000/api/stats
```

### Mobile App Usage
1. Open the app
2. Wait for "Server: online" status
3. Tap "Choose Video" or "Record Video"
4. Wait for analysis to complete
5. View comprehensive results

### Integration with Other Apps
```python
import requests

# Any Python app can use the API
def detect_deepfake(video_path):
    with open(video_path, 'rb') as f:
        response = requests.post(
            'http://localhost:5000/api/detect',
            files={'video': f},
            data={'max_frames': 30}
        )
    return response.json()
```

## 🚀 Production Deployment

### Using Docker (Optional)
```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg libopencv-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 5000
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "5000"]
```

### Cloud Deployment
- **AWS**: Use EC2 with security groups allowing port 5000
- **Google Cloud**: Use Compute Engine with firewall rules
- **Azure**: Use Virtual Machines with network security groups
- **Heroku**: Use web dynos (modify for Heroku port binding)

## 📈 Monitoring and Logs

### Server Logs
```bash
# View real-time logs
tail -f logs/api_server.log

# Check error logs
grep ERROR logs/api_server.log
```

### Performance Monitoring
```python
# Add to api_server.py for monitoring
import psutil
import GPUtil

@app.route('/api/system-stats', methods=['GET'])
def system_stats():
    return jsonify({
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'gpu_memory': GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
    })
```

## 🔒 Security Considerations

### Production Security
- Enable HTTPS with SSL certificates
- Add API key authentication
- Implement rate limiting
- Validate file uploads
- Add CORS restrictions for production

### Example Security Enhancement
```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/detect', methods=['POST'])
@require_api_key
def detect_deepfake():
    # Your existing code here
```

## 🎉 Success! You're Ready to Go

Your DeepSight AI system now has:
- ✅ **Working mobile app** with professional UI
- ✅ **Production API server** with robust error handling
- ✅ **All existing functionality preserved**
- ✅ **Optimized performance** with caching and retry logic
- ✅ **Real-time progress tracking**
- ✅ **Comprehensive result analysis**

Start the system with the provided scripts and enjoy your enhanced deepfake detection platform! 🚀
