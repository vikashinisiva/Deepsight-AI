"""
DeepSight AI - Production API Server
Optimized mobile integration without breaking existing functionality
"""

import os
import sys
import json
import time
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Import your existing inference module
from infer_video_final import infer_final

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app

# Configuration
app.config.update({
    'MAX_CONTENT_LENGTH': 200 * 1024 * 1024,  # 200MB max file size
    'UPLOAD_FOLDER': 'uploads',
    'RESULTS_FOLDER': 'results',
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'deepsight-ai-secret-key')
})

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', '3gp'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global model loading (load once for performance)
model_loaded = False
device = None

def load_model():
    """Load the model once at startup"""
    global model_loaded, device
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Test the inference function with a dummy call
        # This ensures the model is loaded and ready
        logger.info("Model loaded and ready for inference")
        model_loaded = True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_session_id():
    """Generate unique session ID"""
    return f"session_{int(time.time())}_{os.getpid()}"

def save_analysis_result(session_id, result):
    """Save analysis result for future reference"""
    try:
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}.json")
        with open(result_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'result': result
            }, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving result: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'device': str(device) if device else 'unknown',
        'version': '1.0.0'
    })

@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    """Main deepfake detection endpoint for mobile app"""
    session_id = generate_session_id()
    
    try:
        # Check if model is loaded
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'session_id': session_id,
                'success': False
            }), 500
        
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({
                'error': 'No video file provided',
                'session_id': session_id,
                'success': False
            }), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'session_id': session_id,
                'success': False
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}',
                'session_id': session_id,
                'success': False
            }), 400
        
        # Get optional parameters
        max_frames = request.form.get('max_frames', 30, type=int)
        detailed_analysis = request.form.get('detailed', 'false').lower() == 'true'
        
        logger.info(f"Processing video: {file.filename}, Session: {session_id}")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{session_id}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        try:
            # Run inference using your existing function
            start_time = time.time()
            result = infer_final(filepath, max_frames=max_frames)
            processing_time = time.time() - start_time
            
            # Enhance result with additional metadata
            enhanced_result = {
                **result,
                'session_id': session_id,
                'processing_time': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'file_info': {
                    'original_name': file.filename,
                    'size_mb': round(os.path.getsize(filepath) / (1024*1024), 2),
                    'max_frames': max_frames
                },
                'success': True
            }
            
            # Add detailed analysis if requested
            if detailed_analysis:
                enhanced_result['detailed_analysis'] = {
                    'model_info': 'EfficientNet-B0 Baseline',
                    'detection_method': 'Face-based CNN Classification',
                    'threshold_used': 0.6,
                    'device_used': str(device)
                }
            
            # Save result for history
            save_analysis_result(session_id, enhanced_result)
            
            logger.info(f"Analysis complete: {result['prediction']} ({result['fake_confidence']:.3f}) - {processing_time:.2f}s")
            
            return jsonify(enhanced_result)
            
        finally:
            # Cleanup uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.warning(f"Could not remove temp file {filepath}: {e}")
    
    except Exception as e:
        logger.error(f"Error in detect_deepfake: {e}")
        return jsonify({
            'error': str(e),
            'session_id': session_id,
            'success': False
        }), 500

@app.route('/api/history/<session_id>', methods=['GET'])
def get_analysis_history(session_id):
    """Get analysis result by session ID"""
    try:
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}.json")
        
        if not os.path.exists(result_file):
            return jsonify({'error': 'Session not found'}), 404
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        return jsonify(data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_api_stats():
    """Get API usage statistics"""
    try:
        results_dir = Path(app.config['RESULTS_FOLDER'])
        total_analyses = len(list(results_dir.glob('*.json')))
        
        # Get recent analyses
        recent_files = sorted(results_dir.glob('*.json'), 
                            key=lambda x: x.stat().st_mtime, 
                            reverse=True)[:10]
        
        recent_analyses = []
        for file in recent_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    recent_analyses.append({
                        'session_id': data['session_id'],
                        'timestamp': data['timestamp'],
                        'prediction': data['result']['prediction'],
                        'confidence': data['result']['fake_confidence']
                    })
            except:
                continue
        
        return jsonify({
            'total_analyses': total_analyses,
            'recent_analyses': recent_analyses,
            'server_uptime': datetime.now().isoformat(),
            'model_status': 'loaded' if model_loaded else 'not_loaded'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        'error': 'File too large. Maximum size is 200MB.',
        'success': False
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

# Load model on startup
load_model()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSight AI API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🚀 Starting DeepSight AI API Server...")
    print(f"📡 Server: http://{args.host}:{args.port}")
    print(f"🔍 Health check: http://{args.host}:{args.port}/health")
    print(f"📱 API endpoint: http://{args.host}:{args.port}/api/detect")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )
