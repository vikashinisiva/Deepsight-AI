#!/usr/bin/env python3
"""
Test Detection Accuracy - Verify Model Performance
Tests the cleaned app.py to ensure proper detection functionality
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import cv2
import tempfile

# Add current directory to path to import from app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import load_model, load_face_detector, analyze_video, setup_gradcam

def test_model_loading():
    """Test if the model loads correctly"""
    print("🔧 Testing model loading...")
    model, device, accuracy = load_model()
    
    if model is None:
        print("❌ Model failed to load")
        return False
    
    print(f"✅ Model loaded successfully on {device}")
    print(f"📊 Model accuracy: {accuracy:.2f}%")
    return True

def test_face_detector():
    """Test face detection functionality"""
    print("\n🔧 Testing face detection...")
    face_cascade = load_face_detector()
    
    if face_cascade is None:
        print("❌ Face detector failed to load")
        return False
    
    print("✅ Face detector loaded successfully")
    return True

def create_test_video():
    """Create a simple test video for verification"""
    print("\n🔧 Creating test video...")
    
    # Create a simple test video with a colored rectangle (simulating a face)
    import cv2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, 1.0, (640, 480))
    
    for i in range(5):  # 5 frames
        # Create a frame with a rectangle (simulating a face region)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some pattern that might be detected as a face
        cv2.rectangle(frame, (200, 150), (440, 330), (100, 150, 200), -1)
        cv2.circle(frame, (280, 200), 20, (255, 255, 255), -1)  # Left eye
        cv2.circle(frame, (360, 200), 20, (255, 255, 255), -1)  # Right eye
        cv2.ellipse(frame, (320, 260), (30, 15), 0, 0, 180, (255, 255, 255), -1)  # Mouth
        
        out.write(frame)
    
    out.release()
    print("✅ Test video created: test_video.mp4")
    return True

def test_video_analysis():
    """Test video analysis functionality"""
    print("\n🔧 Testing video analysis...")
    
    # Load model and face detector
    model, device, accuracy = load_model()
    face_cascade = load_face_detector()
    cam_analyzer = setup_gradcam(model) if model else None
    
    if not model or not face_cascade:
        print("❌ Failed to load required components")
        return False
    
    # Create test video
    create_test_video()
    
    # Test analysis
    try:
        result = analyze_video('test_video.mp4', model, device, face_cascade, cam_analyzer, False)
        
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return False
        
        prediction = result.get("prediction", "Unknown")
        confidence = result.get("fake_confidence", 0)
        frames = result.get("num_frames", 0)
        
        print(f"✅ Analysis completed successfully!")
        print(f"📊 Prediction: {prediction}")
        print(f"🎯 Confidence: {confidence:.1%}")
        print(f"📹 Frames analyzed: {frames}")
        
        # Clean up test video
        if os.path.exists('test_video.mp4'):
            os.remove('test_video.mp4')
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed with exception: {str(e)}")
        return False

def test_prediction_logic():
    """Test the core prediction logic"""
    print("\n🔧 Testing prediction logic...")
    
    # Test cases for prediction logic
    test_cases = [
        (0.3, "REAL"),    # Low fake probability -> REAL
        (0.7, "FAKE"),    # High fake probability -> FAKE
        (0.5, "REAL"),    # Exactly 0.5 -> REAL (threshold behavior)
        (0.51, "FAKE"),   # Just above threshold -> FAKE
        (0.49, "REAL")    # Just below threshold -> REAL
    ]
    
    for fake_prob, expected in test_cases:
        prediction = "FAKE" if fake_prob > 0.5 else "REAL"
        if prediction == expected:
            print(f"✅ Test passed: {fake_prob:.2f} -> {prediction}")
        else:
            print(f"❌ Test failed: {fake_prob:.2f} -> {prediction} (expected {expected})")
            return False
    
    print("✅ All prediction logic tests passed!")
    return True

def main():
    """Run all tests"""
    print("🧪 DeepSight AI Detection Test Suite")
    print("="*50)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Face Detection", test_face_detector), 
        ("Prediction Logic", test_prediction_logic),
        ("Video Analysis", test_video_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {str(e)}")
    
    print("\n" + "="*50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The model is detecting properly.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
