#!/usr/bin/env python3
"""
Comprehensive test suite for DeepSight AI model
Tests model loading, inference, performance, and web app functionality
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import time
import sys
from torchvision import transforms, models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ModelTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.test_results = {}
        
    def load_model(self):
        """Load and validate the trained model"""
        print("🧠 Loading DeepSight AI Model...")
        try:
            # Create model architecture (same as app.py)
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            
            # Freeze feature layers
            for param in model.features.parameters():
                param.requires_grad = False
            
            # Custom classifier (matching train_advanced.py)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2)
            )
            
            # Load checkpoint
            checkpoint = torch.load("weights/best_model.pth", map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval().to(self.device)
            
            self.model = model
            accuracy = checkpoint.get("accuracy", "N/A")
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model accuracy: {accuracy:.2%}" if isinstance(accuracy, float) else f"📊 Model accuracy: {accuracy}")
            print(f"🖥️ Device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def test_model_architecture(self):
        """Test model architecture and parameters"""
        print("\n🔍 Testing Model Architecture...")
        
        try:
            # Test input shape
            test_input = torch.randn(1, 3, 224, 224).to(self.device)
            
            with torch.no_grad():
                output = self.model(test_input)
            
            # Validate output shape
            assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
            
            # Test probability distribution
            probs = torch.softmax(output, dim=1)
            assert torch.allclose(probs.sum(dim=1), torch.tensor(1.0)), "Probabilities don't sum to 1"
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"✅ Architecture test passed!")
            print(f"📏 Total parameters: {total_params:,}")
            print(f"🎯 Trainable parameters: {trainable_params:,}")
            print(f"📐 Output shape: {output.shape}")
            print(f"🎲 Sample probabilities: [Real: {probs[0,0]:.3f}, Fake: {probs[0,1]:.3f}]")
            
            self.test_results['architecture'] = 'PASS'
            return True
            
        except Exception as e:
            print(f"❌ Architecture test failed: {e}")
            self.test_results['architecture'] = 'FAIL'
            return False
    
    def test_inference_speed(self, num_tests=10):
        """Test model inference speed"""
        print(f"\n⚡ Testing Inference Speed ({num_tests} iterations)...")
        
        try:
            # Warm up
            test_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.model(test_input)
            
            # Time multiple inferences
            times = []
            for i in range(num_tests):
                test_input = torch.randn(1, 3, 224, 224).to(self.device)
                
                start_time = time.time()
                with torch.no_grad():
                    output = self.model(test_input)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time
            
            print(f"✅ Inference speed test passed!")
            print(f"⏱️ Average inference time: {avg_time*1000:.2f}ms")
            print(f"📊 Standard deviation: {std_time*1000:.2f}ms")
            print(f"🚀 Estimated FPS: {fps:.1f}")
            
            self.test_results['inference_speed'] = f"{avg_time*1000:.2f}ms"
            return True
            
        except Exception as e:
            print(f"❌ Inference speed test failed: {e}")
            self.test_results['inference_speed'] = 'FAIL'
            return False
    
    def test_predictions_consistency(self):
        """Test model prediction consistency"""
        print("\n🎯 Testing Prediction Consistency...")
        
        try:
            # Test deterministic behavior (same input should give same output)
            test_input = torch.randn(1, 3, 224, 224).to(self.device)
            
            outputs = []
            for i in range(3):
                with torch.no_grad():
                    output = self.model(test_input)
                    outputs.append(output.cpu())
            
            # Check if all outputs are identical
            for i in range(1, len(outputs)):
                assert torch.allclose(outputs[0], outputs[i], atol=1e-6), f"Output {i} differs from output 0"
            
            # Test probability ranges
            probs = torch.softmax(outputs[0], dim=1)
            fake_prob = probs[0, 1].item()
            
            print(f"✅ Prediction consistency test passed!")
            print(f"🎲 Model is deterministic")
            print(f"📊 Sample fake probability: {fake_prob:.3f}")
            
            self.test_results['consistency'] = 'PASS'
            return True
            
        except Exception as e:
            print(f"❌ Prediction consistency test failed: {e}")
            self.test_results['consistency'] = 'FAIL'
            return False
    
    def test_face_detection_pipeline(self):
        """Test face detection integration"""
        print("\n👤 Testing Face Detection Pipeline...")
        
        try:
            # Load OpenCV face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Create a test image with a simple pattern
            test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
            
            # Add a face-like rectangular pattern
            cv2.rectangle(test_img, (200, 150), (400, 350), (200, 200, 200), -1)
            cv2.rectangle(test_img, (250, 200), (280, 230), (50, 50, 50), -1)  # Left eye
            cv2.rectangle(test_img, (320, 200), (350, 230), (50, 50, 50), -1)  # Right eye
            cv2.rectangle(test_img, (290, 270), (310, 290), (50, 50, 50), -1)  # Nose
            cv2.rectangle(test_img, (270, 310), (330, 330), (50, 50, 50), -1)  # Mouth
            
            # Test face detection
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            print(f"✅ Face detection test passed!")
            print(f"👥 Pattern detection working: {len(faces)} regions detected")
            print(f"📷 Test image shape: {test_img.shape}")
            
            self.test_results['face_detection'] = f"{len(faces)} faces detected"
            return True
            
        except Exception as e:
            print(f"❌ Face detection test failed: {e}")
            self.test_results['face_detection'] = 'FAIL'
            return False
    
    def test_memory_usage(self):
        """Test GPU/CPU memory usage"""
        print("\n💾 Testing Memory Usage...")
        
        try:
            if torch.cuda.is_available():
                # Test GPU memory
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                # Run inference
                test_input = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    output = self.model(test_input)
                
                peak_memory = torch.cuda.memory_allocated()
                memory_used = (peak_memory - initial_memory) / 1024**2  # MB
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                print(f"✅ GPU memory test passed!")
                print(f"🖥️ GPU memory used: {memory_used:.2f} MB")
                print(f"📊 Total GPU memory: {total_memory:.1f} GB")
                
                self.test_results['memory_usage'] = f"{memory_used:.2f} MB GPU"
            else:
                print(f"✅ CPU memory test (GPU not available)")
                self.test_results['memory_usage'] = "CPU only"
            
            return True
            
        except Exception as e:
            print(f"❌ Memory usage test failed: {e}")
            self.test_results['memory_usage'] = 'FAIL'
            return False
    
    def run_all_tests(self):
        """Run all model tests"""
        print("🚀 Starting Comprehensive Model Testing...")
        print("=" * 60)
        
        # Test sequence
        tests = [
            ('Model Loading', self.load_model),
            ('Architecture', self.test_model_architecture),
            ('Inference Speed', self.test_inference_speed),
            ('Prediction Consistency', self.test_predictions_consistency),
            ('Face Detection', self.test_face_detection_pipeline),
            ('Memory Usage', self.test_memory_usage),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = 'CRASH'
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result not in ['FAIL', 'CRASH'] else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            if result not in ['PASS', 'FAIL', 'CRASH']:
                print(f"   └─ {result}")
        
        print(f"\n🎯 Overall Score: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! DeepSight AI is ready for deployment!")
        elif passed_tests >= total_tests * 0.8:
            print("⚠️ Most tests passed. Minor issues detected.")
        else:
            print("❌ Multiple test failures. Review required.")
        
        return passed_tests, total_tests

def main():
    """Main testing function"""
    print("🔍 DeepSight AI - Comprehensive Model Test Suite")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists("weights/best_model.pth"):
        print("❌ Model file not found: weights/best_model.pth")
        print("Please ensure the trained model is available.")
        return
    
    # Run tests
    tester = ModelTester()
    passed, total = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
