#!/usr/bin/env python3
"""
Quick Project Validation and Health Check
Final validation script for DeepSight AI project
"""

import os
import sys
import json
import time
from pathlib import Path

def check_critical_files():
    """Check if all critical files exist"""
    print("📁 Checking Critical Files...")
    
    critical_files = [
        'app.py',
        'train_advanced.py', 
        'emotional_intelligence_ai.py',
        'grad_cam.py',
        'comprehensive_test.py',
        'weights/best_model.pth'
    ]
    
    missing = []
    existing = []
    
    for file in critical_files:
        if os.path.exists(file):
            existing.append(file)
            print(f"✅ {file}")
        else:
            missing.append(file)
            print(f"❌ {file}")
    
    return len(existing), len(missing)

def check_dependencies():
    """Check key dependencies"""
    print("\n📦 Checking Key Dependencies...")
    
    dependencies = [
        'torch', 'torchvision', 'numpy', 'cv2', 'streamlit',
        'plotly', 'PIL', 'sklearn', 'matplotlib', 'tqdm'
    ]
    
    installed = []
    missing = []
    
    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'PIL':
                from PIL import Image
            elif dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            installed.append(dep)
            print(f"✅ {dep}")
        except ImportError:
            missing.append(dep)
            print(f"❌ {dep}")
    
    return len(installed), len(missing)

def test_model_loading():
    """Test if the model can be loaded"""
    print("\n🧠 Testing Model Loading...")
    
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
        
        # Create model architecture
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
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
        
        # Try to load checkpoint
        if os.path.exists('weights/best_model.pth'):
            checkpoint = torch.load('weights/best_model.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            accuracy = checkpoint.get('accuracy', 'Unknown')
            print(f"✅ Model loaded successfully! Accuracy: {accuracy}")
            return True
        else:
            print("❌ Model file not found")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def run_app_syntax_check():
    """Check if main app can be imported"""
    print("\n🚀 Testing App Syntax...")
    
    try:
        # Try to compile the main app file
        with open('app.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, 'app.py', 'exec')
        print("✅ App syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking app.py: {e}")
        return False

def generate_health_report():
    """Generate final health report"""
    print("\n📊 Generating Health Report...")
    
    # Run all checks
    files_ok, files_missing = check_critical_files()
    deps_ok, deps_missing = check_dependencies()
    model_ok = test_model_loading()
    app_ok = run_app_syntax_check()
    
    # Calculate scores
    total_checks = 4
    passed_checks = sum([
        files_missing == 0,
        deps_missing <= 2,  # Allow some missing deps
        model_ok,
        app_ok
    ])
    
    health_score = passed_checks / total_checks
    
    # Generate grade
    if health_score >= 0.9:
        grade = "A"
        status = "🎉 EXCELLENT"
    elif health_score >= 0.75:
        grade = "B"  
        status = "✅ GOOD"
    elif health_score >= 0.5:
        grade = "C"
        status = "⚠️ NEEDS IMPROVEMENT"
    else:
        grade = "D"
        status = "❌ CRITICAL ISSUES"
    
    # Print report
    print("=" * 60)
    print("📋 DEEPSIGHT AI PROJECT HEALTH REPORT")
    print("=" * 60)
    print(f"📊 Overall Score: {health_score:.1%}")
    print(f"🎓 Grade: {grade}")
    print(f"🎯 Status: {status}")
    print()
    print(f"📁 Critical Files: {files_ok}/{files_ok + files_missing}")
    print(f"📦 Dependencies: {deps_ok}/{deps_ok + deps_missing}")
    print(f"🧠 Model Loading: {'✅' if model_ok else '❌'}")
    print(f"🚀 App Syntax: {'✅' if app_ok else '❌'}")
    print()
    
    if health_score >= 0.75:
        print("🚀 READY FOR USE!")
        print("   • Run: streamlit run app.py")
        print("   • Test: python comprehensive_test.py")
        print("   • Train: python train_advanced.py")
    else:
        print("⚠️ ISSUES DETECTED!")
        print("   • Check missing files")
        print("   • Install missing dependencies")
        print("   • Verify model weights")
    
    return {
        'score': health_score,
        'grade': grade,
        'files_ok': files_ok,
        'files_missing': files_missing,
        'deps_ok': deps_ok,
        'deps_missing': deps_missing,
        'model_ok': model_ok,
        'app_ok': app_ok
    }

def main():
    """Main validation function"""
    print("🔍 DeepSight AI - Quick Project Validation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run validation
    report = generate_health_report()
    
    end_time = time.time()
    
    print(f"\n⏱️ Validation completed in {end_time - start_time:.1f} seconds")
    
    # Save report
    with open('health_report.json', 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'report': report
        }, f, indent=2)
    
    print("📄 Report saved to: health_report.json")
    
    return report['score'] >= 0.75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
