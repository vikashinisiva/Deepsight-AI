#!/usr/bin/env python3
"""
DeepSight AI Visual Demo Launcher
Launch the visual pipeline demo with a single command
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'torch',
        'torchvision', 
        'opencv-python',
        'numpy',
        'plotly',
        'matplotlib',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"📦 Install them with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_model():
    """Check if the trained model exists"""
    model_path = "weights/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        print("🏋️ Please train the model first by running: python train_enhanced.py")
        return False
    return True

def check_grad_cam():
    """Check if grad_cam.py exists"""
    grad_cam_path = "grad_cam.py"
    if not os.path.exists(grad_cam_path):
        print(f"❌ grad_cam.py not found")
        print("📁 Please ensure grad_cam.py is in the current directory")
        return False
    return True

def main():
    print("🎥 DeepSight AI Visual Demo Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies found")
    
    # Check model
    print("🔍 Checking model...")
    if not check_model():
        sys.exit(1)
    print("✅ Model found")
    
    # Check grad_cam
    print("🔍 Checking Grad-CAM module...")
    if not check_grad_cam():
        sys.exit(1)
    print("✅ Grad-CAM module found")
    
    # Launch the visual demo
    print("\n🚀 Launching Visual Demo Pipeline...")
    print("📱 Open your browser to the URL shown below")
    print("🎬 Experience the complete deepfake detection pipeline!")
    print("\n" + "=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "visual_demo_app.py",
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
