#!/usr/bin/env python3
"""
Web App Functionality Test
Tests if the Streamlit app is running correctly
"""

import requests
import time
import sys

def test_web_app():
    """Test if the Streamlit web app is accessible"""
    print("🌐 Testing Web App Accessibility...")
    
    try:
        # Test if the app is running on localhost:8502
        response = requests.get("http://localhost:8502", timeout=10)
        
        if response.status_code == 200:
            print("✅ Web app is accessible!")
            print(f"📊 Status code: {response.status_code}")
            print(f"📏 Response size: {len(response.content)} bytes")
            
            # Check if key elements are present
            content = response.text.lower()
            
            checks = [
                ("deepsight ai", "DeepSight AI branding"),
                ("upload", "Upload functionality"),
                ("analyze", "Analysis functionality"), 
                ("efficientnet", "Model information"),
                ("grad-cam", "Grad-CAM feature"),
            ]
            
            print("\n🔍 Checking web app content...")
            for keyword, description in checks:
                if keyword in content:
                    print(f"✅ {description}: Found")
                else:
                    print(f"⚠️ {description}: Not found")
            
            return True
            
        else:
            print(f"❌ Web app returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Web app is not accessible. Is Streamlit running?")
        print("💡 Try running: streamlit run app.py")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ Web app request timed out")
        return False
        
    except Exception as e:
        print(f"❌ Web app test failed: {e}")
        return False

def main():
    """Main web app testing function"""
    print("🔍 DeepSight AI - Web App Functionality Test")
    print("=" * 50)
    
    success = test_web_app()
    
    if success:
        print("\n🎉 Web app test PASSED!")
        print("🚀 DeepSight AI web interface is working correctly!")
    else:
        print("\n❌ Web app test FAILED!")
        print("🔧 Please check if the Streamlit app is running")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
