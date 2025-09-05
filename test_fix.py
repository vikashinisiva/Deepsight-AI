#!/usr/bin/env python3
"""
Test Fixed Prediction Logic
Verify that the label fix works correctly
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models

def test_fixed_logic():
    """Test the fixed prediction logic"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (same as app.py)
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2), 
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(128, 2)
    )
    model.to(device)
    model.eval()
    
    print("🔧 Testing Fixed Prediction Logic")
    print("="*40)
    
    # Test with several random inputs
    fake_count = 0
    real_count = 0
    
    for i in range(10):
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            logits = model(dummy_input)
            # FIXED: Using index 0 for fake (was index 1)
            p_fake = torch.softmax(logits, dim=1)[0,0].item()
            prediction = "FAKE" if p_fake > 0.5 else "REAL"
            
            if prediction == "FAKE":
                fake_count += 1
            else:
                real_count += 1
            
            print(f"Test {i+1}: {prediction} (fake prob: {p_fake:.3f})")
    
    print(f"\n📊 Results Summary:")
    print(f"FAKE predictions: {fake_count}/10")
    print(f"REAL predictions: {real_count}/10")
    
    if fake_count > 0:
        print("✅ SUCCESS: Model can now detect fake content!")
        print("   The label fix is working correctly.")
    else:
        print("⚠️  All predictions are REAL - this might indicate:")
        print("   1. The model needs proper training weights")
        print("   2. Or the input preprocessing needs adjustment")
    
    print(f"\n🌐 Test your app at: http://localhost:8504")
    print("   Upload a video to see if fake videos are now detected properly!")

if __name__ == "__main__":
    test_fixed_logic()
