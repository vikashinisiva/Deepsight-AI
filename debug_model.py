#!/usr/bin/env python3
"""
Debug Model Predictions
Check what the model is actually outputting
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models

def load_model():
    """Load the model and check its outputs"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load EfficientNet-B3 model (same as in app.py)
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
        nn.Linear(128, 2)  # 2 classes: [0, 1]
    )
    model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    print(f"Model output classes: 2 (index 0 and 1)")
    
    return model, device

def test_predictions():
    """Test what the model predicts"""
    model, device = load_model()
    
    print("\n🧪 Testing Model Predictions")
    print("="*40)
    
    # Test with random inputs
    for i in range(5):
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            logits = model(dummy_input)
            probabilities = torch.softmax(logits, dim=1)
            
            # Current app logic
            p_fake_current = probabilities[0, 1].item()  # Index 1 for fake
            prediction_current = "FAKE" if p_fake_current > 0.5 else "REAL"
            
            # Alternative logic (in case labels are swapped)
            p_fake_alt = probabilities[0, 0].item()  # Index 0 for fake
            prediction_alt = "FAKE" if p_fake_alt > 0.5 else "REAL"
            
            print(f"\nTest {i+1}:")
            print(f"  Raw logits: {logits[0].cpu().numpy()}")
            print(f"  Probabilities: {probabilities[0].cpu().numpy()}")
            print(f"  Current logic (index 1 = fake): {prediction_current} ({p_fake_current:.3f})")
            print(f"  Alt logic (index 0 = fake): {prediction_alt} ({p_fake_alt:.3f})")

def main():
    print("🔍 Model Prediction Debug")
    print("Checking if the model labels are correct...")
    test_predictions()
    
    print("\n💡 Analysis:")
    print("If most predictions show similar values for both indices,")
    print("the model might not be properly trained or the labels might be swapped.")
    print("\nTo fix fake videos showing as real:")
    print("1. Check if index 0 should be fake instead of index 1")
    print("2. Verify the model was trained with correct labels")
    print("3. Test with known fake/real videos to confirm")

if __name__ == "__main__":
    main()
