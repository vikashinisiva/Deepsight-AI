#!/usr/bin/env python3
"""
Verification test for the label swap fix
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms, models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def load_model():
    """Load the model exactly as in app.py"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    checkpoint = torch.load("weights/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    
    return model, device

def test_fixed_prediction_logic():
    """Test the corrected prediction logic"""
    print("🧪 Testing Fixed Prediction Logic")
    print("=" * 50)
    
    model, device = load_model()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create a natural face-like pattern (should be REAL)
    natural_face = np.ones((224, 224, 3), dtype=np.uint8)
    # Skin color
    natural_face[:, :] = [220, 180, 150]
    # Add face features
    cv2.circle(natural_face, (75, 90), 8, (50, 50, 50), -1)   # Left eye
    cv2.circle(natural_face, (149, 90), 8, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(natural_face, (112, 120), (8, 12), 0, 0, 180, (100, 50, 50), -1)  # Nose
    cv2.ellipse(natural_face, (112, 160), (25, 8), 0, 0, 180, (150, 100, 100), -1)  # Mouth
    
    # Test the pattern
    pil_img = Image.fromarray(natural_face)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()
    
    # Apply the FIXED prediction logic from app.py
    avg_fake_prob = fake_prob  # In this case, single frame
    prediction_fixed = "REAL" if avg_fake_prob > 0.5 else "FAKE"
    
    # Compare with old logic
    prediction_old = "FAKE" if avg_fake_prob > 0.5 else "REAL"
    
    print(f"Natural face pattern test:")
    print(f"  Real probability: {real_prob:.3f}")
    print(f"  Fake probability: {fake_prob:.3f}")
    print(f"  OLD logic prediction: {prediction_old}")
    print(f"  FIXED logic prediction: {prediction_fixed}")
    
    if prediction_fixed == "REAL":
        print("✅ SUCCESS: Natural pattern now correctly classified as REAL!")
    else:
        print("❌ ISSUE: Natural pattern still classified as FAKE")
    
    # Test with artificial pattern (should be FAKE)
    print(f"\n" + "-" * 50)
    artificial_img = np.zeros((224, 224, 3), dtype=np.uint8)
    artificial_img[50:174, 50:174] = [255, 0, 255]  # Magenta square
    
    pil_img = Image.fromarray(artificial_img)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()
    
    avg_fake_prob = fake_prob
    prediction_fixed = "REAL" if avg_fake_prob > 0.5 else "FAKE"
    prediction_old = "FAKE" if avg_fake_prob > 0.5 else "REAL"
    
    print(f"Artificial pattern test:")
    print(f"  Real probability: {real_prob:.3f}")
    print(f"  Fake probability: {fake_prob:.3f}")
    print(f"  OLD logic prediction: {prediction_old}")
    print(f"  FIXED logic prediction: {prediction_fixed}")
    
    if prediction_fixed == "FAKE":
        print("✅ SUCCESS: Artificial pattern correctly classified as FAKE!")
    else:
        print("❌ ISSUE: Artificial pattern incorrectly classified as REAL")

if __name__ == "__main__":
    test_fixed_prediction_logic()
