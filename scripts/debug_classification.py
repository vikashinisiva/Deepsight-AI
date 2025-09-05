#!/usr/bin/env python3
"""
Debug script to investigate why real videos are being classified as fake
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
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
    
    return model, device, checkpoint

def analyze_model_bias():
    """Analyze model predictions on different types of inputs"""
    print("🔍 Analyzing Model Bias and Prediction Patterns...")
    print("=" * 60)
    
    model, device, checkpoint = load_model()
    
    print(f"📊 Model accuracy from checkpoint: {checkpoint.get('accuracy', 'N/A')}")
    print(f"🖥️ Device: {device}")
    
    # Test with different input patterns
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_cases = []
    
    # 1. Random noise (should be neutral)
    random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_cases.append(("Random Noise", random_img))
    
    # 2. Natural-looking patterns (should lean towards real)
    natural_img = np.ones((224, 224, 3), dtype=np.uint8)
    natural_img[:, :, 0] = 120  # Skin-like color
    natural_img[:, :, 1] = 100
    natural_img[:, :, 2] = 80
    # Add some natural variation
    noise = np.random.normal(0, 20, (224, 224, 3))
    natural_img = np.clip(natural_img.astype(float) + noise, 0, 255).astype(np.uint8)
    test_cases.append(("Natural Pattern", natural_img))
    
    # 3. High contrast artificial pattern (should lean towards fake)
    artificial_img = np.zeros((224, 224, 3), dtype=np.uint8)
    artificial_img[50:174, 50:174] = [255, 0, 255]  # Magenta square
    test_cases.append(("Artificial Pattern", artificial_img))
    
    # 4. Gaussian distributed image
    gaussian_img = np.random.normal(128, 50, (224, 224, 3))
    gaussian_img = np.clip(gaussian_img, 0, 255).astype(np.uint8)
    test_cases.append(("Gaussian Distribution", gaussian_img))
    
    # 5. Black image
    black_img = np.zeros((224, 224, 3), dtype=np.uint8)
    test_cases.append(("Black Image", black_img))
    
    # 6. White image
    white_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    test_cases.append(("White Image", white_img))
    
    print(f"\n🧪 Testing {len(test_cases)} different input patterns:")
    print("-" * 60)
    
    results = []
    
    for name, img in test_cases:
        # Convert to tensor
        pil_img = Image.fromarray(img)
        tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            real_prob = probs[0, 0].item()
            fake_prob = probs[0, 1].item()
        
        results.append({
            'name': name,
            'real_prob': real_prob,
            'fake_prob': fake_prob,
            'prediction': 'REAL' if real_prob > fake_prob else 'FAKE'
        })
        
        print(f"{name:20} | Real: {real_prob:.3f} | Fake: {fake_prob:.3f} | → {results[-1]['prediction']}")
    
    # Analyze bias
    fake_predictions = sum(1 for r in results if r['prediction'] == 'FAKE')
    real_predictions = sum(1 for r in results if r['prediction'] == 'REAL')
    
    print("\n" + "=" * 60)
    print("📊 BIAS ANALYSIS")
    print("=" * 60)
    print(f"Fake predictions: {fake_predictions}/{len(results)} ({fake_predictions/len(results)*100:.1f}%)")
    print(f"Real predictions: {real_predictions}/{len(results)} ({real_predictions/len(results)*100:.1f}%)")
    
    if fake_predictions > real_predictions:
        print("⚠️ MODEL SHOWS BIAS TOWARDS FAKE CLASSIFICATION!")
        print("   This could explain why real videos are classified as fake.")
    elif real_predictions > fake_predictions:
        print("ℹ️ Model shows bias towards real classification")
    else:
        print("✅ Model shows balanced predictions on test patterns")
    
    return results

def check_training_data_format():
    """Check if there's a mismatch in training data format"""
    print("\n🔍 Checking Training Data Format...")
    print("=" * 60)
    
    # Check if the model was trained with a different label encoding
    model, device, checkpoint = load_model()
    
    # Test what the model thinks are "definitely real" vs "definitely fake" patterns
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create a very "natural" looking face-like pattern
    natural_face = np.ones((224, 224, 3), dtype=np.uint8)
    # Skin color
    natural_face[:, :] = [220, 180, 150]
    # Add face features
    cv2.circle(natural_face, (75, 90), 8, (50, 50, 50), -1)   # Left eye
    cv2.circle(natural_face, (149, 90), 8, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(natural_face, (112, 120), (8, 12), 0, 0, 180, (100, 50, 50), -1)  # Nose
    cv2.ellipse(natural_face, (112, 160), (25, 8), 0, 0, 180, (150, 100, 100), -1)  # Mouth
    
    # Test this natural pattern
    pil_img = Image.fromarray(natural_face)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        raw_output = output[0].cpu().numpy()
        
    print(f"Natural face pattern:")
    print(f"  Raw output: [{raw_output[0]:.3f}, {raw_output[1]:.3f}]")
    print(f"  Probabilities: [Real: {probs[0,0]:.3f}, Fake: {probs[0,1]:.3f}]")
    print(f"  Prediction: {'REAL' if probs[0,0] > probs[0,1] else 'FAKE'}")
    
    # Check if labels might be swapped
    if probs[0,1] > probs[0,0]:  # If fake probability is higher for natural pattern
        print("\n⚠️ POTENTIAL LABEL SWAP DETECTED!")
        print("   The model classifies natural patterns as fake.")
        print("   This suggests the labels might be swapped during training or inference.")
        return True
    else:
        print("\n✅ Label encoding appears correct for natural patterns")
        return False

def test_preprocessing_pipeline():
    """Test if preprocessing is causing issues"""
    print("\n🔍 Testing Preprocessing Pipeline...")
    print("=" * 60)
    
    # Test the exact preprocessing used in the app
    from PIL import Image
    
    # Create a simple test image
    test_img = np.ones((160, 160, 3), dtype=np.uint8) * 128  # Gray image
    
    # Test different preprocessing approaches
    transform_app = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert BGR to RGB (common issue)
    test_img_rgb = test_img  # Already RGB
    test_img_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    test_img_bgr = cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB)  # Convert back
    
    model, device, _ = load_model()
    
    for name, img in [("Original RGB", test_img_rgb), ("BGR->RGB", test_img_bgr)]:
        pil_img = Image.fromarray(img)
        tensor = transform_app(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
        
        print(f"{name:15} | Real: {probs[0,0]:.3f} | Fake: {probs[0,1]:.3f}")
    
    print("✅ Preprocessing pipeline tested")

def main():
    """Main debugging function"""
    print("🚨 DeepSight AI - Real Video Misclassification Debug")
    print("=" * 70)
    
    if not os.path.exists("weights/best_model.pth"):
        print("❌ Model file not found: weights/best_model.pth")
        return
    
    # Run diagnostic tests
    bias_results = analyze_model_bias()
    label_swap = check_training_data_format()
    test_preprocessing_pipeline()
    
    print("\n" + "=" * 70)
    print("🔧 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    if label_swap:
        print("❌ CRITICAL ISSUE: Labels appear to be swapped!")
        print("   SOLUTION: Swap the class indices in the prediction logic:")
        print("   - Change: prediction = 'FAKE' if probs[0,1] > 0.5 else 'REAL'")
        print("   - To:     prediction = 'REAL' if probs[0,1] > 0.5 else 'FAKE'")
        print("   - Or swap the output indices in the softmax")
    
    fake_bias = sum(1 for r in bias_results if r['prediction'] == 'FAKE')
    if fake_bias > len(bias_results) * 0.7:
        print("⚠️ Model shows strong bias towards fake classification")
        print("   POSSIBLE CAUSES:")
        print("   - Training data imbalance")
        print("   - Incorrect data augmentation")
        print("   - Wrong normalization values")
        print("   - Model overfitting to fake patterns")
    
    print("\n💡 IMMEDIATE FIXES TO TRY:")
    print("1. Check if class labels are swapped in prediction logic")
    print("2. Verify training data had correct labels")
    print("3. Test with threshold adjustment (try 0.3 or 0.7 instead of 0.5)")
    print("4. Check if model was trained with different preprocessing")

if __name__ == "__main__":
    main()
