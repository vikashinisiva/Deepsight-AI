import os, torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm

def load_model(model_path, model_name, device):
    """Load a trained model"""
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        # Use the same custom classifier as in train_advanced.py
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
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
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded {model_name} from {model_path}")
    else:
        print(f"⚠️  Model not found: {model_path}")
        return None

    model.to(device)
    model.eval()
    return model

def evaluate_model(model, val_loader, device, model_name):
    """Evaluate a single model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100

    # Fake detection accuracy (class 1)
    fake_mask = np.array(all_labels) == 1
    if np.sum(fake_mask) > 0:
        fake_accuracy = np.mean(np.array(all_preds)[fake_mask] == 1) * 100
    else:
        fake_accuracy = 0

    # Real detection accuracy (class 0)
    real_mask = np.array(all_labels) == 0
    if np.sum(real_mask) > 0:
        real_accuracy = np.mean(np.array(all_preds)[real_mask] == 0) * 100
    else:
        real_accuracy = 0

    return {
        'model': model_name,
        'accuracy': accuracy,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def create_comparison_report(results):
    """Create a detailed comparison report"""
    print("\n" + "="*60)
    print("🎯 MODEL COMPARISON REPORT")
    print("="*60)

    # Create summary table
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['model'],
            'Overall Accuracy': f"{result['accuracy']:.2f}%",
            'Real Detection': f"{result['real_accuracy']:.2f}%",
            'Fake Detection': f"{result['fake_accuracy']:.2f}%"
        })

    df = pd.DataFrame(summary_data)
    print("\n📊 Accuracy Comparison:")
    print(df.to_string(index=False))

    # Find best models
    best_overall = max(results, key=lambda x: x['accuracy'])
    best_real = max(results, key=lambda x: x['real_accuracy'])
    best_fake = max(results, key=lambda x: x['fake_accuracy'])

    print("\n🏆 Best Performers:")
    print(f"Overall Accuracy: {best_overall['model']} ({best_overall['accuracy']:.2f}%)")
    print(f"Real Detection: {best_real['model']} ({best_real['real_accuracy']:.2f}%)")
    print(f"Fake Detection: {best_fake['model']} ({best_fake['fake_accuracy']:.2f}%)")

    # Detailed classification reports
    print("\n📋 Detailed Classification Reports:")
    for result in results:
        print(f"\n🔍 {result['model']}:")
        print(classification_report(result['labels'], result['predictions'],
                                  target_names=['REAL', 'FAKE'], digits=3))

    return df

def main():
    """Compare different trained models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Model Comparison on {device}")

    # Models to compare - dynamically detect available models
    models_to_test = [
        ("weights/baseline.pth", "efficientnet_b0"),
        ("weights/baseline_improved.pth", "efficientnet_b0"),
        ("weights/advanced_model.pth", "resnet50"),  # Last model trained was resnet50
    ]

    # Check which models exist
    available_models = []
    for path, name in models_to_test:
        if os.path.exists(path):
            available_models.append((path, name))
        else:
            print(f"⚠️  Skipping {name}: {path} not found")

    if not available_models:
        print("❌ No trained models found!")
        print("💡 Train some models first using:")
        print("   python train_baseline.py")
        print("   python train_advanced.py")
        return

    # Create validation dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder("crops_improved", transform=transform)
    n = len(dataset)
    n_val = max(1, int(0.2 * n))
    _, val_dataset = random_split(dataset, [n - n_val, n_val],
                                 generator=torch.Generator().manual_seed(42))

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"📊 Testing on {len(val_dataset)} validation samples")

    # Evaluate all models
    results = []
    for model_path, model_name in available_models:
        print(f"\n🔬 Testing {model_name}...")
        model = load_model(model_path, model_name, device)

        if model is not None:
            result = evaluate_model(model, val_loader, device, model_name)
            results.append(result)
            print(f"✅ {model_name}: {result['accuracy']:.2f}% accuracy")
    # Create comparison report
    if results:
        df = create_comparison_report(results)

        # Save results
        df.to_csv('model_comparison.csv', index=False)
        print("\n💾 Results saved to: model_comparison.csv")

        # Recommendations
        best_model = max(results, key=lambda x: x['accuracy'])
        print("\n🎯 Recommendations:")
        print(f"🏆 Use {best_model['model']} for best overall performance")

        if best_model['accuracy'] > 95:
            print("🎉 Excellent accuracy! Model is production-ready")
        elif best_model['accuracy'] > 90:
            print("✅ Good accuracy! Consider ensemble methods for further improvement")
        else:
            print("📈 Model needs improvement. Try:")
            print("   - More training data")
            print("   - Advanced augmentation")
            print("   - Different architectures")
            print("   - Ensemble methods")

    else:
        print("❌ No models were successfully evaluated")

if __name__ == "__main__":
    main()
