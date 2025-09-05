import os, torch, torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import pickle

class EnsembleModel(nn.Module):
    """Ensemble of multiple models for better accuracy"""
    def __init__(self, model_paths, device):
        super(EnsembleModel, self).__init__()
        self.models = []
        self.device = device

        for path in model_paths:
            if os.path.exists(path):
                # Load model architecture based on filename
                if "efficientnet_b2" in path:
                    model = models.efficientnet_b2(weights=None)
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
                elif "efficientnet_b3" in path:
                    model = models.efficientnet_b3(weights=None)
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
                elif "resnet50" in path:
                    model = models.resnet50(weights=None)
                    model.fc = nn.Linear(model.fc.in_features, 2)
                else:
                    # Default to EfficientNet-B0
                    model = models.efficientnet_b0(weights=None)
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

                # Load weights
                checkpoint = torch.load(path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                model.to(device)
                model.eval()
                self.models.append(model)
                print(f"✅ Loaded model: {os.path.basename(path)}")
            else:
                print(f"⚠️  Model not found: {path}")

        if not self.models:
            raise ValueError("No valid models found!")

        print(f"🎯 Ensemble created with {len(self.models)} models")

    def forward(self, x):
        """Ensemble prediction using majority voting"""
        predictions = []

        with torch.no_grad():
            for model in self.models:
                outputs = model(x)
                _, preds = outputs.max(1)
                predictions.append(preds)

        # Majority voting
        predictions = torch.stack(predictions, dim=0)
        ensemble_pred, _ = torch.mode(predictions, dim=0)

        return ensemble_pred

def create_ensemble_model(model_paths, device):
    """Create and return ensemble model"""
    return EnsembleModel(model_paths, device)

def evaluate_ensemble(ensemble_model, val_loader, device):
    """Evaluate ensemble model performance"""
    ensemble_model.eval()
    all_preds = []
    all_labels = []

    print("🔍 Evaluating ensemble model...")

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            predictions = ensemble_model(inputs)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    return accuracy, all_preds, all_labels

def train_ensemble_components():
    """Train individual models for ensemble"""
    from train_advanced import AdvancedDeepfakeTrainer

    models_to_train = ["efficientnet_b2", "efficientnet_b3"]
    trained_models = []

    for model_name in models_to_train:
        print(f"\n🏗️  Training {model_name} for ensemble...")

        trainer = AdvancedDeepfakeTrainer(
            data_dir="crops_improved",
            batch_size=16,
            epochs=25  # Fewer epochs for ensemble components
        )

        accuracy = trainer.train(model_name)
        trained_models.append((model_name, trainer.best_model_path, accuracy))

        print(f"✅ {model_name} trained with {accuracy:.2f}% accuracy")

    return trained_models

def main():
    """Main ensemble training and evaluation"""
    os.makedirs("weights", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🎯 Deepfake Detection Ensemble Training")
    print("=" * 50)

    # Option 1: Train new models for ensemble
    print("\n1️⃣  Training individual models for ensemble...")
    trained_models = train_ensemble_components()

    # Option 2: Use existing trained models
    existing_models = [
        "weights/advanced_model.pth",
        "weights/baseline_improved.pth"
    ]

    # Combine all available models
    model_paths = []
    for _, path, _ in trained_models:
        if os.path.exists(path):
            model_paths.append(path)

    for path in existing_models:
        if os.path.exists(path):
            model_paths.append(path)

    if len(model_paths) < 2:
        print("⚠️  Need at least 2 models for ensemble. Training additional models...")
        # Train one more model
        trainer = AdvancedDeepfakeTrainer(
            data_dir="crops_improved", batch_size=16, epochs=20
        )
        trainer.train("resnet50")
        model_paths.append(trainer.best_model_path)

    print(f"\n🔧 Creating ensemble with {len(model_paths)} models:")
    for i, path in enumerate(model_paths, 1):
        print(f"  {i}. {os.path.basename(path)}")

    # Create data loader for evaluation
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

    # Create and evaluate ensemble
    try:
        ensemble = create_ensemble_model(model_paths, device)
        ensemble_accuracy, preds, labels = evaluate_ensemble(ensemble, val_loader, device)

        print("
🎉 Ensemble Results:"        print(f"Individual model accuracies: {[f'{acc:.1f}%' for _, _, acc in trained_models]}")
        print(f"Ensemble accuracy: {ensemble_accuracy:.2f}%")

        if ensemble_accuracy > 95:
            print("🎯 Excellent! Ensemble accuracy exceeds 95%")
        elif ensemble_accuracy > 92:
            print("✅ Great! Ensemble accuracy exceeds 92%")
        else:
            print("📈 Good! Ensemble provides accuracy boost")

        # Save ensemble model
        ensemble_path = "weights/ensemble_model.pkl"
        with open(ensemble_path, 'wb') as f:
            pickle.dump({
                'model_paths': model_paths,
                'accuracy': ensemble_accuracy,
                'device': str(device)
            }, f)

        print(f"💾 Ensemble configuration saved to: {ensemble_path}")

        # Save ensemble predictions for analysis
        np.savez('ensemble_results.npz',
                predictions=preds,
                labels=labels,
                accuracy=ensemble_accuracy)

    except Exception as e:
        print(f"❌ Error creating ensemble: {e}")
        print("💡 Try training individual models first")

if __name__ == "__main__":
    main()
