import os, torch, torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image

class AdvancedDeepfakeTrainer:
    def __init__(self, data_dir="crops_improved", batch_size=32, epochs=50):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_accuracy = 0.0
        self.best_model_path = "weights/advanced_model.pth"

        # Advanced data augmentation
        self.train_transforms = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(var_limit=(10, 50), p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.val_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        print(f"🚀 Advanced Deepfake Trainer initialized on {self.device}")

    def create_model(self, model_name="efficientnet_b2"):
        """Create advanced model with better architecture"""
        if model_name == "efficientnet_b2":
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            num_features = model.classifier[1].in_features
        elif model_name == "efficientnet_b3":
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            num_features = model.classifier[1].in_features
        elif model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_features = model.fc.in_features
            model.fc = nn.Identity()
        else:
            # Default to EfficientNet-B0
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            num_features = model.classifier[1].in_features

        # Advanced classifier with dropout and batch norm
        if "efficientnet" in model_name:
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
        elif model_name == "resnet50":
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

        # Freeze early layers, fine-tune later layers
        if "efficientnet" in model_name:
            for param in model.features[:5].parameters():
                param.requires_grad = False
        elif model_name == "resnet50":
            for param in list(model.parameters())[:-10]:
                param.requires_grad = False

        return model.to(self.device)

    def create_data_loaders(self):
        """Create advanced data loaders with augmentation and balancing"""

        class AlbumentationsDataset(datasets.ImageFolder):
            def __init__(self, root, transform=None, **kwargs):
                super().__init__(root, **kwargs)
                self.transform = transform

            def __getitem__(self, index):
                path, target = self.samples[index]
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.transform:
                    transformed = self.transform(image=image)
                    image = transformed["image"]

                return image, target

        # Create datasets
        train_dataset = AlbumentationsDataset(
            self.data_dir, transform=self.train_transforms
        )
        val_dataset = AlbumentationsDataset(
            self.data_dir, transform=self.val_transforms
        )

        # Split dataset
        n = len(train_dataset)
        n_val = max(1, int(0.2 * n))
        n_train = n - n_val

        train_dataset, val_dataset = random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        # Handle class imbalance with weighted sampling
        train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
        class_counts = np.bincount(train_targets)
        class_weights = 1. / class_counts
        sample_weights = class_weights[train_targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            sampler=sampler, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )

        return train_loader, val_loader, len(train_dataset), len(val_dataset)

    def train_epoch(self, model, train_loader, optimizer, criterion, scheduler):
        """Train for one epoch with advanced techniques"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # MixUp augmentation (optional, advanced technique)
            if np.random.random() < 0.1:  # 10% chance
                lam = np.random.beta(0.4, 0.4)
                rand_index = torch.randperm(inputs.size()[0]).to(self.device)
                inputs = lam * inputs + (1 - lam) * inputs[rand_index]
                labels_a, labels_b = labels, labels[rand_index]

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        scheduler.step()
        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, model, val_loader, criterion):
        """Validate the model"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100. * correct / total
        return total_loss / len(val_loader), accuracy, all_preds, all_labels

    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training history"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(train_accs, label='Train Accuracy')
        ax2.plot(val_accs, label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def train(self, model_name="efficientnet_b2"):
        """Main training function"""
        print("🔧 Setting up model and data loaders...")

        # Create model
        model = self.create_model(model_name)

        # Create data loaders
        train_loader, val_loader, n_train, n_val = self.create_data_loaders()

        print(f"📊 Dataset: {n_train} train, {n_val} validation samples")
        print(f"🏗️  Model: {model_name}")
        print(f"🎯 Target: Improve accuracy beyond 90%")

        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Advanced optimizer
        optimizer = optim.AdamW([
            {'params': model.features.parameters() if hasattr(model, 'features') else
                      [p for n, p in model.named_parameters() if 'classifier' not in n and 'fc' not in n],
             'lr': 1e-4, 'weight_decay': 1e-4},
            {'params': model.classifier.parameters() if hasattr(model, 'classifier') else
                      model.fc.parameters(),
             'lr': 1e-3, 'weight_decay': 1e-3}
        ])

        # Learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Training history
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        print("🚀 Starting advanced training...")

        for epoch in range(self.epochs):
            print(f"\n📈 Epoch {epoch + 1}/{self.epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler
            )

            # Validate
            val_loss, val_acc, preds, labels = self.validate(
                model, val_loader, criterion
            )

            # Store history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_acc,
                    'model_name': model_name
                }, self.best_model_path)
                print(f"💾 New best model saved! Accuracy: {val_acc:.2f}%")

            # Early stopping (optional)
            if epoch > 10 and val_acc < 85:
                print("⚠️  Validation accuracy too low, considering early stopping...")
                break

        print("\n🎉 Training completed!")
        print(f"🏆 Best validation accuracy: {self.best_accuracy:.2f}%")
        print(f"💾 Best model saved to: {self.best_model_path}")

        # Generate final report
        print("\n📊 Generating final report...")
        final_val_loss, final_val_acc, final_preds, final_labels = self.validate(
            model, val_loader, criterion
        )

        print("\n📈 Classification Report:")
        print(classification_report(final_labels, final_preds,
                           target_names=['REAL', 'FAKE']))

        # Plot results
        try:
            self.plot_training_history(train_losses, val_losses, train_accs, val_accs)
            self.plot_confusion_matrix(final_labels, final_preds, ['REAL', 'FAKE'])
        except ImportError:
            print("⚠️  Matplotlib/seaborn not available for plotting")

        return self.best_accuracy

def main():
    """Main function to run advanced training"""
    os.makedirs("weights", exist_ok=True)

    # Try different models for ensemble
    models_to_try = ["efficientnet_b2", "efficientnet_b3", "resnet50"]
    best_overall_accuracy = 0.0
    best_model_name = ""

    for model_name in models_to_try:
        print(f"\n🔬 Training {model_name}...")
        trainer = AdvancedDeepfakeTrainer(
            data_dir="crops_improved",
            batch_size=16,  # Smaller batch for better generalization
            epochs=30
        )

        accuracy = trainer.train(model_name)

        if accuracy > best_overall_accuracy:
            best_overall_accuracy = accuracy
            best_model_name = model_name

    print("\n🎯 Best Model Results:")
    print(f"Model: {best_model_name}")
    print(f"Accuracy: {best_overall_accuracy:.2f}%")

    if best_overall_accuracy > 95:
        print("🎉 Excellent! Model accuracy exceeds 95%")
    elif best_overall_accuracy > 90:
        print("✅ Good! Model accuracy exceeds 90%")
    else:
        print("⚠️  Model needs further improvement")

if __name__ == "__main__":
    main()
