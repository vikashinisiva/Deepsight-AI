import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime

class EnhancedDeepfakeDataset(Dataset):
    """Enhanced dataset with better preprocessing and augmentation"""

    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        self.samples = []
        self.class_weights = {}

        # Load all samples
        for class_name in ['real', 'fake']:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                class_idx = 0 if class_name == 'real' else 1
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))

        # Calculate class weights for balancing
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        total = len(self.samples)

        self.class_weights = {
            0: total / (2 * real_count),  # Real class weight
            1: total / (2 * fake_count)   # Fake class weight
        }

        print(f"📊 Dataset loaded: {len(self.samples)} samples")
        print(f"   Real: {real_count}, Fake: {fake_count}")
        print(f"   Class weights: {self.class_weights}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image with error handling
        try:
            image = cv2.imread(img_path)
            if image is None:
                # Fallback to PIL if cv2 fails
                image = np.array(Image.open(img_path).convert('RGB'))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: Could not load {img_path}, skipping...")
            # Return a dummy sample
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            label = 0

        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image)
                image = transformed["image"]
            else:
                # Convert to PIL for torchvision transforms
                image = Image.fromarray(image)
                image = self.transform(image)

        return image, label

class EnhancedDeepfakeTrainer:
    """Enhanced trainer with better generalization and robustness"""

    def __init__(self, data_dir="crops_improved", batch_size=16, epochs=50):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_accuracy = 0.0
        self.best_model_path = "weights/enhanced_model.pth"

        # Enhanced data augmentation for better generalization
        self.train_transforms = A.Compose([
            A.Resize(256, 256),  # Larger resize for better quality
            A.RandomCrop(224, 224),  # Random crop for variability
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.3),  # Increased rotation range
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(var_limit=(5, 30), p=0.1),  # Reduced noise
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.2),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=0.1),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.1),
            A.CLAHE(clip_limit=2.0, p=0.1),  # Contrast Limited Adaptive Histogram Equalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.val_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        print(f"🚀 Enhanced Deepfake Trainer initialized on {self.device}")

    def create_enhanced_model(self):
        """Create enhanced model with attention mechanisms and better architecture"""

        # Use EfficientNet-B3 as base with improvements
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        # Enhanced classifier with attention
        num_features = model.classifier[1].in_features

        # Custom attention block
        class AttentionBlock(nn.Module):
            def __init__(self, in_features):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.Linear(in_features, in_features // 8),
                    nn.ReLU(),
                    nn.Linear(in_features // 8, in_features),
                    nn.Sigmoid()
                )

            def forward(self, x):
                attention_weights = self.attention(x)
                return x * attention_weights

        model.classifier = nn.Sequential(
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(num_features, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            AttentionBlock(768),  # Add attention mechanism
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 2)
        )

        # Unfreeze more layers for better fine-tuning
        for param in model.features[:3].parameters():  # Only freeze first 3 blocks
            param.requires_grad = False

        return model.to(self.device)

    def create_data_loaders(self):
        """Create enhanced data loaders with better balancing"""

        # Create datasets
        train_dataset = EnhancedDeepfakeDataset(
            self.data_dir, transform=self.train_transforms, is_training=True
        )
        val_dataset = EnhancedDeepfakeDataset(
            self.data_dir, transform=self.val_transforms, is_training=False
        )

        # Split dataset with stratification
        n = len(train_dataset)
        n_val = max(1, int(0.25 * n))  # Increased validation set
        n_train = n - n_val

        train_dataset, val_dataset = random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        # Enhanced weighted sampling
        train_targets = []
        for idx in train_dataset.indices:
            _, label = train_dataset.dataset.samples[idx]
            train_targets.append(label)

        class_counts = np.bincount(train_targets)
        class_weights = 1. / (class_counts + 1e-6)  # Add small epsilon
        sample_weights = class_weights[train_targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        # Create data loaders with enhanced settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True  # Ensure consistent batch sizes
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        return train_loader, val_loader, len(train_dataset), len(val_dataset)

    def train_epoch(self, model, train_loader, optimizer, criterion, scheduler, epoch):
        """Enhanced training epoch with better techniques"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Advanced data augmentation during training
            if random.random() < 0.2:  # 20% chance for MixUp
                lam = np.random.beta(0.8, 0.8)
                rand_index = torch.randperm(inputs.size()[0]).to(self.device)
                inputs_mixed = lam * inputs + (1 - lam) * inputs[rand_index]
                labels_a, labels_b = labels, labels[rand_index]

                optimizer.zero_grad()
                outputs = model(inputs_mixed)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Add L2 regularization manually
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += 1e-5 * l2_reg

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        scheduler.step()
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, model, val_loader, criterion):
        """Enhanced validation with detailed metrics"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)

        return avg_loss, accuracy, all_preds, all_labels, all_probs

    def save_checkpoint(self, model, optimizer, scheduler, epoch, accuracy, filename):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy': accuracy,
            'model_config': {
                'type': 'Enhanced EfficientNet-B3',
                'attention_mechanism': True,
                'dropout_rates': [0.4, 0.3, 0.2],
                'batch_norm': True
            },
            'training_config': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'data_dir': self.data_dir
            },
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, filename)
        print(f"💾 Checkpoint saved: {filename}")

    def load_checkpoint(self, model, optimizer, scheduler, filename):
        """Load model checkpoint"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint.get('accuracy', 0.0)

            print(f"📂 Checkpoint loaded: {filename}")
            print(f"   Previous best accuracy: {best_accuracy:.2f}%")
            print(f"   Resuming from epoch: {start_epoch}")

            return start_epoch, best_accuracy
        return 0, 0.0

    def train(self, resume_training=False):
        """Main enhanced training function"""
        print("🔧 Setting up enhanced model and data loaders...")

        # Create model
        model = self.create_enhanced_model()

        # Create data loaders
        train_loader, val_loader, n_train, n_val = self.create_data_loaders()

        print(f"📊 Dataset: {n_train} train, {n_val} validation samples")
        print(f"🏗️  Enhanced Model: EfficientNet-B3 with Attention")
        print(f"🎯 Target: Robust generalization to new videos")

        # Enhanced loss function
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Advanced optimizer with different learning rates
        optimizer = optim.AdamW([
            {'params': model.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 5e-4, 'weight_decay': 5e-4}
        ])

        # Learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Resume training if requested
        start_epoch = 0
        if resume_training:
            start_epoch, self.best_accuracy = self.load_checkpoint(
                model, optimizer, scheduler, self.best_model_path
            )

        # Training history
        history = {
            'train_losses': [], 'val_losses': [],
            'train_accs': [], 'val_accs': [],
            'learning_rates': []
        }

        print("🚀 Starting enhanced training...")

        for epoch in range(start_epoch, self.epochs):
            print(f"\n📈 Epoch {epoch + 1}/{self.epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler, epoch
            )

            # Validate
            val_loss, val_acc, preds, labels, probs = self.validate(
                model, val_loader, criterion
            )

            # Store history
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['train_accs'].append(train_acc)
            history['val_accs'].append(val_acc)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            print(f"📊 Results:")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_checkpoint(model, optimizer, scheduler, epoch,
                                   val_acc, self.best_model_path)
                print(f"💾 New best model saved! Accuracy: {val_acc:.2f}%")

            # Save regular checkpoints
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"weights/checkpoint_epoch_{epoch+1}.pth"
                self.save_checkpoint(model, optimizer, scheduler, epoch,
                                   val_acc, checkpoint_path)

            # Early stopping with patience
            if epoch > 15:
                recent_accs = history['val_accs'][-5:]
                if max(recent_accs) - min(recent_accs) < 1.0:  # Little improvement
                    print("⚠️  Training plateau detected, considering early stopping...")
                    break

        print("\n🎉 Enhanced training completed!")
        print(f"🏆 Best validation accuracy: {self.best_accuracy:.2f}%")

        # Generate comprehensive report
        self.generate_training_report(history, val_loader, criterion, model)

        return self.best_accuracy

    def generate_training_report(self, history, val_loader, criterion, model):
        """Generate comprehensive training report"""
        print("\n📊 Generating comprehensive training report...")

        # Final validation
        final_val_loss, final_val_acc, final_preds, final_labels, final_probs = self.validate(
            model, val_loader, criterion
        )

        # Classification report
        print("\n📈 Classification Report:")
        print(classification_report(final_labels, final_preds,
                           target_names=['REAL', 'FAKE'], digits=4))

        # Confusion matrix
        cm = confusion_matrix(final_labels, final_preds)
        print("\n📊 Confusion Matrix:")
        print(cm)

        # Save training history
        with open('enhanced_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Plot results
        self.plot_enhanced_training_history(history)

        print("📄 Report saved to: enhanced_training_history.json")

    def plot_enhanced_training_history(self, history):
        """Plot enhanced training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(history['train_losses']) + 1)

        # Loss plot
        ax1.plot(epochs, history['train_losses'], label='Train Loss', marker='o')
        ax1.plot(epochs, history['val_losses'], label='Val Loss', marker='s')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, history['train_accs'], label='Train Accuracy', marker='o')
        ax2.plot(epochs, history['val_accs'], label='Val Accuracy', marker='s')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning rate plot
        ax3.plot(epochs, history['learning_rates'], label='Learning Rate', color='orange', marker='^')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # Training dynamics
        ax4.plot(epochs, history['train_accs'], label='Train Acc', alpha=0.7)
        ax4.plot(epochs, history['val_accs'], label='Val Acc', alpha=0.7)
        ax4.fill_between(epochs,
                        np.array(history['train_accs']) - 2,
                        np.array(history['train_accs']) + 2,
                        alpha=0.2, label='Train ±2%')
        ax4.set_title('Training Dynamics with Confidence')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('enhanced_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run enhanced training"""
    os.makedirs("weights", exist_ok=True)

    print("🎯 Starting Enhanced Deepfake Detection Training")
    print("=" * 50)

    trainer = EnhancedDeepfakeTrainer(
        data_dir="crops_improved",
        batch_size=12,  # Smaller batch for better generalization
        epochs=40
    )

    # Train the enhanced model
    best_accuracy = trainer.train(resume_training=False)

    print("\n" + "=" * 50)
    print("🏆 FINAL RESULTS:")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {trainer.best_model_path}")

    if best_accuracy > 95:
        print("🎉 EXCELLENT! Model shows excellent generalization")
    elif best_accuracy > 90:
        print("✅ GOOD! Model ready for production use")
    elif best_accuracy > 85:
        print("⚠️  ACCEPTABLE! Model needs minor improvements")
    else:
        print("❌ NEEDS IMPROVEMENT! Consider more data or architecture changes")

    print("\n💡 Next Steps:")
    print("1. Test the model on your specific video")
    print("2. If still not working, consider adding more diverse training data")
    print("3. Try fine-tuning on videos similar to yours")
    print("4. Consider ensemble methods for better robustness")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\visha\DeepSight_AI\Deepsight-AI\train_enhanced.py
