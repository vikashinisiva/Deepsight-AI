import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import tempfile
import shutil
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from grad_cam import GradCAM, make_infer_transform
import json
from datetime import datetime

class VideoFineTuner:
    """Fine-tune model on user's specific video for better recognition"""

    def __init__(self, video_path=None, model_path="weights/enhanced_model.pth"):
        self.video_path = video_path
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = "fine_tune_temp"

        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Data transforms
        self.train_transforms = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_faces_from_video(self, video_path, label, max_frames=200):
        """Extract faces from video for fine-tuning"""
        print(f"🎬 Extracting faces from video: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return []

        faces_dir = os.path.join(self.temp_dir, str(label))
        os.makedirs(faces_dir, exist_ok=True)

        frame_count = 0
        face_count = 0
        extracted_faces = []

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract faces every 10th frame for efficiency
            if frame_count % 10 == 0:
                faces = self.detect_faces(frame)

                for i, (x, y, w, h) in enumerate(faces):
                    if w > 50 and h > 50:  # Minimum face size
                        face_crop = frame[y:y+h, x:x+w]

                        # Save face crop
                        face_filename = "06d"
                        face_path = os.path.join(faces_dir, face_filename)
                        cv2.imwrite(face_path, face_crop)

                        extracted_faces.append(face_path)
                        face_count += 1

                        if face_count >= 100:  # Limit faces per video
                            break

            frame_count += 1

            if face_count >= 100:
                break

        cap.release()
        print(f"✅ Extracted {face_count} faces from {frame_count} frames")

        return extracted_faces

    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        return faces

    def create_fine_tune_dataset(self, user_video_path, user_label):
        """Create dataset from user's video for fine-tuning"""
        print("🔧 Creating fine-tuning dataset...")

        # Extract faces from user's video
        user_faces = self.extract_faces_from_video(user_video_path, user_label)

        if len(user_faces) < 10:
            print(f"⚠️  Only extracted {len(user_faces)} faces. Need more data for fine-tuning.")
            return None

        # Load some existing data for contrastive learning
        existing_real = glob.glob("crops_expanded/real/*.jpg")[:200] if os.path.exists("crops_expanded/real") else []
        existing_fake = glob.glob("crops_expanded/fake/*.jpg")[:200] if os.path.exists("crops_expanded/fake") else []

        # Create balanced dataset
        all_images = []
        all_labels = []

        # Add user's video faces
        for face_path in user_faces:
            all_images.append(face_path)
            all_labels.append(user_label)

        # Add contrastive examples
        contrast_label = 1 if user_label == 0 else 0
        contrast_images = existing_real if contrast_label == 0 else existing_fake

        for img_path in contrast_images[:len(user_faces)]:
            all_images.append(img_path)
            all_labels.append(contrast_label)

        print(f"📊 Fine-tuning dataset: {len(all_images)} images")
        print(f"   User video ({'REAL' if user_label == 0 else 'FAKE'}): {len(user_faces)} faces")
        print(f"   Contrast examples: {len(contrast_images[:len(user_faces)])} faces")

        return all_images, all_labels

    def fine_tune_model(self, video_path, user_label, epochs=10):
        """Fine-tune the model on user's video"""
        print("🎯 Starting model fine-tuning...")
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Label: {'REAL' if user_label == 0 else 'FAKE'}")

        # Create fine-tuning dataset
        dataset_info = self.create_fine_tune_dataset(video_path, user_label)
        if dataset_info is None:
            return False

        images, labels = dataset_info

        # Load pre-trained model
        model = self.load_model_for_fine_tuning()

        # Create data loader
        train_dataset = FineTuneDataset(images, labels, self.train_transforms)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        # Fine-tuning setup
        criterion = nn.CrossEntropyLoss()

        # Freeze most layers, only fine-tune classifier
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classifier and last few layers
        for param in model.classifier.parameters():
            param.requires_grad = True

        for param in model.features[-5:].parameters():  # Fine-tune last 5 blocks
            param.requires_grad = True

        optimizer = optim.AdamW([
            {'params': model.features[-5:].parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-4}
        ])

        # Fine-tuning loop
        model.train()
        best_acc = 0.0

        print("🚀 Fine-tuning in progress...")

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total
            avg_loss = total_loss / len(train_loader)

            print(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

            if accuracy > best_acc:
                best_acc = accuracy
                self.save_fine_tuned_model(model, video_path, user_label, accuracy)

        print(f"🏆 Fine-tuning completed! Best accuracy: {best_acc:.2f}%")
        return True

    def load_model_for_fine_tuning(self):
        """Load model for fine-tuning"""
        if os.path.exists(self.model_path):
            print(f"📂 Loading model: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Create model architecture
            model = models.efficientnet_b3(weights=None)
            num_features = model.classifier[1].in_features

            model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(0.3),
                AttentionBlock(768),
                nn.Linear(768, 384),
                nn.BatchNorm1d(384),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(384, 2)
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model loaded successfully")
        else:
            print("⚠️  Model not found, using base EfficientNet-B3")
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            num_features = model.classifier[1].in_features

            model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(768, 384),
                nn.BatchNorm1d(384),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(384, 2)
            )

        return model.to(self.device)

    def save_fine_tuned_model(self, model, video_path, user_label, accuracy):
        """Save fine-tuned model"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        label_name = "real" if user_label == 0 else "fake"

        fine_tuned_path = f"weights/fine_tuned_{label_name}_{video_name}.pth"

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'fine_tuned_on': video_path,
            'user_label': user_label,
            'timestamp': datetime.now().isoformat(),
            'model_config': 'Enhanced EfficientNet-B3 with Attention'
        }

        torch.save(checkpoint, fine_tuned_path)
        print(f"💾 Fine-tuned model saved: {fine_tuned_path}")

    def test_fine_tuned_model(self, video_path, model_path):
        """Test the fine-tuned model on the original video"""
        print("🧪 Testing fine-tuned model...")

        # Load fine-tuned model
        checkpoint = torch.load(model_path, map_location=self.device)

        model = models.efficientnet_b3(weights=None)
        num_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 2)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        # Test on original video
        result = self.analyze_video(video_path, model)

        if result:
            prediction = "REAL" if result['prediction'] == 0 else "FAKE"
            confidence = result['confidence']

            print("🎯 Fine-tuned model results:")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Frames analyzed: {result['frames_analyzed']}")

            return prediction, confidence

        return None, None

    def analyze_video(self, video_path, model):
        """Analyze video with the model"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        predictions = []
        frame_count = 0

        while frame_count < 100:  # Analyze first 100 frames
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces = self.detect_faces(frame)

            for (x, y, w, h) in faces:
                if w > 50 and h > 50:
                    face_crop = frame[y:y+h, x:x+w]

                    # Preprocess face
                    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    transform = make_infer_transform()
                    tensor = transform(rgb).unsqueeze(0).to(self.device)

                    # Get prediction
                    with torch.no_grad():
                        outputs = model(tensor)
                        probs = torch.softmax(outputs, dim=1)[0]
                        pred = outputs.max(1)[1].item()
                        confidence = probs[pred].item()

                    predictions.append((pred, confidence))
                    break  # Only process first face per frame

            frame_count += 1

        cap.release()

        if predictions:
            avg_pred = np.mean([p[0] for p in predictions])
            avg_confidence = np.mean([p[1] for p in predictions])

            return {
                'prediction': round(avg_pred),
                'confidence': avg_confidence,
                'frames_analyzed': len(predictions)
            }

        return None

    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("🧹 Cleaned up temporary files")

class AttentionBlock(nn.Module):
    """Attention mechanism for better feature focus"""
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

class FineTuneDataset(Dataset):
    """Dataset for fine-tuning"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def main():
    """Main function for video fine-tuning"""
    print("🎯 Video-Specific Model Fine-Tuning")
    print("=" * 50)

    # Example usage - replace with your video
    video_path = input("Enter path to your video: ").strip()
    if not video_path:
        print("❌ No video path provided")
        return

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    # Ask user for the correct label
    print("\nWhat is the correct label for this video?")
    print("0 = REAL (authentic video)")
    print("1 = FAKE (deepfake video)")

    try:
        user_label = int(input("Enter label (0 or 1): ").strip())
        if user_label not in [0, 1]:
            print("❌ Invalid label. Must be 0 or 1")
            return
    except ValueError:
        print("❌ Invalid input. Must be 0 or 1")
        return

    # Initialize fine-tuner
    fine_tuner = VideoFineTuner(video_path)

    try:
        # Fine-tune the model
        success = fine_tuner.fine_tune_model(video_path, user_label, epochs=15)

        if success:
            # Test the fine-tuned model
            model_path = f"weights/fine_tuned_{'real' if user_label == 0 else 'fake'}_{os.path.splitext(os.path.basename(video_path))[0]}.pth"
            prediction, confidence = fine_tuner.test_fine_tuned_model(video_path, model_path)

            if prediction and confidence:
                print("\n" + "=" * 50)
                print("🎉 FINE-TUNING COMPLETED!")
                print(f"Final Prediction: {prediction}")
                print(f"Confidence: {confidence:.1%}")

                if (prediction == "REAL" and user_label == 0) or (prediction == "FAKE" and user_label == 1):
                    print("✅ SUCCESS! Model now correctly recognizes your video")
                else:
                    print("⚠️  Model still has issues. Consider:")
                    print("   - Providing more video samples")
                    print("   - Checking video quality")
                    print("   - Trying different preprocessing")
            else:
                print("❌ Could not test the fine-tuned model")
        else:
            print("❌ Fine-tuning failed")

    finally:
        fine_tuner.cleanup()

if __name__ == "__main__":
    main()
