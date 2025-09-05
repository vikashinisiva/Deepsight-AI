import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import glob
import shutil
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from datetime import datetime

class DatasetExpander:
    """Expand dataset with diverse sources and enhanced preprocessing"""

    def __init__(self, base_dir="crops_improved"):
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expanded_dir = "crops_expanded"

        # Enhanced preprocessing transforms
        self.preprocess_transforms = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Data augmentation for expansion
        self.augmentation_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(var_limit=(10, 50), p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.1),
            A.Resize(224, 224)
        ])

        os.makedirs(self.expanded_dir, exist_ok=True)
        os.makedirs(os.path.join(self.expanded_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(self.expanded_dir, "fake"), exist_ok=True)

    def analyze_current_dataset(self):
        """Analyze current dataset quality and diversity"""
        print("🔍 Analyzing current dataset...")

        real_dir = os.path.join(self.base_dir, "real")
        fake_dir = os.path.join(self.base_dir, "fake")

        real_images = glob.glob(os.path.join(real_dir, "*.jpg"))
        fake_images = glob.glob(os.path.join(fake_dir, "*.jpg"))

        print(f"Current dataset: {len(real_images)} real, {len(fake_images)} fake")

        # Analyze image quality metrics
        real_stats = self.analyze_image_quality(real_images, "REAL")
        fake_stats = self.analyze_image_quality(fake_images, "FAKE")

        return real_stats, fake_stats

    def analyze_image_quality(self, image_paths, label):
        """Analyze image quality metrics"""
        stats = {
            'count': len(image_paths),
            'avg_brightness': [],
            'avg_contrast': [],
            'avg_sharpness': [],
            'resolutions': []
        }

        for img_path in tqdm(image_paths[:100], desc=f"Analyzing {label}"):  # Sample first 100
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Brightness
                brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2])
                stats['avg_brightness'].append(brightness)

                # Contrast
                contrast = img.std()
                stats['avg_contrast'].append(contrast)

                # Sharpness (using Laplacian variance)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                stats['avg_sharpness'].append(sharpness)

                # Resolution
                stats['resolutions'].append(img.shape[:2])

            except Exception as e:
                continue

        # Calculate averages
        for key in ['avg_brightness', 'avg_contrast', 'avg_sharpness']:
            if stats[key]:
                stats[key] = np.mean(stats[key])
            else:
                stats[key] = 0

        return stats

    def expand_dataset(self, target_samples=2000):
        """Expand dataset to target number of samples"""
        print(f"🔄 Expanding dataset to {target_samples} samples per class...")

        real_dir = os.path.join(self.base_dir, "real")
        fake_dir = os.path.join(self.base_dir, "fake")

        real_images = glob.glob(os.path.join(real_dir, "*.jpg"))
        fake_images = glob.glob(os.path.join(fake_dir, "*.jpg"))

        current_real = len(real_images)
        current_fake = len(fake_images)

        print(f"Current: {current_real} real, {current_fake} fake")

        # Expand real images
        if current_real < target_samples:
            self.expand_class(real_images, "real", target_samples - current_real)

        # Expand fake images
        if current_fake < target_samples:
            self.expand_class(fake_images, "fake", target_samples - current_fake)

        # Verify expansion
        final_real = len(glob.glob(os.path.join(self.expanded_dir, "real", "*.jpg")))
        final_fake = len(glob.glob(os.path.join(self.expanded_dir, "fake", "*.jpg")))

        print(f"Expanded: {final_real} real, {final_fake} fake")

    def expand_class(self, image_paths, class_name, needed_samples):
        """Expand a specific class using augmentation"""
        print(f"📈 Expanding {class_name} class with {needed_samples} new samples...")

        output_dir = os.path.join(self.expanded_dir, class_name)
        os.makedirs(output_dir, exist_ok=True)

        # Copy original images
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(output_dir, filename))

        # Generate augmented samples
        generated = 0
        attempts = 0
        max_attempts = needed_samples * 2  # Allow some failures

        while generated < needed_samples and attempts < max_attempts:
            # Randomly select base image
            base_img_path = random.choice(image_paths)

            try:
                # Load and augment image
                img = cv2.imread(base_img_path)
                if img is None:
                    attempts += 1
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Apply augmentation
                augmented = self.augmentation_transforms(image=img)
                aug_img = augmented["image"]

                # Save augmented image
                base_name = os.path.splitext(os.path.basename(base_img_path))[0]
                new_filename = "04d"
                output_path = os.path.join(output_dir, new_filename)

                # Convert back to BGR for saving
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, aug_img_bgr)

                generated += 1
                attempts += 1

            except Exception as e:
                attempts += 1
                continue

        print(f"✅ Generated {generated} augmented samples for {class_name}")

    def add_diverse_sources(self):
        """Add diverse data sources for better generalization"""
        print("🌍 Adding diverse data sources...")

        # This would typically involve downloading or using additional datasets
        # For now, we'll create synthetic variations

        real_dir = os.path.join(self.expanded_dir, "real")
        fake_dir = os.path.join(self.expanded_dir, "fake")

        # Add lighting variations
        self.add_lighting_variations(real_dir, "real")
        self.add_lighting_variations(fake_dir, "fake")

        # Add quality variations (compression artifacts)
        self.add_compression_artifacts(real_dir, "real")
        self.add_compression_artifacts(fake_dir, "fake")

    def add_lighting_variations(self, source_dir, class_name):
        """Add different lighting conditions"""
        print(f"💡 Adding lighting variations for {class_name}...")

        image_paths = glob.glob(os.path.join(source_dir, "*.jpg"))

        for i, img_path in enumerate(image_paths[:50]):  # Process first 50
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Create different lighting conditions
                lighting_variations = [
                    self.adjust_brightness(img, 0.7),  # Darker
                    self.adjust_brightness(img, 1.3),  # Brighter
                    self.adjust_contrast(img, 0.8),    # Lower contrast
                    self.adjust_contrast(img, 1.2),    # Higher contrast
                ]

                for j, var_img in enumerate(lighting_variations):
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    new_filename = "04d"
                    output_path = os.path.join(source_dir, new_filename)
                    cv2.imwrite(output_path, var_img)

            except Exception as e:
                continue

    def add_compression_artifacts(self, source_dir, class_name):
        """Add compression artifacts for robustness"""
        print(f"🗜️ Adding compression artifacts for {class_name}...")

        image_paths = glob.glob(os.path.join(source_dir, "*.jpg"))

        for i, img_path in enumerate(image_paths[:30]):  # Process first 30
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Create compression artifacts
                for quality in [30, 50, 70]:  # Different JPEG qualities
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    new_filename = "04d"
                    output_path = os.path.join(source_dir, new_filename)

                    # Save with different quality
                    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

            except Exception as e:
                continue

    def adjust_brightness(self, img, factor):
        """Adjust image brightness"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def adjust_contrast(self, img, factor):
        """Adjust image contrast"""
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

    def validate_expanded_dataset(self):
        """Validate the expanded dataset quality"""
        print("✅ Validating expanded dataset...")

        real_dir = os.path.join(self.expanded_dir, "real")
        fake_dir = os.path.join(self.expanded_dir, "fake")

        real_count = len(glob.glob(os.path.join(real_dir, "*.jpg")))
        fake_count = len(glob.glob(os.path.join(fake_dir, "*.jpg")))

        print(f"Expanded dataset: {real_count} real, {fake_count} fake")

        # Check for corrupted images
        corrupted_real = self.check_corrupted_images(real_dir)
        corrupted_fake = self.check_corrupted_images(fake_dir)

        print(f"Corrupted images: {corrupted_real} real, {corrupted_fake} fake")

        return real_count, fake_count

    def check_corrupted_images(self, directory):
        """Check for corrupted images in directory"""
        corrupted = 0
        image_paths = glob.glob(os.path.join(directory, "*.jpg"))

        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None or img.size == 0:
                    corrupted += 1
                    os.remove(img_path)  # Remove corrupted file
            except:
                corrupted += 1
                try:
                    os.remove(img_path)
                except:
                    pass

        return corrupted

    def create_dataset_report(self):
        """Create comprehensive dataset report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_dataset': {
                'real_count': len(glob.glob(os.path.join(self.base_dir, "real", "*.jpg"))),
                'fake_count': len(glob.glob(os.path.join(self.base_dir, "fake", "*.jpg")))
            },
            'expanded_dataset': {
                'real_count': len(glob.glob(os.path.join(self.expanded_dir, "real", "*.jpg"))),
                'fake_count': len(glob.glob(os.path.join(self.expanded_dir, "fake", "*.jpg")))
            },
            'expansion_techniques': [
                'horizontal_flip',
                'rotation',
                'gaussian_blur',
                'gaussian_noise',
                'brightness_contrast_adjustment',
                'hue_saturation_value_shift',
                'lighting_variations',
                'compression_artifacts'
            ]
        }

        with open('dataset_expansion_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("📄 Dataset expansion report saved to: dataset_expansion_report.json")

        return report

def main():
    """Main function to expand dataset"""
    print("🚀 Starting Dataset Expansion")
    print("=" * 50)

    expander = DatasetExpander()

    # Step 1: Analyze current dataset
    print("\n1️⃣ Analyzing current dataset...")
    real_stats, fake_stats = expander.analyze_current_dataset()

    # Step 2: Expand dataset
    print("\n2️⃣ Expanding dataset...")
    expander.expand_dataset(target_samples=1500)  # Expand to 1500 samples each

    # Step 3: Add diverse sources
    print("\n3️⃣ Adding diverse data sources...")
    expander.add_diverse_sources()

    # Step 4: Validate expanded dataset
    print("\n4️⃣ Validating expanded dataset...")
    real_count, fake_count = expander.validate_expanded_dataset()

    # Step 5: Create report
    print("\n5️⃣ Creating dataset report...")
    report = expander.create_dataset_report()

    print("\n" + "=" * 50)
    print("🎉 DATASET EXPANSION COMPLETED!")
    print(f"Original: {report['original_dataset']['real_count']} real, {report['original_dataset']['fake_count']} fake")
    print(f"Expanded: {report['expanded_dataset']['real_count']} real, {report['expanded_dataset']['fake_count']} fake")
    print(f"Expansion factor: {report['expanded_dataset']['real_count'] / max(report['original_dataset']['real_count'], 1):.1f}x")

    print("\n💡 Next Steps:")
    print("1. Train the enhanced model on the expanded dataset")
    print("2. Test on your specific video to see improvement")
    print("3. If still issues, consider collecting more real video data")
    print("4. Try fine-tuning on videos from similar domains")

if __name__ == "__main__":
    main()
