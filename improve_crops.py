import os, glob, cv2, numpy as np, shutil
from tqdm import tqdm

def improved_crop_selection():
    """
    Instead of changing the face detector, let's improve the crop selection:
    1. Filter out very small or very large faces (likely false positives)
    2. Filter out low quality/blurry crops
    3. Create a balanced, high-quality training subset
    """
    
    def is_good_quality_crop(img_path, min_size=50, max_size=500):
        """Check if a crop is good quality"""
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        
        # Size filters
        if h < min_size or w < min_size or h > max_size or w > max_size:
            return False
        
        # Aspect ratio filter (faces should be roughly square-ish)
        aspect_ratio = max(h, w) / min(h, w)
        if aspect_ratio > 2.0:  # Too elongated
            return False
        
        # Blur detection using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Too blurry
            return False
        
        # Brightness check (avoid too dark or too bright)
        mean_brightness = np.mean(gray)
        if mean_brightness < 30 or mean_brightness > 220:
            return False
        
        return True
    
    print("=== IMPROVING CROP QUALITY ===")
    
    # Analyze current crops
    real_crops = glob.glob("crops/real/*.jpg")
    fake_crops = glob.glob("crops/fake/*.jpg")
    
    print(f"Original crops: {len(real_crops)} real, {len(fake_crops)} fake")
    
    # Filter for quality
    good_real = [p for p in tqdm(real_crops, desc="Filtering real crops") if is_good_quality_crop(p)]
    good_fake = [p for p in tqdm(fake_crops, desc="Filtering fake crops") if is_good_quality_crop(p)]
    
    print(f"Good quality crops: {len(good_real)} real ({len(good_real)/len(real_crops)*100:.1f}%), {len(good_fake)} fake ({len(good_fake)/len(fake_crops)*100:.1f}%)")
    
    # Create improved training set
    improved_dir = "crops_improved"
    os.makedirs(f"{improved_dir}/real", exist_ok=True)
    os.makedirs(f"{improved_dir}/fake", exist_ok=True)
    
    # Balance the dataset - take equal numbers
    max_per_class = min(len(good_real), len(good_fake), 1500)  # Cap at 1500 per class for faster training
    
    # Randomly sample to balance
    import random
    random.seed(42)  # For reproducibility
    selected_real = random.sample(good_real, max_per_class)
    selected_fake = random.sample(good_fake, max_per_class)
    
    # Copy selected crops
    for src_path in tqdm(selected_real, desc="Copying improved real crops"):
        dst_path = os.path.join(f"{improved_dir}/real", os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
    
    for src_path in tqdm(selected_fake, desc="Copying improved fake crops"):
        dst_path = os.path.join(f"{improved_dir}/fake", os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
    
    print(f"\n=== IMPROVED DATASET SUMMARY ===")
    print(f"Improved dataset: {max_per_class} real + {max_per_class} fake = {max_per_class * 2} total")
    print(f"Quality filters applied:")
    print(f"  - Size: 50-500 pixels")
    print(f"  - Aspect ratio: < 2.0")
    print(f"  - Blur detection: Laplacian variance > 100")
    print(f"  - Brightness: 30-220 range")
    print(f"  - Balanced sampling for equal class distribution")
    
    return max_per_class * 2

def analyze_crop_statistics():
    """Analyze the statistics of different crop datasets"""
    datasets = {
        "Original MTCNN": "crops",
        "Enhanced OpenCV": "crops_enhanced", 
        "Improved Quality": "crops_improved"
    }
    
    print("\n=== CROP DATASET COMPARISON ===")
    for name, path in datasets.items():
        if os.path.exists(path):
            real_count = len(glob.glob(f"{path}/real/*.jpg"))
            fake_count = len(glob.glob(f"{path}/fake/*.jpg"))
            total = real_count + fake_count
            print(f"{name:20}: {real_count:4d} real + {fake_count:4d} fake = {total:4d} total")
        else:
            print(f"{name:20}: Not available")

if __name__ == "__main__":
    print("Starting improved crop selection and quality filtering...")
    
    # Create improved quality dataset
    total_improved = improved_crop_selection()
    
    # Show comparison
    analyze_crop_statistics()
    
    print(f"\n=== RECOMMENDATION ===")
    print(f"For best results, retrain your model using 'crops_improved' directory")
    print(f"This provides:")
    print(f"  ✅ Higher quality face crops")
    print(f"  ✅ Balanced dataset (equal real/fake)")
    print(f"  ✅ Filtered out poor quality images")
    print(f"  ✅ Faster training with {total_improved} high-quality samples")
    
    print(f"\nTo retrain with improved dataset:")
    print(f"1. Update train_baseline.py to use 'crops_improved' instead of 'crops_subset'")
    print(f"2. Run training again for better performance")
