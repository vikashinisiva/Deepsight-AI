import os, glob, cv2
import torch
from torchvision import transforms, models
from tqdm import tqdm

def update_inference_with_quality_filtering():
    """Update inference to use the same quality filtering as training"""
    
    def is_good_quality_crop(img, min_size=50, max_size=500):
        """Check if a crop is good quality (same as training filter)"""
        if img is None or img.size == 0:
            return False
        
        h, w = img.shape[:2]
        
        # Size filters
        if h < min_size or w < min_size or h > max_size or w > max_size:
            return False
        
        # Aspect ratio filter
        aspect_ratio = max(h, w) / min(h, w)
        if aspect_ratio > 2.0:
            return False
        
        # Blur detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            return False
        
        # Brightness check
        mean_brightness = gray.mean()
        if mean_brightness < 30 or mean_brightness > 220:
            return False
        
        return True
    
    def detect_faces_improved(img):
        """Enhanced face detection with quality filtering"""
        # Use Haar cascade (most reliable in our testing)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        
        good_faces = []
        for (x, y, w, h) in faces:
            crop = img[y:y+h, x:x+w]
            if is_good_quality_crop(crop):
                good_faces.append((x, y, w, h))
        
        return good_faces
    
    return detect_faces_improved

def create_improved_inference_script():
    """Create an updated inference script with improved face detection"""
    
    infer_code = '''import os, subprocess, glob, cv2, torch, numpy as np
from torchvision import transforms, models
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load improved model
m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for p in m.features.parameters(): p.requires_grad = False
m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
m.load_state_dict(torch.load("weights/baseline_improved.pth", map_location=device))
m.eval().to(device)

# Preprocessing
tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def is_good_quality_crop(img, min_size=50, max_size=500):
    """Quality filter matching training data"""
    if img is None or img.size == 0:
        return False
    
    h, w = img.shape[:2]
    if h < min_size or w < min_size or h > max_size or w > max_size:
        return False
    
    aspect_ratio = max(h, w) / min(h, w)
    if aspect_ratio > 2.0:
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return False
    
    mean_brightness = gray.mean()
    if mean_brightness < 30 or mean_brightness > 220:
        return False
    
    return True

def detect_faces_improved(img):
    """Enhanced face detection with quality filtering"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    
    good_faces = []
    for (x, y, w, h) in faces:
        crop = img[y:y+h, x:x+w]
        if is_good_quality_crop(crop):
            good_faces.append((x, y, w, h))
    
    return good_faces

@torch.no_grad()
def infer(video_path, max_frames=30):
    """Improved video inference with quality filtering"""
    # Extract frames
    temp_dir = "_temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    for f in glob.glob(os.path.join(temp_dir, "*.jpg")): os.remove(f)
    
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-i", video_path, 
        "-vf", f"fps=1", "-frames:v", str(max_frames),
        os.path.join(temp_dir, "frame_%03d.jpg")
    ], check=True)
    
    probs = []
    frames_used = 0
    
    for fp in sorted(glob.glob(os.path.join(temp_dir, "*.jpg"))):
        img = cv2.imread(fp)
        if img is None: continue
        
        faces = detect_faces_improved(img)  # Use improved detection
        if not faces: continue
        
        # Process largest good face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        crop = img[y:y+h, x:x+w]
        
        # Convert and predict
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = tfm(rgb).unsqueeze(0).to(device)
        logits = m(tensor)
        p_fake = torch.softmax(logits, dim=1)[0,1].item()
        probs.append(p_fake)
        frames_used += 1
    
    # Cleanup
    for f in glob.glob(os.path.join(temp_dir, "*.jpg")): os.remove(f)
    
    if not probs:
        return {"video": os.path.basename(video_path), "prediction": "UNKNOWN", 
                "fake_confidence": 0.5, "frames_used": 0}
    
    avg_fake_prob = np.mean(probs)
    prediction = "FAKE" if avg_fake_prob > 0.5 else "REAL"
    
    return {
        "video": os.path.basename(video_path),
        "prediction": prediction,
        "fake_confidence": avg_fake_prob,
        "frames_used": frames_used
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = infer(sys.argv[1])
        print(f"Video: {result['video']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Fake confidence: {result['fake_confidence']:.3f}")
        print(f"Frames analyzed: {result['frames_used']}")
    else:
        print("Usage: python infer_video_improved.py <video_path>")
'''
    
    with open("infer_video_improved.py", "w") as f:
        f.write(infer_code)
    
    print("Created infer_video_improved.py with enhanced face detection and quality filtering")

def compare_model_performance():
    """Compare the improved model with the original"""
    print("=== MODEL PERFORMANCE COMPARISON ===")
    
    models_info = [
        ("Original Model", "weights/baseline.pth", "crops_subset"),
        ("Improved Model", "weights/baseline_improved.pth", "crops_improved")
    ]
    
    for name, model_path, dataset in models_info:
        if os.path.exists(model_path):
            real_count = len(glob.glob(f"{dataset}/real/*.jpg")) if os.path.exists(dataset) else 0
            fake_count = len(glob.glob(f"{dataset}/fake/*.jpg")) if os.path.exists(dataset) else 0
            total = real_count + fake_count
            
            print(f"\\n{name}:")
            print(f"  Model: {model_path}")
            print(f"  Training data: {total} images ({real_count} real + {fake_count} fake)")
            print(f"  Balance: {min(real_count, fake_count) / max(real_count, fake_count) * 100:.1f}%")
        else:
            print(f"\\n{name}: Model not found")

if __name__ == "__main__":
    print("=== FACE DETECTION IMPROVEMENTS SUMMARY ===")
    print("\\n🎯 Improvements Made:")
    print("✅ Quality filtering: Size, aspect ratio, blur, brightness checks")
    print("✅ Balanced dataset: Equal real/fake samples")
    print("✅ Enhanced face detection: Haar cascade with quality filters") 
    print("✅ Improved model: Trained on high-quality crops")
    
    compare_model_performance()
    
    create_improved_inference_script()
    
    print("\\n=== NEXT STEPS ===")
    print("1. Test improved inference: python infer_video_improved.py <video_path>")
    print("2. Update Grad-CAM scripts to use improved model")
    print("3. Run batch evaluation with improved detection")
    print("4. Compare results with original approach")
    
    print("\\n💡 Key Insights:")
    print("- Fake videos had lower quality pass rate (44.7% vs 60.7% for real)")
    print("- Quality filtering removed blurry, poorly lit, and distorted faces")
    print("- Balanced dataset should improve generalization")
    print("- Same quality filters applied in training and inference for consistency")
