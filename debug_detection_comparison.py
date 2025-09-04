import os, subprocess, glob, cv2
from mtcnn import MTCNN

def compare_face_detection():
    """Compare face detection between MTCNN and enhanced methods"""
    
    print("🔍 COMPARING FACE DETECTION METHODS")
    print("="*50)
    
    # Test on one real video
    test_video = "ffpp_data/real_videos/033.mp4"
    
    # Extract a few frames for comparison
    temp_dir = "_compare_frames"
    os.makedirs(temp_dir, exist_ok=True)
    for f in glob.glob(os.path.join(temp_dir, "*.jpg")): os.remove(f)
    
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-i", test_video, 
        "-vf", "fps=1", "-frames:v", "3",
        os.path.join(temp_dir, "frame_%03d.jpg")
    ], check=True)
    
    # Initialize detectors
    mtcnn_detector = MTCNN()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for frame_file in sorted(glob.glob(os.path.join(temp_dir, "*.jpg"))):
        print(f"\n📷 {os.path.basename(frame_file)}:")
        
        img = cv2.imread(frame_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # MTCNN detection
        mtcnn_faces = mtcnn_detector.detect_faces(img)
        print(f"  MTCNN: {len(mtcnn_faces)} faces")
        for i, face in enumerate(mtcnn_faces):
            x, y, w, h = face['box']
            conf = face['confidence']
            print(f"    Face {i+1}: {w}x{h} at ({x},{y}), confidence: {conf:.3f}")
        
        # Enhanced detection (from final script)
        haar_faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        enhanced_faces = [(x,y,w,h) for x,y,w,h in haar_faces if w >= 60 and h >= 60]
        print(f"  Enhanced: {len(enhanced_faces)} faces")
        for i, (x, y, w, h) in enumerate(enhanced_faces):
            print(f"    Face {i+1}: {w}x{h} at ({x},{y})")
        
        # Check if they find the same faces
        if len(mtcnn_faces) != len(enhanced_faces):
            print(f"  ⚠️  DIFFERENT NUMBER OF FACES DETECTED!")
    
    # Cleanup
    for f in glob.glob(os.path.join(temp_dir, "*.jpg")): os.remove(f)

def check_model_weights():
    """Check if model weights are loaded correctly"""
    import torch
    from torchvision import models
    
    print("\n🎯 CHECKING MODEL WEIGHTS")
    print("="*30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load both models
    original_weights = torch.load("weights/baseline.pth", map_location=device)
    improved_weights = torch.load("weights/baseline_improved.pth", map_location=device)
    
    # Check classifier weights
    orig_classifier = original_weights['classifier.1.weight']
    impr_classifier = improved_weights['classifier.1.weight']
    
    print(f"Original classifier weights shape: {orig_classifier.shape}")
    print(f"Improved classifier weights shape: {impr_classifier.shape}")
    print(f"Weights are different: {not torch.equal(orig_classifier, impr_classifier)}")
    
    # Check means (rough indication of bias)
    print(f"Original classifier bias toward class 1 (FAKE): {original_weights['classifier.1.bias'][1].item():.3f}")
    print(f"Improved classifier bias toward class 1 (FAKE): {improved_weights['classifier.1.bias'][1].item():.3f}")

if __name__ == "__main__":
    compare_face_detection()
    check_model_weights()
