import os, glob, cv2, numpy as np
from tqdm import tqdm
import mediapipe as mp

def mediapipe_crop_dir(src, dst):
    """Enhanced face cropping using MediaPipe face detection"""
    os.makedirs(dst, exist_ok=True)
    
    # Initialize MediaPipe face detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    total_frames = 0
    faces_found = 0
    crops_saved = 0
    
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        for p in tqdm(sorted(glob.glob(os.path.join(src, "*.jpg"))), desc=f"cropping {src} (MediaPipe)"):
            total_frames += 1
            img = cv2.imread(p)
            if img is None: continue
            
            h, w, _ = img.shape
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image and get face detections
            results = face_detection.process(rgb_img)
            
            if results.detections:
                faces_found += 1
                
                # Take the most confident detection (they're sorted by confidence)
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + w_box)
                y2 = min(h, y + h_box)
                
                crop = img[y:y2, x:x2]
                if crop.size > 0:
                    cv2.imwrite(os.path.join(dst, os.path.basename(p)), crop)
                    crops_saved += 1
    
    print(f"\n{src} Results (MediaPipe):")
    print(f"  Total frames: {total_frames}")
    print(f"  Faces detected: {faces_found} ({faces_found/total_frames*100:.1f}%)")
    print(f"  Crops saved: {crops_saved} ({crops_saved/total_frames*100:.1f}%)")
    return total_frames, faces_found, crops_saved

def compare_detection_methods():
    """Compare different face detection methods on a small sample"""
    test_dir = "frames/real"
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))[:100]  # Test on first 100 images
    
    print("=== COMPARING FACE DETECTION METHODS ===")
    print(f"Testing on {len(test_files)} sample images...")
    
    # MTCNN (if available)
    try:
        from mtcnn import MTCNN
        mtcnn_detector = MTCNN()
        mtcnn_count = 0
        for p in test_files:
            img = cv2.imread(p)
            if img is None: continue
            faces = mtcnn_detector.detect_faces(img)
            if faces: mtcnn_count += 1
        print(f"MTCNN: {mtcnn_count}/{len(test_files)} ({mtcnn_count/len(test_files)*100:.1f}%)")
    except:
        print("MTCNN: Not available")
    
    # Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    haar_count = 0
    for p in test_files:
        img = cv2.imread(p)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        if len(faces) > 0: haar_count += 1
    print(f"Haar Cascade: {haar_count}/{len(test_files)} ({haar_count/len(test_files)*100:.1f}%)")
    
    # MediaPipe
    mp_face_detection = mp.solutions.face_detection
    mp_count = 0
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        for p in test_files:
            img = cv2.imread(p)
            if img is None: continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_img)
            if results.detections: mp_count += 1
    print(f"MediaPipe: {mp_count}/{len(test_files)} ({mp_count/len(test_files)*100:.1f}%)")

if __name__ == "__main__":
    print("MediaPipe face detection cropping starting...")
    
    # First, compare methods
    compare_detection_methods()
    
    # Create new crops directories with MediaPipe
    mp_crops_dir = "crops_mediapipe"
    os.makedirs(f"{mp_crops_dir}/real", exist_ok=True)
    os.makedirs(f"{mp_crops_dir}/fake", exist_ok=True)
    
    print(f"\nCropping to {mp_crops_dir}/")
    
    # Process both real and fake frames
    real_stats = mediapipe_crop_dir("frames/real", f"{mp_crops_dir}/real")
    fake_stats = mediapipe_crop_dir("frames/fake", f"{mp_crops_dir}/fake")
    
    # Summary
    total_frames = real_stats[0] + fake_stats[0]
    total_crops = real_stats[2] + fake_stats[2]
    
    print(f"\n=== MEDIAPIPE CROPPING SUMMARY ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Total crops saved: {total_crops}")
    print(f"Overall success rate: {total_crops/total_frames*100:.1f}%")
    
    # Compare with original MTCNN results
    original_real = len(glob.glob("crops/real/*.jpg"))
    original_fake = len(glob.glob("crops/fake/*.jpg"))
    original_total = original_real + original_fake
    
    print(f"\n=== COMPARISON WITH MTCNN ===")
    print(f"Original MTCNN crops: {original_total}")
    print(f"MediaPipe detection crops: {total_crops}")
    print(f"Improvement: {total_crops - original_total:+d} crops ({(total_crops/original_total-1)*100:+.1f}%)")
    print(f"Ready for training with MediaPipe face detection!")
