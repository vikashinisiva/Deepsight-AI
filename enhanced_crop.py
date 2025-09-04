import os, glob, cv2, numpy as np
from tqdm import tqdm

def improved_crop_dir(src, dst):
    """Enhanced face cropping using OpenCV DNN face detector"""
    os.makedirs(dst, exist_ok=True)
    
    # Load OpenCV DNN face detector (more accurate than Haar cascades)
    net = cv2.dnn.readNetFromTensorflow(
        cv2.data.haarcascades + 'opencv_face_detector_uint8.pb',
        cv2.data.haarcascades + 'opencv_face_detector.pbtxt'
    )
    
    total_frames = 0
    faces_found = 0
    crops_saved = 0
    
    for p in tqdm(sorted(glob.glob(os.path.join(src, "*.jpg"))), desc=f"cropping {src}"):
        total_frames += 1
        img = cv2.imread(p)
        if img is None: continue
        
        h, w = img.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123])
        net.setInput(blob)
        detections = net.forward()
        
        best_confidence = 0
        best_box = None
        
        # Find the most confident face detection
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > best_confidence:  # Minimum confidence threshold
                best_confidence = confidence
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                best_box = (x1, y1, x2, y2)
        
        if best_box is not None:
            faces_found += 1
            x1, y1, x2, y2 = best_box
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(os.path.join(dst, os.path.basename(p)), crop)
                crops_saved += 1
    
    print(f"\n{src} Results:")
    print(f"  Total frames: {total_frames}")
    print(f"  Faces detected: {faces_found} ({faces_found/total_frames*100:.1f}%)")
    print(f"  Crops saved: {crops_saved} ({crops_saved/total_frames*100:.1f}%)")
    return total_frames, faces_found, crops_saved

def fallback_crop_dir(src, dst):
    """Fallback using Haar cascades if DNN models are not available"""
    os.makedirs(dst, exist_ok=True)
    
    # Load Haar cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    total_frames = 0
    faces_found = 0
    crops_saved = 0
    
    for p in tqdm(sorted(glob.glob(os.path.join(src, "*.jpg"))), desc=f"cropping {src} (fallback)"):
        total_frames += 1
        img = cv2.imread(p)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        
        if len(faces) > 0:
            faces_found += 1
            # Take the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            crop = img[y:y+h, x:x+w]
            if crop.size > 0:
                cv2.imwrite(os.path.join(dst, os.path.basename(p)), crop)
                crops_saved += 1
    
    print(f"\n{src} Results (Haar cascade fallback):")
    print(f"  Total frames: {total_frames}")
    print(f"  Faces detected: {faces_found} ({faces_found/total_frames*100:.1f}%)")
    print(f"  Crops saved: {crops_saved} ({crops_saved/total_frames*100:.1f}%)")
    return total_frames, faces_found, crops_saved

def crop_dir_with_stats(src, dst):
    """Try improved detection first, fallback if needed"""
    try:
        return improved_crop_dir(src, dst)
    except Exception as e:
        print(f"DNN detector failed ({e}), using Haar cascade fallback...")
        return fallback_crop_dir(src, dst)

if __name__ == "__main__":
    print("Enhanced face cropping starting...")
    
    # Create new crops directories with improved detection
    enhanced_crops_dir = "crops_enhanced"
    os.makedirs(f"{enhanced_crops_dir}/real", exist_ok=True)
    os.makedirs(f"{enhanced_crops_dir}/fake", exist_ok=True)
    
    print(f"Cropping to {enhanced_crops_dir}/")
    
    # Process both real and fake frames
    real_stats = crop_dir_with_stats("frames/real", f"{enhanced_crops_dir}/real")
    fake_stats = crop_dir_with_stats("frames/fake", f"{enhanced_crops_dir}/fake")
    
    # Summary
    total_frames = real_stats[0] + fake_stats[0]
    total_crops = real_stats[2] + fake_stats[2]
    
    print(f"\n=== ENHANCED CROPPING SUMMARY ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Total crops saved: {total_crops}")
    print(f"Overall success rate: {total_crops/total_frames*100:.1f}%")
    
    # Compare with original MTCNN results
    original_real = len(glob.glob("crops/real/*.jpg"))
    original_fake = len(glob.glob("crops/fake/*.jpg"))
    original_total = original_real + original_fake
    
    print(f"\n=== COMPARISON WITH MTCNN ===")
    print(f"Original MTCNN crops: {original_total}")
    print(f"Enhanced detection crops: {total_crops}")
    print(f"Improvement: {total_crops - original_total:+d} crops ({(total_crops/original_total-1)*100:+.1f}%)")
    print(f"Ready for training with improved face detection!")
