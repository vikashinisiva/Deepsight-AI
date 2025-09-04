import os, subprocess, glob, cv2, torch, numpy as np
from torchvision import transforms, models

# Test script to debug face detection and quality filtering

def is_good_quality_crop(img, min_size=50, max_size=500):
    """Quality filter matching training data"""
    if img is None or img.size == 0:
        return False, "Empty image"
    
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        return False, f"Too small: {w}x{h}"
    if h > max_size or w > max_size:
        return False, f"Too large: {w}x{h}"
    
    aspect_ratio = max(h, w) / min(h, w)
    if aspect_ratio > 2.0:
        return False, f"Bad aspect ratio: {aspect_ratio:.2f}"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return False, f"Too blurry: {laplacian_var:.1f}"
    
    mean_brightness = gray.mean()
    if mean_brightness < 30:
        return False, f"Too dark: {mean_brightness:.1f}"
    if mean_brightness > 220:
        return False, f"Too bright: {mean_brightness:.1f}"
    
    return True, f"Good: {w}x{h}, blur={laplacian_var:.1f}, bright={mean_brightness:.1f}"

def detect_faces_debug(img):
    """Debug face detection"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    
    print(f"  Found {len(faces)} faces")
    good_faces = []
    for i, (x, y, w, h) in enumerate(faces):
        crop = img[y:y+h, x:x+w]
        is_good, reason = is_good_quality_crop(crop)
        print(f"    Face {i+1}: {reason}")
        if is_good:
            good_faces.append((x, y, w, h))
    
    print(f"  {len(good_faces)} good quality faces")
    return good_faces

def debug_video_inference(video_path):
    """Debug what's happening in video inference"""
    print(f"Debugging: {video_path}")
    
    # Extract frames
    temp_dir = "_debug_frames"
    os.makedirs(temp_dir, exist_ok=True)
    for f in glob.glob(os.path.join(temp_dir, "*.jpg")): os.remove(f)
    
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-i", video_path, 
        "-vf", f"fps=1", "-frames:v", "5",  # Just 5 frames for debugging
        os.path.join(temp_dir, "frame_%03d.jpg")
    ], check=True)
    
    frames_with_faces = 0
    
    for fp in sorted(glob.glob(os.path.join(temp_dir, "*.jpg"))):
        frame_name = os.path.basename(fp)
        print(f"\nFrame: {frame_name}")
        
        img = cv2.imread(fp)
        if img is None: 
            print("  Could not read image")
            continue
        
        faces = detect_faces_debug(img)
        if faces:
            frames_with_faces += 1
    
    print(f"\nSummary: {frames_with_faces}/5 frames had good quality faces")
    
    # Cleanup
    for f in glob.glob(os.path.join(temp_dir, "*.jpg")): os.remove(f)

if __name__ == "__main__":
    print("=== DEBUGGING FACE DETECTION QUALITY FILTER ===")
    
    # Test on real video
    debug_video_inference("ffpp_data/real_videos/033.mp4")
    
    # Test on fake video
    debug_video_inference("ffpp_data/fake_videos/035_036.mp4")
