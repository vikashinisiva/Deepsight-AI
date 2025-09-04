import os, glob, cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

def crop_dir(src, dst):
    os.makedirs(dst, exist_ok=True)
    app = FaceAnalysis(name="buffalo_l")  # accurate, still fast
    app.prepare(ctx_id=0 if cv2.cuda.getCudaEnabledDeviceCount()>0 else -1, det_size=(640,640))

    for p in tqdm(sorted(glob.glob(os.path.join(src, "*.jpg"))), desc=f"crop {src}"):
        img = cv2.imread(p); 
        if img is None: continue
        faces = app.get(img)
        if not faces: continue
        # largest face
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        x1,y1,x2,y2 = map(int, f.bbox)
        x1,y1 = max(0,x1), max(0,y1)
        crop = img[y1:y2, x1:x2]
        if crop.size: cv2.imwrite(os.path.join(dst, os.path.basename(p)), crop)

def crop_dir_with_stats(src, dst):
    """Enhanced version with detailed statistics"""
    os.makedirs(dst, exist_ok=True)
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if cv2.cuda.getCudaEnabledDeviceCount()>0 else -1, det_size=(640,640))
    
    total_frames = 0
    faces_found = 0
    crops_saved = 0
    
    for p in tqdm(sorted(glob.glob(os.path.join(src, "*.jpg"))), desc=f"cropping {src}"):
        total_frames += 1
        img = cv2.imread(p)
        if img is None: continue
        
        faces = app.get(img)
        if not faces: continue
        
        faces_found += 1
        # largest face
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        x1,y1,x2,y2 = map(int, f.bbox)
        x1,y1 = max(0,x1), max(0,y1)
        crop = img[y1:y2, x1:x2]
        
        if crop.size > 0:
            cv2.imwrite(os.path.join(dst, os.path.basename(p)), crop)
            crops_saved += 1
    
    print(f"\n{src} Results:")
    print(f"  Total frames: {total_frames}")
    print(f"  Faces detected: {faces_found} ({faces_found/total_frames*100:.1f}%)")
    print(f"  Crops saved: {crops_saved} ({crops_saved/total_frames*100:.1f}%)")
    return total_frames, faces_found, crops_saved

if __name__ == "__main__":
    print("RetinaFace face cropping starting...")
    
    # Create new crops directories with RetinaFace
    retina_crops_dir = "crops_retina"
    os.makedirs(f"{retina_crops_dir}/real", exist_ok=True)
    os.makedirs(f"{retina_crops_dir}/fake", exist_ok=True)
    
    print(f"Cropping to {retina_crops_dir}/")
    
    # Process both real and fake frames
    real_stats = crop_dir_with_stats("frames/real", f"{retina_crops_dir}/real")
    fake_stats = crop_dir_with_stats("frames/fake", f"{retina_crops_dir}/fake")
    
    # Summary
    total_frames = real_stats[0] + fake_stats[0]
    total_crops = real_stats[2] + fake_stats[2]
    
    print(f"\n=== RETINAFACE CROPPING SUMMARY ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Total crops saved: {total_crops}")
    print(f"Overall success rate: {total_crops/total_frames*100:.1f}%")
    print(f"Ready for training with improved face detection!")
