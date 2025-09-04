# cam_on_video.py
import os, sys, glob, subprocess, cv2, torch, numpy as np
from torchvision import transforms, models
try:
    from mtcnn import MTCNN
    use_mtcnn = True
except ImportError:
    use_mtcnn = False
    print("MTCNN not available, using OpenCV face detection")

from grad_cam import GradCAM, overlay_cam_on_image, make_infer_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load your model (matching train_baseline.py) ====
m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for p in m.features.parameters(): p.requires_grad = False
m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
m.load_state_dict(torch.load("weights/baseline.pth", map_location=device))

# For Grad-CAM, we need gradients on the last conv layer
for p in m.features[-1].parameters():
    p.requires_grad = True

m.eval().to(device)

# Hook last conv block for Grad-CAM
target_layer = m.features[-1]
cam = GradCAM(m, target_layer)
tfm = make_infer_transform()

# Face detection setup
if use_mtcnn:
    det = MTCNN()
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    """Detect faces using MTCNN or OpenCV"""
    if use_mtcnn:
        faces = det.detect_faces(img)
        return [f["box"] for f in faces]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, w, h) for (x, y, w, h) in faces]

def prob_fake_on_crop(bgr_crop):
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    ten = tfm(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = m(ten)
        p_fake = torch.softmax(logits, dim=1)[0,1].item()
    return p_fake, ten  # also return tensor for CAM

def extract_frames(video_path, out_dir, fps=1):
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.jpg")): os.remove(f)
    subprocess.run(["ffmpeg","-loglevel","error","-i",video_path,"-r",str(fps),os.path.join(out_dir,"f_%03d.jpg")])

def run(video_path):
    tmp = "_cam_frames"
    extract_frames(video_path, tmp, fps=1)

    # 1) Score all frames, keep face boxes + tensors for the best frame
    best = {"p": -1, "img": None, "box": None, "ten": None, "path": None}
    for fp in sorted(glob.glob(os.path.join(tmp,"*.jpg"))):
        img = cv2.imread(fp)
        if img is None: continue
        faces = detect_faces(img)
        if not faces: continue
        
        # take largest face
        x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
        x,y = max(0,x), max(0,y)
        crop = img[y:y+h, x:x+w]
        if crop.size == 0: continue

        p_fake, ten = prob_fake_on_crop(crop)
        if p_fake > best["p"]:
            best = {"p": p_fake, "img": img.copy(), "box": (x,y,w,h), "ten": ten, "path": fp}

    if best["p"] < 0:
        print("No face found.")
        return None

    # 2) Grad-CAM on best face crop for FAKE class (index 1)
    # Forward already done for 'ten' in prob_fake_on_crop; call CAM and resize
    cam_map = cam(best["ten"], class_idx=1)  # (Hc,Wc) in 224x224 space
    # Resize CAM to crop size
    crop_h, crop_w = best["box"][3], best["box"][2]
    cam_map = cv2.resize(cam_map, (crop_w, crop_h))
    cam_map = np.clip(cam_map, 0, 1)

    # 3) Paste overlay only inside the face box
    img = best["img"]
    x,y,w,h = best["box"]
    face_region = img[y:y+h, x:x+w]
    overlay_face = overlay_cam_on_image(face_region, cam_map, alpha=0.45)
    out_img = img.copy()
    out_img[y:y+h, x:x+w] = overlay_face

    # Draw rectangle + score
    cv2.rectangle(out_img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(out_img, f"Fake prob: {best['p']:.2f}", (x, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Save
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = f"gradcam_{base}.jpg"
    cv2.imwrite(out_path, out_img)
    print(f"Saved Grad-CAM overlay: {out_path}")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cam_on_video.py <video_path>")
        sys.exit(1)
    
    try:
        result = run(sys.argv[1])
        if result:
            print(f"Grad-CAM visualization saved to: {result}")
        else:
            print("Failed to process video - no faces detected.")
    finally:
        cam.remove_hooks()
