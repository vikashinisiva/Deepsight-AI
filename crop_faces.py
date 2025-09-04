import os, glob, cv2
from tqdm import tqdm
from mtcnn import MTCNN

def crop_dir(src, dst):
    os.makedirs(dst, exist_ok=True)
    det = MTCNN()
    imgs = sorted(glob.glob(os.path.join(src, "*.jpg")))
    for p in tqdm(imgs, desc=f"cropping {src}"):
        img = cv2.imread(p)
        if img is None: continue
        faces = det.detect_faces(img)
        if not faces: continue
        # largest face
        x,y,w,h = max([f['box'] for f in faces], key=lambda b: b[2]*b[3])
        x,y = max(0,x), max(0,y)
        crop = img[y:y+h, x:x+w]
        if crop.size:
            cv2.imwrite(os.path.join(dst, os.path.basename(p)), crop)

if __name__ == "__main__":
    crop_dir("frames/real", "crops/real")
    crop_dir("frames/fake", "crops/fake")
