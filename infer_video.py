import os, torch, cv2
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

# Load the trained model
def load_model(model_path="weights/baseline.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the same model architecture as training
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for p in model.features.parameters(): 
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Initialize OpenCV face detector (more reliable than MTCNN for this use case)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces_from_video(video_path, max_frames=30):
    """Extract faces from video frames"""
    faces = []
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return faces
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // max_frames)  # Sample frames evenly
    
    frame_idx = 0
    while len(faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_step == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using OpenCV
            try:
                gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                faces_detected = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces_detected) > 0:
                    # Take the largest face
                    largest_face = max(faces_detected, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    
                    # Extract face region
                    face_img = rgb_frame[y:y+h, x:x+w]
                    # Resize to 160x160 for consistency
                    face_resized = cv2.resize(face_img, (160, 160))
                    faces.append(face_resized)
            except:
                pass  # Skip frames with detection errors
                
        frame_idx += 1
    
    cap.release()
    return faces

def infer(video_path):
    """
    Infer if a video is real or fake
    Returns dict with prediction results
    """
    try:
        # Load model
        model, device = load_model()
        
        # Extract faces from video
        faces = extract_faces_from_video(video_path, max_frames=30)
        
        if not faces:
            return {
                "video": os.path.basename(video_path),
                "prediction": "UNKNOWN",
                "fake_confidence": 0.5,
                "frames_used": 0
            }
        
        # Process each face
        predictions = []
        
        with torch.no_grad():
            for face in faces:
                # Preprocess face
                face_tensor = transform(face).unsqueeze(0).to(device)
                
                # Get prediction
                output = model(face_tensor)
                prob = torch.softmax(output, dim=1)
                fake_conf = prob[0][0].item()  # Index 0 is 'fake' class
                
                predictions.append(fake_conf)
        
        # Average predictions across all faces
        avg_fake_conf = np.mean(predictions)
        prediction = "FAKE" if avg_fake_conf > 0.5 else "REAL"
        
        return {
            "video": os.path.basename(video_path),
            "prediction": prediction,
            "fake_confidence": float(avg_fake_conf),
            "frames_used": len(faces)
        }
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return {
            "video": os.path.basename(video_path),
            "prediction": "ERROR",
            "fake_confidence": 0.5,
            "frames_used": 0
        }

if __name__ == "__main__":
    # Test on a single video
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        result = infer(video_path)
        print(f"Video: {result['video']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Fake Confidence: {result['fake_confidence']:.3f}")
        print(f"Frames Used: {result['frames_used']}")
    else:
        print("Usage: python infer_video.py <video_path>")
