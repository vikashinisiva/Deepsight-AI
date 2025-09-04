import streamlit as st
import cv2, torch, numpy as np
import glob, os, subprocess, tempfile
from torchvision import transforms, models
from PIL import Image
import plotly.graph_objects as go
from grad_cam import GradCAM, overlay_cam_on_image, make_infer_transform

# Set page config
st.set_page_config(
    page_title="DeepSight AI - Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for p in m.features.parameters(): p.requires_grad = False
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load("weights/baseline.pth", map_location=device))
    
    # For Grad-CAM, we need gradients on the last conv layer
    for p in m.features[-1].parameters():
        p.requires_grad = True
    
    m.eval().to(device)
    return m, device

@st.cache_resource
def setup_gradcam(_model):
    target_layer = _model.features[-1]
    cam = GradCAM(_model, target_layer)
    return cam

# Face detection using OpenCV
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return [(x, y, w, h) for (x, y, w, h) in faces]

def extract_frames(video_path, out_dir, fps=1):
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.jpg")): 
        os.remove(f)
    subprocess.run(["ffmpeg","-loglevel","error","-i",video_path,"-r",str(fps),os.path.join(out_dir,"f_%03d.jpg")])

def analyze_video(video_path, model, device, face_cascade, cam_analyzer=None, show_gradcam=False):
    tfm = make_infer_transform()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extract frames
        extract_frames(video_path, tmp_dir, fps=1)
        
        probs = []
        best_frame_data = {"p": -1, "img": None, "box": None, "tensor": None}
        
        for fp in sorted(glob.glob(os.path.join(tmp_dir, "*.jpg"))):
            img = cv2.imread(fp)
            if img is None: continue
            
            faces = detect_faces(img, face_cascade)
            if not faces: continue
            
            # Take largest face
            x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
            x, y = max(0, x), max(0, y)
            crop = img[y:y+h, x:x+w]
            if crop.size == 0: continue
            
            # Get prediction
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = tfm(rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor)
                p_fake = torch.softmax(logits, dim=1)[0,1].item()
            
            probs.append(p_fake)
            
            # Keep track of most suspicious frame for Grad-CAM
            if p_fake > best_frame_data["p"]:
                best_frame_data = {
                    "p": p_fake, 
                    "img": img.copy(), 
                    "box": (x, y, w, h), 
                    "tensor": tensor
                }
        
        if not probs:
            return None, None, None
        
        # Generate prediction
        avg_fake_prob = np.mean(probs)
        prediction = "FAKE" if avg_fake_prob > 0.5 else "REAL"
        
        # Generate visualization
        viz_img = None
        if best_frame_data["img"] is not None:
            img = best_frame_data["img"]
            x, y, w, h = best_frame_data["box"]
            
            # Draw face box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"Fake prob: {best_frame_data['p']:.2f}", (x, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add Grad-CAM if requested
            if show_gradcam and cam_analyzer is not None:
                try:
                    cam_map = cam_analyzer(best_frame_data["tensor"], class_idx=1)
                    cam_map = cv2.resize(cam_map, (w, h))
                    cam_map = np.clip(cam_map, 0, 1)
                    
                    face_region = img[y:y+h, x:x+w]
                    overlay = overlay_cam_on_image(face_region, cam_map, alpha=0.45)
                    img[y:y+h, x:x+w] = overlay
                except Exception as e:
                    st.warning(f"Grad-CAM failed: {e}")
            
            viz_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return {
            "prediction": prediction,
            "fake_confidence": avg_fake_prob,
            "frames_analyzed": len(probs),
            "probability_distribution": probs
        }, viz_img, best_frame_data["p"]

def main():
    st.title("🔍 DeepSight AI - Deepfake Detection")
    st.markdown("### Advanced AI-powered deepfake detection with explainable AI")
    
    # Load model and setup
    model, device = load_model()
    face_cascade = load_face_detector()
    cam_analyzer = setup_gradcam(model)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Analysis Settings")
    show_gradcam = st.sidebar.toggle("🔥 Show Grad-CAM Heatmap", value=True, 
                                    help="Visualize what the AI focuses on when making decisions")
    show_confidence = st.sidebar.toggle("📊 Show Confidence Distribution", value=True)
    
    # Device info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**🚀 Device:** {device}")
    st.sidebar.markdown(f"**🧠 Model:** EfficientNet-B0")
    st.sidebar.markdown(f"**🎯 Accuracy:** 75.45%")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video to analyze for deepfake content"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Display video info
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            st.video(uploaded_file)
            
            # Analyze button
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("🧠 AI is analyzing the video..."):
                    result, viz_img, max_fake_prob = analyze_video(
                        video_path, model, device, face_cascade, 
                        cam_analyzer if show_gradcam else None, show_gradcam
                    )
                
                if result is None:
                    st.error("❌ No faces detected in the video")
                else:
                    # Store results in session state
                    st.session_state.result = result
                    st.session_state.viz_img = viz_img
                    st.session_state.max_fake_prob = max_fake_prob
            
            # Clean up temp file
            try:
                os.unlink(video_path)
            except:
                pass
    
    with col2:
        st.header("📊 Analysis Results")
        
        if hasattr(st.session_state, 'result') and st.session_state.result:
            result = st.session_state.result
            viz_img = st.session_state.viz_img
            max_fake_prob = st.session_state.max_fake_prob
            
            # Main prediction
            prediction = result["prediction"]
            confidence = result["fake_confidence"]
            
            # Color-coded prediction
            if prediction == "FAKE":
                st.error(f"🚨 **DEEPFAKE DETECTED** (Confidence: {confidence:.1%})")
            else:
                st.success(f"✅ **AUTHENTIC VIDEO** (Fake probability: {confidence:.1%})")
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("🎯 Prediction", prediction)
            with col_b:
                st.metric("📈 Confidence", f"{confidence:.1%}")
            with col_c:
                st.metric("🖼️ Frames Analyzed", result["frames_analyzed"])
            
            # Visualization
            if viz_img is not None:
                st.subheader("🔍 Most Suspicious Frame")
                caption = f"Frame with highest fake probability: {max_fake_prob:.1%}"
                if show_gradcam:
                    caption += " (with Grad-CAM heatmap)"
                st.image(viz_img, caption=caption, use_column_width=True)
                
                if show_gradcam:
                    st.info("🔥 **Grad-CAM Explanation:** Red/yellow areas show where the AI detected potential deepfake artifacts (texture inconsistencies, blending artifacts, unnatural features)")
            
            # Confidence distribution
            if show_confidence and len(result["probability_distribution"]) > 1:
                st.subheader("📊 Frame-by-Frame Analysis")
                
                probs = result["probability_distribution"]
                frames = list(range(1, len(probs) + 1))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=frames, 
                    y=probs,
                    mode='lines+markers',
                    name='Fake Probability',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                             annotation_text="Decision Threshold")
                fig.update_layout(
                    title="Fake Probability per Frame",
                    xaxis_title="Frame Number",
                    yaxis_title="Fake Probability",
                    yaxis=dict(range=[0, 1]),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Upload a video and click 'Analyze Video' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**DeepSight AI** - Built with PyTorch, EfficientNet-B0, and Grad-CAM | "
        "Trained on FaceForensics++ dataset"
    )

if __name__ == "__main__":
    main()
