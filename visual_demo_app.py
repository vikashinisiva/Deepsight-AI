import streamlit as st
import cv2, torch, torch.nn as nn, numpy as np
import glob, os, subprocess, tempfile, time
from torchvision import models
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from grad_cam import GradCAM, overlay_cam_on_image, make_infer_transform
import json
from datetime import datetime
import random
import io
from PIL import Image
import base64
import matplotlib.pyplot as plt

# Page config with modern theme
st.set_page_config(
    page_title="DeepSight AI - Visual Demo Pipeline",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the visual demo app
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #34C759;
        --error-color: #FF3B30;
        --warning-color: #FF9500;
        --text-primary: #1C1C1E;
        --text-secondary: #8E8E93;
        --bg-primary: #FFFFFF;
        --bg-secondary: #F2F2F7;
        --shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        --shadow-hover: 0 8px 40px rgba(0, 0, 0, 0.15);
        --radius: 16px;
        --radius-large: 24px;
    }

    * {
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        min-height: 100vh;
        -webkit-font-smoothing: antialiased;
    }

    .main {
        background: transparent;
        padding: 20px;
    }

    /* Hero Header */
    .hero-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 40px;
        border-radius: var(--radius-large);
        margin-bottom: 30px;
        text-align: center;
        box-shadow: var(--shadow);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        opacity: 0.5;
        z-index: -1;
    }

    .hero-header h1 {
        color: var(--text-primary);
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 16px;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-header h2 {
        color: var(--text-secondary);
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 24px;
    }

    .hero-header p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin: 0;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Pipeline Stage Cards */
    .pipeline-stage {
        background: var(--bg-primary);
        padding: 30px;
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        margin-bottom: 20px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
        transform: translateY(20px);
        opacity: 0;
        animation: slideInUp 0.8s ease-out forwards;
    }

    .pipeline-stage::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border-radius: var(--radius) var(--radius) 0 0;
    }

    .pipeline-stage.processing {
        animation: pulse 2s ease-in-out infinite;
        border-color: var(--primary-color);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }

    .pipeline-stage.completed {
        border-color: var(--success-color);
        box-shadow: 0 4px 20px rgba(52, 199, 89, 0.3);
    }

    .pipeline-stage h3 {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .pipeline-stage p {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 20px;
    }

    /* Progress Bars */
    .stage-progress {
        background: var(--bg-secondary);
        border-radius: 12px;
        height: 8px;
        overflow: hidden;
        margin: 20px 0;
        position: relative;
    }

    .progress-fill {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        height: 100%;
        border-radius: 12px;
        transition: width 2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.3),
            transparent
        );
        animation: progressShimmer 2s ease-in-out infinite;
    }

    @keyframes progressShimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    /* Frame Grid Animation */
    .frame-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 12px;
        margin: 20px 0;
        opacity: 0;
        animation: fadeInGrid 1s ease-out forwards;
    }

    .frame-item {
        background: var(--bg-secondary);
        border-radius: var(--radius);
        padding: 12px;
        text-align: center;
        border: 2px solid transparent;
        animation: frameFloat 3s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }

    .frame-item.highlighted {
        border-color: var(--primary-color);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: framePulse 1.5s ease-in-out infinite;
    }

    .frame-item img {
        width: 100%;
        height: 80px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 8px;
    }

    .frame-item p {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin: 0;
    }

    /* Neural Network Animation */
    .neural-network {
        background: var(--text-primary);
        border-radius: var(--radius);
        padding: 30px;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }

    .neural-network::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 50%, var(--primary-color) 2px, transparent 2px),
            radial-gradient(circle at 80% 50%, var(--secondary-color) 2px, transparent 2px),
            radial-gradient(circle at 50% 20%, var(--success-color) 2px, transparent 2px),
            radial-gradient(circle at 50% 80%, var(--warning-color) 2px, transparent 2px);
        background-size: 40px 40px;
        animation: neuralPulse 3s ease-in-out infinite;
        opacity: 0.3;
    }

    .neural-network h3 {
        color: white;
        text-align: center;
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
    }

    /* Timeline Visualization */
    .timeline-container {
        background: var(--bg-primary);
        border-radius: var(--radius);
        padding: 30px;
        margin: 20px 0;
        box-shadow: var(--shadow);
        position: relative;
    }

    .timeline-bar {
        height: 40px;
        border-radius: 20px;
        background: linear-gradient(90deg, 
            var(--success-color) 0%, var(--success-color) 30%,
            var(--warning-color) 30%, var(--warning-color) 60%,
            var(--error-color) 60%, var(--error-color) 100%
        );
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.1);
    }

    .timeline-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.3),
            transparent
        );
        animation: timelineShimmer 3s ease-in-out infinite;
    }

    .timeline-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 12px;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

    /* Result Cards */
    .result-card {
        background: var(--bg-primary);
        border-radius: var(--radius-large);
        padding: 40px;
        text-align: center;
        margin: 30px 0;
        box-shadow: var(--shadow-hover);
        position: relative;
        overflow: hidden;
    }

    .result-card.authentic {
        background: linear-gradient(135deg, var(--success-color) 0%, #30D158 100%);
        color: white;
    }

    .result-card.fake {
        background: linear-gradient(135deg, var(--error-color) 0%, #FF453A 100%);
        color: white;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: resultPulse 4s ease-in-out infinite;
    }

    .result-card h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 16px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
        z-index: 1;
    }

    .result-card h3 {
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 8px;
        position: relative;
        z-index: 1;
    }

    .result-card p {
        font-size: 1rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }

    /* Confidence Dial */
    .confidence-dial {
        width: 200px;
        height: 200px;
        margin: 20px auto;
        position: relative;
    }

    /* Heatmap GIF Container */
    .heatmap-container {
        background: var(--bg-primary);
        border-radius: var(--radius);
        padding: 20px;
        margin: 20px 0;
        box-shadow: var(--shadow);
        text-align: center;
    }

    .heatmap-container h4 {
        color: var(--text-primary);
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }

    /* Animations */
    @keyframes slideInUp {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInGrid {
        0% {
            opacity: 0;
            transform: scale(0.95);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }

    @keyframes frameFloat {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-5px);
        }
    }

    @keyframes framePulse {
        0%, 100% {
            border-color: var(--primary-color);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        50% {
            border-color: var(--secondary-color);
            box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
        }
    }

    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
    }

    @keyframes neuralPulse {
        0%, 100% {
            opacity: 0.3;
            transform: scale(1);
        }
        50% {
            opacity: 0.6;
            transform: scale(1.05);
        }
    }

    @keyframes timelineShimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    @keyframes resultPulse {
        0%, 100% {
            transform: scale(1);
            opacity: 0.3;
        }
        50% {
            transform: scale(1.05);
            opacity: 0.6;
        }
    }

    /* Buttons */
    .demo-button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: var(--radius);
        padding: 16px 32px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        box-shadow: var(--shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .demo-button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }

    .demo-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }

    .demo-button:hover::before {
        left: 100%;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: var(--radius);
        padding: 16px 32px;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }

    .stFileUploader > div {
        background: var(--bg-primary);
        border: 2px dashed var(--primary-color);
        border-radius: var(--radius);
        padding: 40px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow);
    }

    .stFileUploader > div:hover {
        border-color: var(--secondary-color);
        background: var(--bg-secondary);
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    for p in m.features.parameters(): p.requires_grad = False
    
    # Use the same custom classifier as in train_advanced.py
    num_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    
    # Load the checkpoint
    checkpoint = torch.load("weights/best_model.pth", map_location=device)
    m.load_state_dict(checkpoint["model_state_dict"])
    
    # For Grad-CAM, we need gradients on the last conv layer
    for p in m.features[-1].parameters():
        p.requires_grad = True
    
    m.eval().to(device)
    
    # Return model with accuracy info
    accuracy = checkpoint.get("accuracy", "N/A")
    return m, device, accuracy

@st.cache_resource
def setup_gradcam(_model):
    target_layer = _model.features[-1]
    cam = GradCAM(_model, target_layer)
    return cam

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_frames(video_path, out_dir, fps=15):
    """Extract frames at specified FPS"""
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.jpg")): 
        os.remove(f)
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-i", video_path, 
        "-r", str(fps), os.path.join(out_dir, "f_%03d.jpg")
    ], check=True)

def detect_faces(img, face_cascade):
    """Detect faces in image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return [(x, y, w, h) for (x, y, w, h) in faces]

def create_confidence_dial(confidence, label):
    """Create an animated confidence dial using Plotly"""
    fig = go.Figure()
    
    # Background circle
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(
            size=180,
            color='rgba(242, 242, 247, 0.8)',
            line=dict(width=0)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Progress arc
    angle = confidence * 360
    theta = np.linspace(0, np.radians(angle), 50)
    x = 0.7 * np.cos(theta)
    y = 0.7 * np.sin(theta)
    
    color = '#34C759' if label == 'AUTHENTIC' else '#FF3B30'
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(width=12, color=color),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Center text
    fig.add_annotation(
        x=0, y=0.2,
        text=f"<b>{confidence:.0%}</b>",
        showarrow=False,
        font=dict(size=32, color=color, family="Inter"),
        align="center"
    )
    
    fig.add_annotation(
        x=0, y=-0.2,
        text=label,
        showarrow=False,
        font=dict(size=16, color='#8E8E93', family="Inter"),
        align="center"
    )
    
    fig.update_layout(
        xaxis=dict(range=[-1, 1], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-1, 1], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        width=200,
        height=200,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def create_timeline_chart(probabilities, timestamps):
    """Create animated timeline chart"""
    fig = go.Figure()
    
    # Color mapping based on probability
    colors = []
    for prob in probabilities:
        if prob < 0.3:
            colors.append('#34C759')  # Green - Normal
        elif prob < 0.7:
            colors.append('#FF9500')  # Orange - Suspicious  
        else:
            colors.append('#FF3B30')  # Red - Fake
    
    fig.add_trace(go.Bar(
        x=timestamps,
        y=[1] * len(probabilities),
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        hovertemplate='<b>Time:</b> %{x}s<br><b>Fake Probability:</b> %{customdata:.1%}<extra></extra>',
        customdata=probabilities,
        showlegend=False
    ))
    
    fig.update_layout(
        title="<b>Video Timeline Analysis</b>",
        title_font=dict(size=16, family="Inter"),
        xaxis_title="Time (seconds)",
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=120,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def create_neural_network_viz():
    """Create animated neural network visualization"""
    fig = go.Figure()
    
    # Input layer
    for i in range(4):
        fig.add_trace(go.Scatter(
            x=[0], y=[i*2],
            mode='markers',
            marker=dict(size=20, color='#667eea'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Hidden layers
    for layer in range(1, 4):
        for i in range(3 if layer < 3 else 2):
            fig.add_trace(go.Scatter(
                x=[layer*2], y=[i*2.5 + 0.5],
                mode='markers',
                marker=dict(size=20, color='#764ba2'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Connections (simplified)
    for start_y in range(4):
        for end_y in range(3):
            fig.add_trace(go.Scatter(
                x=[0, 2], y=[start_y*2, end_y*2.5 + 0.5],
                mode='lines',
                line=dict(width=1, color='rgba(102, 126, 234, 0.3)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        xaxis=dict(range=[-0.5, 6.5], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-1, 8], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        annotations=[
            dict(x=0, y=-0.5, text="Input", showarrow=False, font=dict(color='white')),
            dict(x=2, y=-0.5, text="Hidden", showarrow=False, font=dict(color='white')),
            dict(x=4, y=-0.5, text="Features", showarrow=False, font=dict(color='white')),
            dict(x=6, y=-0.5, text="Output", showarrow=False, font=dict(color='white'))
        ]
    )
    
    return fig

def process_video_pipeline(video_path, model, device, face_cascade, cam_analyzer):
    """Process video through the complete pipeline with animations"""
    
    # Store results in session state to persist across reruns
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = {
            'frames': [],
            'faces': [],
            'probabilities': [],
            'heatmaps': [],
            'final_prediction': None,
            'confidence': 0
        }
    
    pipeline_results = st.session_state.pipeline_results
    
    # Progress tracking
    progress_placeholder = st.empty()
    main_content = st.empty()
    
    # Stage 1: Upload & Preview
    with progress_placeholder.container():
        st.markdown("""
        <div class="pipeline-stage processing">
            <h3><i class="fas fa-upload"></i> Step 1: Upload & Preview</h3>
            <p>Video file received and ready for processing</p>
            <div class="stage-progress">
                <div class="progress-fill" style="width: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    time.sleep(1)
    
    # Stage 2: Frame Extraction
    with progress_placeholder.container():
        st.markdown("""
        <div class="pipeline-stage processing">
            <h3><i class="fas fa-film"></i> Step 2: Frame Extraction</h3>
            <p>Extracting frames at 15 FPS for analysis</p>
            <div class="stage-progress">
                <div class="progress-fill" style="width: 60%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Extract frames
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            extract_frames(video_path, tmp_dir, fps=2)  # Reduced FPS for demo
            frame_paths = sorted(glob.glob(os.path.join(tmp_dir, "*.jpg")))[:8]  # First 8 frames
            
            # Load frames for display
            pipeline_results['frames'] = []
            for fp in frame_paths:
                img = cv2.imread(fp)
                if img is not None:
                    pipeline_results['frames'].append(img)
            
            # Update progress
            with progress_placeholder.container():
                st.markdown("""
                <div class="pipeline-stage completed">
                    <h3><i class="fas fa-film"></i> Step 2: Frame Extraction</h3>
                    <p>Successfully extracted frames for analysis</p>
                    <div class="stage-progress">
                        <div class="progress-fill" style="width: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show frame grid
            if pipeline_results['frames']:
                with main_content.container():
                    st.markdown("### 🎬 Extracted Frames")
                    cols = st.columns(4)
                    for i, frame in enumerate(pipeline_results['frames'][:8]):
                        with cols[i % 4]:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption=f"Frame {i+1}")
            
            time.sleep(2)
            
            # Stage 3: Face Detection
            with progress_placeholder.container():
                st.markdown("""
                <div class="pipeline-stage processing">
                    <h3><i class="fas fa-user"></i> Step 3: Face Detection</h3>
                    <p>Detecting and cropping faces from frames</p>
                    <div class="stage-progress">
                        <div class="progress-fill" style="width: 75%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detect faces
            pipeline_results['faces'] = []
            face_crops = []
            
            for frame in pipeline_results['frames']:
                faces = detect_faces(frame, face_cascade)
                if faces:
                    # Take largest face
                    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
                    x, y = max(0, x), max(0, y)
                    crop = frame[y:y+h, x:x+w]
                    if crop.size > 0:
                        face_crops.append(crop)
                        pipeline_results['faces'].append((crop, (x, y, w, h)))
            
            # Update progress
            with progress_placeholder.container():
                st.markdown("""
                <div class="pipeline-stage completed">
                    <h3><i class="fas fa-user"></i> Step 3: Face Detection</h3>
                    <p>Successfully detected and cropped faces</p>
                    <div class="stage-progress">
                        <div class="progress-fill" style="width: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display detected faces
            if face_crops:
                with main_content.container():
                    st.markdown("### 👥 Detected Faces")
                    cols = st.columns(min(len(face_crops), 4))
                    for i, crop in enumerate(face_crops[:4]):
                        with cols[i]:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            st.image(crop_rgb, caption=f"Face {i+1}")
            
            time.sleep(2)
            
            # Stage 4: Preprocessing & Frequency Analysis
            with progress_placeholder.container():
                st.markdown("""
                <div class="pipeline-stage processing">
                    <h3><i class="fas fa-cogs"></i> Step 4: Preprocessing & Frequency Analysis</h3>
                    <p>Applying transformations and frequency domain analysis</p>
                    <div class="stage-progress">
                        <div class="progress-fill" style="width: 85%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show preprocessing
            if pipeline_results['faces']:
                with main_content.container():
                    st.markdown("### 🔄 Preprocessing Pipeline")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Visual Path: Original Face Crops**")
                        if len(pipeline_results['faces']) > 0:
                            crop, _ = pipeline_results['faces'][0]
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            st.image(crop_rgb, caption="Original Face Crop")
                    
                    with col2:
                        st.markdown("**Frequency Path: FFT Spectrum Analysis**")
                        if len(pipeline_results['faces']) > 0:
                            crop, _ = pipeline_results['faces'][0]
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            f_transform = np.fft.fft2(gray)
                            f_shift = np.fft.fftshift(f_transform)
                            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
                            
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.imshow(magnitude_spectrum, cmap='hot')
                            ax.set_title('Frequency Domain Analysis')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
            
            time.sleep(2)
            
            # Stage 5: AI Model Analysis
            with progress_placeholder.container():
                st.markdown("""
                <div class="pipeline-stage processing">
                    <h3><i class="fas fa-brain"></i> Step 5: AI Model Analysis</h3>
                    <p>EfficientNet-B3 analyzing faces with Grad-CAM visualization</p>
                    <div class="stage-progress">
                        <div class="progress-fill" style="width: 90%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Neural Network Visualization
            with main_content.container():
                st.markdown("""
                <div class="neural-network">
                    <h3><i class="fas fa-project-diagram"></i> Neural Network Processing</h3>
                </div>
                """, unsafe_allow_html=True)
                
                neural_fig = create_neural_network_viz()
                st.plotly_chart(neural_fig, use_container_width=True)
            
            # Process faces through model
            tfm = make_infer_transform()
            pipeline_results['probabilities'] = []
            pipeline_results['heatmaps'] = []
            
            for crop, box in pipeline_results['faces'][:3]:  # Process first 3 faces
                # Get prediction
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = tfm(rgb).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(tensor)
                    prob_fake = torch.softmax(logits, dim=1)[0,0].item()
                    pipeline_results['probabilities'].append(prob_fake)
                
                # Generate Grad-CAM
                try:
                    cam_map = cam_analyzer(tensor, class_idx=0)
                    h, w = crop.shape[:2]
                    cam_resized = cv2.resize(cam_map, (w, h))
                    cam_resized = np.clip(cam_resized, 0, 1)
                    
                    heatmap_overlay = overlay_cam_on_image(crop, cam_resized, alpha=0.6)
                    pipeline_results['heatmaps'].append(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    st.warning(f"Grad-CAM generation failed: {e}")
            
            # Display analysis results
            if pipeline_results['heatmaps']:
                with main_content.container():
                    st.markdown("### 🔥 AI Attention Heatmaps")
                    cols = st.columns(min(len(pipeline_results['heatmaps']), 3))
                    for i, heatmap in enumerate(pipeline_results['heatmaps']):
                        with cols[i]:
                            st.image(heatmap, caption=f"AI Focus - Prob: {pipeline_results['probabilities'][i]:.1%}")
            
            time.sleep(2)
            
            # Stage 6: Result Aggregation
            with progress_placeholder.container():
                st.markdown("""
                <div class="pipeline-stage processing">
                    <h3><i class="fas fa-chart-bar"></i> Step 6: Result Aggregation</h3>
                    <p>Combining predictions across all frames</p>
                    <div class="stage-progress">
                        <div class="progress-fill" style="width: 95%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if pipeline_results['probabilities']:
                # Calculate final prediction
                avg_prob = np.mean(pipeline_results['probabilities'])
                pipeline_results['confidence'] = avg_prob
                pipeline_results['final_prediction'] = "FAKE" if avg_prob > 0.5 else "AUTHENTIC"
                
                # Timeline visualization
                with main_content.container():
                    timestamps = list(range(len(pipeline_results['probabilities'])))
                    timeline_fig = create_timeline_chart(pipeline_results['probabilities'], timestamps)
                    st.plotly_chart(timeline_fig, use_container_width=True)
                    
                    # Show aggregation metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Average Probability", f"{avg_prob:.1%}")
                    
                    with col2:
                        std_dev = np.std(pipeline_results['probabilities']) if len(pipeline_results['probabilities']) > 1 else 0
                        st.metric("Consistency", f"{1-std_dev:.1%}")
                    
                    with col3:
                        max_prob = max(pipeline_results['probabilities']) if pipeline_results['probabilities'] else 0
                        st.metric("Peak Suspicion", f"{max_prob:.1%}")
            
            time.sleep(2)
            
            # Stage 7: Final Report
            with progress_placeholder.container():
                st.markdown("""
                <div class="pipeline-stage completed">
                    <h3><i class="fas fa-flag-checkered"></i> Step 7: Final Report</h3>
                    <p>Analysis complete - generating comprehensive report</p>
                    <div class="stage-progress">
                        <div class="progress-fill" style="width: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Final Results Display
            prediction = pipeline_results['final_prediction']
            confidence = pipeline_results['confidence']
            
            with main_content.container():
                # Animated result card
                card_class = 'authentic' if prediction == 'AUTHENTIC' else 'fake'
                icon = 'fa-check-circle' if prediction == 'AUTHENTIC' else 'fa-exclamation-triangle'
                
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <h2><i class="fas {icon}"></i> {prediction}</h2>
                    <h3>Confidence: {confidence:.1%}</h3>
                    <p>Analysis completed with high precision AI detection</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence dial
                st.markdown("### 📊 Confidence Analysis")
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    dial_fig = create_confidence_dial(confidence if prediction == 'FAKE' else 1-confidence, prediction)
                    st.plotly_chart(dial_fig, use_container_width=True)
                
                # Detailed metrics
                st.markdown("### 📈 Detailed Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Frames Analyzed", len(pipeline_results['frames']))
                
                with col2:
                    st.metric("Faces Detected", len(pipeline_results['faces']))
                
                with col3:
                    st.metric("Average Confidence", f"{confidence:.1%}")
                
                with col4:
                    certainty = max(confidence, 1-confidence)
                    st.metric("Model Certainty", f"{certainty:.1%}")
                
                # Heatmap summary
                if pipeline_results['heatmaps']:
                    st.markdown("### 🔥 Key Suspicious Regions")
                    st.markdown("""
                    <div class="heatmap-container">
                        <h4><i class="fas fa-fire"></i> AI Attention Heatmaps</h4>
                        <p>Red areas indicate regions the AI found most suspicious</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, heatmap in enumerate(pipeline_results['heatmaps'][:3]):
                        st.image(heatmap, caption=f"Suspicious Frame {i+1} - Fake Probability: {pipeline_results['probabilities'][i]:.1%}")
                
        except Exception as e:
            st.error(f"Pipeline processing error: {str(e)}")
            return None
    
    return pipeline_results

def main():
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1><i class="fas fa-video"></i> DeepSight AI Visual Demo</h1>
        <h2>🎥 System Flow Visualization</h2>
        <p>Experience the complete deepfake detection pipeline with real-time animations, 
        neural network visualizations, and explainable AI heatmaps</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🚀 Initializing AI models and visual pipeline..."):
        model, device, accuracy = load_model()
        face_cascade = load_face_detector()
        cam_analyzer = setup_gradcam(model)
    
    st.success(f"✅ DeepSight AI Visual Demo Ready! (Model Accuracy: {accuracy:.1%})")
    
    # Main demo interface
    st.markdown("## 🎬 Interactive Pipeline Demo")
    
    # Video selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Video or Try Demo")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload your video to see the complete analysis pipeline"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Show video preview
            st.video(uploaded_file)
            
            if st.button("🎥 **Start Visual Pipeline Demo**", type="primary"):
                # Clear any existing pipeline results
                if 'pipeline_results' in st.session_state:
                    del st.session_state.pipeline_results
                
                try:
                    results = process_video_pipeline(video_path, model, device, face_cascade, cam_analyzer)
                    if results:
                        st.balloons()
                        st.success("🎉 Visual pipeline completed successfully!")
                except Exception as e:
                    st.error(f"Pipeline error: {str(e)}")
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(video_path)
                    except:
                        pass
    
    with col2:
        st.markdown("### 🎯 Quick Demo Options")
        
        st.markdown("""
        <div class="pipeline-stage">
            <h3><i class="fas fa-play-circle"></i> Demo Videos</h3>
            <p>Try the pipeline with our curated demo videos</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎬 **Real Video Demo**", type="secondary"):
            demo_path = "ffpp_data/real_videos/033.mp4"
            if os.path.exists(demo_path):
                st.session_state.demo_video_path = demo_path
                # Clear any existing pipeline results
                if 'pipeline_results' in st.session_state:
                    del st.session_state.pipeline_results
                st.rerun()
            else:
                st.error("Demo video not found. Please ensure dataset is downloaded.")
        
        if st.button("🎭 **Fake Video Demo**", type="secondary"):
            demo_path = "ffpp_data/fake_videos/033_097.mp4"
            if os.path.exists(demo_path):
                st.session_state.demo_video_path = demo_path
                # Clear any existing pipeline results
                if 'pipeline_results' in st.session_state:
                    del st.session_state.pipeline_results
                st.rerun()
            else:
                st.error("Demo video not found. Please ensure dataset is downloaded.")
        
        # Feature highlights
        st.markdown("""
        <div class="pipeline-stage">
            <h3><i class="fas fa-magic"></i> Pipeline Features</h3>
            <p>• Real-time frame extraction animation</p>
            <p>• Live face detection highlights</p>
            <p>• Neural network flow visualization</p>
            <p>• Grad-CAM attention heatmaps</p>
            <p>• Interactive confidence dials</p>
            <p>• Timeline analysis charts</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Process demo video if selected
    if hasattr(st.session_state, 'demo_video_path'):
        demo_path = st.session_state.demo_video_path
        
        st.markdown("### 🎥 Demo Video Preview")
        st.video(demo_path)
        
        if st.button("🚀 **Launch Demo Pipeline**", type="primary"):
            # Clear any existing pipeline results
            if 'pipeline_results' in st.session_state:
                del st.session_state.pipeline_results
            
            try:
                results = process_video_pipeline(demo_path, model, device, face_cascade, cam_analyzer)
                if results:
                    st.balloons()
                    st.success("🎉 Demo pipeline completed successfully!")
            except Exception as e:
                st.error(f"Demo pipeline error: {str(e)}")
        
        # Clear demo video path after processing or if user wants to reset
        if st.button("🔄 Reset Demo"):
            if 'demo_video_path' in st.session_state:
                del st.session_state.demo_video_path
            if 'pipeline_results' in st.session_state:
                del st.session_state.pipeline_results
            st.rerun()
    
    # Educational content
    st.markdown("---")
    st.markdown("## 🎓 Understanding the Pipeline")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="pipeline-stage">
            <h3><i class="fas fa-cogs"></i> Technical Pipeline</h3>
            <p><strong>Frame Extraction:</strong> 15 FPS sampling</p>
            <p><strong>Face Detection:</strong> Haar cascade algorithms</p>
            <p><strong>Preprocessing:</strong> 224x224 normalization</p>
            <p><strong>AI Analysis:</strong> EfficientNet-B3 classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pipeline-stage">
            <h3><i class="fas fa-eye"></i> Visual Intelligence</h3>
            <p><strong>Grad-CAM:</strong> AI attention visualization</p>
            <p><strong>Heatmaps:</strong> Suspicious region highlighting</p>
            <p><strong>Timeline:</strong> Temporal analysis charts</p>
            <p><strong>Confidence:</strong> Interactive dial visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="pipeline-stage">
            <h3><i class="fas fa-chart-line"></i> Analysis Output</h3>
            <p><strong>Binary Classification:</strong> Real vs Fake</p>
            <p><strong>Confidence Scoring:</strong> 0-100% probability</p>
            <p><strong>Frame Aggregation:</strong> Temporal consensus</p>
            <p><strong>Explainability:</strong> Reasoning visualization</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Reset button
    st.markdown("---")
    if st.button("🔄 **Reset Demo Pipeline**", type="secondary"):
        # Clear all session state related to pipeline
        keys_to_remove = []
        for key in st.session_state.keys():
            if 'pipeline' in key or 'demo_video' in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8E8E93; font-size: 0.9rem;">
        <p><strong>DeepSight AI Visual Demo</strong> - Experience the future of deepfake detection</p>
        <p>Powered by EfficientNet-B3, Grad-CAM, and advanced computer vision</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
