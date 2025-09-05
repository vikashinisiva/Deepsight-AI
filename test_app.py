# Test app functionality
try:
    import streamlit as st
    import cv2, torch, torch.nn as nn, numpy as np
    import glob, os, subprocess, tempfile
    from torchvision import transforms, models
    from PIL import Image
    import plotly.graph_objects as go
    import plotly.express as px
    from grad_cam import GradCAM, overlay_cam_on_image, make_infer_transform
    import time
    print('✅ All imports successful')

    # Test model loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  Device: {device}')

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
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

    checkpoint = torch.load('weights/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print('✅ Model loaded successfully')
    print(f'Model accuracy: {checkpoint["accuracy"]:.2f}%')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
