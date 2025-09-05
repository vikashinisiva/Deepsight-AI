# Video-Specific Model Fine-Tuning Guide

## Overview
This script allows you to fine-tune your deepfake detection model specifically on your video to improve recognition accuracy.

## Features
- 🎯 **Targeted Fine-tuning**: Adapts the model to recognize your specific video patterns
- 🔧 **Automatic Face Extraction**: Extracts faces from your video for training
- ⚖️ **Balanced Training**: Uses contrastive learning with existing data
- 🧪 **Built-in Testing**: Automatically tests the fine-tuned model on your video
- 💾 **Model Saving**: Saves fine-tuned models with metadata

## Usage

### 1. Prepare Your Video
- Ensure your video file is accessible
- The video should contain clear face shots
- Supported formats: MP4, AVI, MOV, etc.

### 2. Run the Fine-tuning Script
```bash
python fine_tune_video.py
```

### 3. Input Required Information
- **Video Path**: Full path to your video file
- **Label**: 0 for REAL, 1 for FAKE

### 4. What Happens During Fine-tuning
1. **Face Extraction**: Extracts up to 100 face images from your video
2. **Dataset Creation**: Combines your faces with existing contrastive examples
3. **Model Fine-tuning**: Fine-tunes the last layers of the model
4. **Testing**: Tests the fine-tuned model on your original video

## Output
- Fine-tuned model saved as: `weights/fine_tuned_[label]_[video_name].pth`
- Real-time progress updates during training
- Final prediction and confidence score

## Tips for Best Results
- ✅ Use videos with clear, well-lit faces
- ✅ Provide the correct label (0=REAL, 1=FAKE)
- ✅ Videos with multiple angles/lighting work best
- ✅ Longer videos (30+ seconds) provide more training data

## Troubleshooting
- **"Only extracted X faces"**: Video may have poor quality or obscured faces
- **Low confidence after fine-tuning**: Try with a different video or check label accuracy
- **Import errors**: Ensure all dependencies are installed (see requirements.txt)

## Dependencies
- PyTorch
- OpenCV
- Albumentations
- tqdm
- NumPy

The script will automatically handle model loading and face detection setup.
