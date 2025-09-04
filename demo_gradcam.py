#!/usr/bin/env python3
"""
Demo script showing Grad-CAM analysis on multiple videos
"""
import os
import glob
from cam_on_video import run

def demo_gradcam():
    """Run Grad-CAM on sample videos"""
    print("🔍 DeepSight AI - Grad-CAM Demo")
    print("=" * 50)
    
    # Test videos
    real_videos = glob.glob("ffpp_data/real_videos/*.mp4")[:3]
    fake_videos = glob.glob("ffpp_data/fake_videos/*.mp4")[:3]
    
    all_videos = real_videos + fake_videos
    
    if not all_videos:
        print("❌ No videos found. Make sure you have videos in ffpp_data/")
        return
    
    print(f"📹 Found {len(all_videos)} videos to analyze")
    print()
    
    results = []
    
    for i, video_path in enumerate(all_videos, 1):
        video_name = os.path.basename(video_path)
        video_type = "REAL" if "real_videos" in video_path else "FAKE"
        
        print(f"[{i}/{len(all_videos)}] Analyzing: {video_name} ({video_type})")
        
        try:
            output_path = run(video_path)
            if output_path:
                results.append({
                    "video": video_name,
                    "type": video_type,
                    "gradcam": output_path
                })
                print(f"✅ Grad-CAM saved: {output_path}")
            else:
                print(f"❌ Failed to process {video_name}")
        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")
        
        print()
    
    print("📊 Summary")
    print("=" * 50)
    print(f"Successfully processed: {len(results)}/{len(all_videos)} videos")
    
    if results:
        print("\n🖼️ Generated Grad-CAM visualizations:")
        for result in results:
            print(f"  • {result['gradcam']} ({result['type']} video)")
    
    print("\n🔥 Grad-CAM Legend:")
    print("  • Red/Yellow areas: High activation (potential deepfake artifacts)")
    print("  • Blue/Green areas: Low activation")
    print("  • The heatmap shows where the AI 'looks' when detecting deepfakes")

if __name__ == "__main__":
    demo_gradcam()
