import os, glob
import subprocess

def test_model_accuracy():
    """Test the model on multiple real and fake videos to check accuracy"""
    
    print("🧪 TESTING MODEL ACCURACY ON REAL VS FAKE VIDEOS")
    print("="*60)
    
    # Get available videos
    real_videos = glob.glob("ffpp_data/real_videos/*.mp4")[:5]  # Test first 5
    fake_videos = glob.glob("ffpp_data/fake_videos/*.mp4")[:5]  # Test first 5
    
    print(f"Testing {len(real_videos)} real videos and {len(fake_videos)} fake videos\n")
    
    # Test different inference methods
    inference_scripts = [
        ("Original (MTCNN)", "infer_video.py"),
        ("Final (Enhanced)", "infer_video_final.py")
    ]
    
    for method_name, script in inference_scripts:
        print(f"📊 TESTING WITH {method_name}")
        print("-" * 40)
        
        real_correct = 0
        fake_correct = 0
        real_total = 0
        fake_total = 0
        
        # Test real videos
        print("🎭 REAL VIDEOS:")
        for video_path in real_videos:
            video_name = os.path.basename(video_path)
            try:
                result = subprocess.run([
                    "c:/Users/visha/DeepSight_AI/Deepsight-AI/.venv/Scripts/python.exe",
                    script, video_path
                ], capture_output=True, text=True, timeout=60)
                
                output = result.stdout
                if "Prediction: REAL" in output:
                    real_correct += 1
                    status = "✅ CORRECT"
                elif "Prediction: FAKE" in output:
                    status = "❌ WRONG (predicted FAKE)"
                elif "Prediction: UNKNOWN" in output:
                    status = "⚠️  UNKNOWN"
                else:
                    status = "❓ ERROR"
                
                # Extract confidence if available
                confidence = "N/A"
                for line in output.split('\n'):
                    if "confidence:" in line.lower():
                        confidence = line.split(':')[-1].strip()
                        break
                
                print(f"  {video_name}: {status} (conf: {confidence})")
                real_total += 1
                
            except Exception as e:
                print(f"  {video_name}: ❌ ERROR - {str(e)}")
                real_total += 1
        
        # Test fake videos  
        print("\n🎬 FAKE VIDEOS:")
        for video_path in fake_videos:
            video_name = os.path.basename(video_path)
            try:
                result = subprocess.run([
                    "c:/Users/visha/DeepSight_AI/Deepsight-AI/.venv/Scripts/python.exe", 
                    script, video_path
                ], capture_output=True, text=True, timeout=60)
                
                output = result.stdout
                if "Prediction: FAKE" in output:
                    fake_correct += 1
                    status = "✅ CORRECT"
                elif "Prediction: REAL" in output:
                    status = "❌ WRONG (predicted REAL)"
                elif "Prediction: UNKNOWN" in output:
                    status = "⚠️  UNKNOWN"
                else:
                    status = "❓ ERROR"
                
                # Extract confidence if available
                confidence = "N/A"
                for line in output.split('\n'):
                    if "confidence:" in line.lower():
                        confidence = line.split(':')[-1].strip()
                        break
                
                print(f"  {video_name}: {status} (conf: {confidence})")
                fake_total += 1
                
            except Exception as e:
                print(f"  {video_name}: ❌ ERROR - {str(e)}")
                fake_total += 1
        
        # Calculate accuracy
        total_correct = real_correct + fake_correct
        total_videos = real_total + fake_total
        overall_accuracy = total_correct / total_videos if total_videos > 0 else 0
        real_accuracy = real_correct / real_total if real_total > 0 else 0
        fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
        
        print(f"\n📈 {method_name} RESULTS:")
        print(f"  Real Videos: {real_correct}/{real_total} correct ({real_accuracy:.1%})")
        print(f"  Fake Videos: {fake_correct}/{fake_total} correct ({fake_accuracy:.1%})")
        print(f"  Overall: {total_correct}/{total_videos} correct ({overall_accuracy:.1%})")
        print(f"  {'='*60}\n")

if __name__ == "__main__":
    test_model_accuracy()
