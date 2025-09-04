import os, glob, csv
from infer_video import infer  # re-use function above

# Update paths to match our actual dataset structure
REAL_DIR = r"ffpp_data\real_videos"
FAKE_DIR = r"ffpp_data\fake_videos"
OUT_CSV  = "video_scores.csv"

def run_dir(dir_path, true_label):
    """Process all videos in a directory"""
    rows = []
    video_files = sorted(glob.glob(os.path.join(dir_path, "*.mp4")))
    
    print(f"Processing {len(video_files)} videos from {dir_path}...")
    
    for i, vp in enumerate(video_files):
        print(f"  [{i+1}/{len(video_files)}] {os.path.basename(vp)}")
        r = infer(vp)
        r["true"] = true_label
        rows.append(r)
        
        # Print result for this video
        status = "✓" if r["true"] == r["prediction"] else "✗"
        print(f"    {status} True: {true_label}, Pred: {r['prediction']}, Conf: {r['fake_confidence']:.3f}, Frames: {r['frames_used']}")
    
    return rows

if __name__ == "__main__":
    print("=== Batch Video Inference ===")
    print("Loading model and starting evaluation...")
    
    all_rows = []
    
    # Check if directories exist
    if not os.path.exists(REAL_DIR):
        print(f"Warning: Real videos directory not found: {REAL_DIR}")
    else:
        print(f"\n--- Processing REAL videos ---")
        all_rows += run_dir(REAL_DIR, "REAL")
    
    if not os.path.exists(FAKE_DIR):
        print(f"Warning: Fake videos directory not found: {FAKE_DIR}")
    else:
        print(f"\n--- Processing FAKE videos ---")
        all_rows += run_dir(FAKE_DIR, "FAKE")

    if not all_rows:
        print("No videos found to process!")
        exit(1)

    # Save CSV
    print(f"\n--- Saving Results ---")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "true", "prediction", "fake_confidence", "frames_used"])
        w.writeheader()
        w.writerows(all_rows)

    # Calculate and display accuracy
    print(f"\n=== RESULTS ===")
    correct = sum(1 for r in all_rows if r["true"] == r["prediction"])
    total = len(all_rows)
    acc = correct / total if total > 0 else 0.0
    
    # Calculate per-class accuracy
    real_videos = [r for r in all_rows if r["true"] == "REAL"]
    fake_videos = [r for r in all_rows if r["true"] == "FAKE"]
    
    real_correct = sum(1 for r in real_videos if r["prediction"] == "REAL")
    fake_correct = sum(1 for r in fake_videos if r["prediction"] == "FAKE")
    
    real_acc = real_correct / len(real_videos) if real_videos else 0.0
    fake_acc = fake_correct / len(fake_videos) if fake_videos else 0.0
    
    print(f"Videos evaluated: {total}")
    print(f"Overall accuracy: {acc:.3f} ({correct}/{total})")
    print(f"Real video accuracy: {real_acc:.3f} ({real_correct}/{len(real_videos)})")
    print(f"Fake video accuracy: {fake_acc:.3f} ({fake_correct}/{len(fake_videos)})")
    print(f"Results saved to: {OUT_CSV}")
    
    # Display confusion matrix
    print(f"\n=== CONFUSION MATRIX ===")
    print(f"               Predicted")
    print(f"               REAL  FAKE")
    print(f"Actual  REAL   {real_correct:4d}  {len(real_videos)-real_correct:4d}")
    print(f"        FAKE   {len(fake_videos)-fake_correct:4d}  {fake_correct:4d}")
