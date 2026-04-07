import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_arid_dataset(list_path, dataset_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    class_counts = {}
    fps_list, duration_list = [], []
    brightness_list = []
    resolutions = set()
    
    # Track which classes we have saved a sample frame for
    sampled_classes = set()
    
    with open(list_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"Total videos in list: {len(lines)}")
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            label_name = parts[2].split('/')[0]
            full_path = os.path.join(dataset_root, parts[2])
            
            class_counts[label_name] = class_counts.get(label_name, 0) + 1
            
            if os.path.exists(full_path):
                cap = cv2.VideoCapture(full_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if fps > 0 and frames > 0:
                    fps_list.append(fps)
                    duration_list.append(frames / fps)
                    resolutions.add(f"{w}x{h}")
                    
                    # Calculate average brightness of the video (sample up to 5 frames)
                    frame_count = 0
                    video_brightness = 0
                    while frame_count < 5:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Convert to grayscale to evaluate brightness
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        video_brightness += np.mean(gray)
                        frame_count += 1
                        
                        # Save one sample frame for qualitative analysis per class
                        if label_name not in sampled_classes:
                            cv2.imwrite(os.path.join(output_dir, f"arid_sample_{label_name}.jpg"), frame)
                            sampled_classes.add(label_name)
                            
                    if frame_count > 0:
                        brightness_list.append(video_brightness / frame_count)
                        
                cap.release()
                    
    # Visualization 1: Class Distribution
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='coral')
    plt.title('ARID-mini Class Distribution')
    plt.xlabel('Action Category')
    plt.ylabel('Number of Videos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "arid_class_distribution.png"))
    plt.close()
    
    # Visualization 2: Duration Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(duration_list, bins=20, color='mediumpurple', edgecolor='black')
    plt.title('ARID-mini Video Duration Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "arid_duration_distribution.png"))
    plt.close()

    # Visualization 3: Brightness Distribution 
    plt.figure(figsize=(10, 5))
    plt.hist(brightness_list, bins=20, color='darkgray', edgecolor='black')
    plt.title('ARID-mini Average Brightness Distribution (0-255)')
    plt.xlabel('Average Pixel Intensity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "arid_brightness_distribution.png"))
    plt.close()

    print("\n=== ARID-mini Dataset Statistics ===")
    print(f"Total categories: {len(class_counts)}")
    print(f"Class distribution: {class_counts}")
    print(f"Video resolutions: {resolutions}")
    print(f"Average FPS: {np.mean(fps_list):.2f} FPS")
    print(f"Average duration: {np.mean(duration_list):.2f} sec")
    print(f"Min duration: {np.min(duration_list):.2f} sec, Max duration: {np.max(duration_list):.2f} sec")
    print(f"Average brightness: {np.mean(brightness_list):.2f} (0-255 scale)")

if __name__ == "__main__":
    train_list = "data/arid-mini/arid_train.txt"
    data_root = "data/arid-mini"
    output_dir = "eda_results_arid"
    print("Starting ARID-mini EDA...")
    analyze_arid_dataset(train_list, data_root, output_dir)
    print("Done! Results saved in", output_dir)