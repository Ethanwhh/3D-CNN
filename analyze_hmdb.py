import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset(list_path, dataset_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    class_counts = {}
    fps_list, frame_counts_list, duration_list = [], [], []
    resolutions = set()
    
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
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
                    cap.release()
                    
                    if fps > 0 and frames > 0:
                        fps_list.append(fps)
                        frame_counts_list.append(frames)
                        duration_list.append(frames / fps)
                        resolutions.add(f"{w}x{h}")
                        
    # Plot class distribution
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.title('HMDB51-mini Class Distribution (Train)')
    plt.xlabel('Action Category')
    plt.ylabel('Number of Videos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hmdb_class_distribution.png"))
    plt.close()
    
    # Plot duration distribution
    plt.figure(figsize=(10, 5))
    plt.hist(duration_list, bins=20, color='lightgreen', edgecolor='black')
    plt.title('HMDB51-mini Video Duration Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hmdb_duration_distribution.png"))
    plt.close()

    print("=== HMDB51-mini Dataset Statistics ===")
    print(f"Total categories: {len(class_counts)}")
    print(f"Class distribution: {class_counts}")
    print(f"Video resolutions: {resolutions}")
    print(f"Average FPS: {np.mean(fps_list):.2f} FPS")
    print(f"Average duration: {np.mean(duration_list):.2f} sec")
    print(f"Min duration: {np.min(duration_list):.2f} sec, Max duration: {np.max(duration_list):.2f} sec")

# Run analysis
analyze_dataset("data/hmdb51-mini/hmdb51_train.txt", "data/hmdb51-mini", "eda_results_hmdb")
