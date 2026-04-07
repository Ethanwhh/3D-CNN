import os
import cv2

def find_bad_videos(list_path, dataset_root):
    print(f"=====================================")
    print(f"Checking list: {list_path}")
    if not os.path.exists(list_path):
        print(f"List file does not exist: {list_path}")
        return
        
    bad_count = 0
    with open(list_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            if len(parts) >= 3:
                 rel_path = parts[2]
            else:
                 rel_path = parts[-1]
                 
            full_path = os.path.join(dataset_root, rel_path)
            
            is_bad = False
            reason = ""
            if not os.path.exists(full_path):
                is_bad = True
                reason = "File does not exist"
            elif os.path.getsize(full_path) == 0:
                is_bad = True
                reason = "File size is 0 bytes"
            else:
                try:
                    cap = cv2.VideoCapture(full_path)
                    if not cap.isOpened():
                        is_bad = True
                        reason = "OpenCV cannot open the video"
                    else:
                        ok, _ = cap.read()
                        if not ok:
                            is_bad = True
                            reason = "Cannot read the first frame (possibly corrupted)"
                    cap.release()
                except Exception as e:
                    is_bad = True
                    reason = f"Read exception: {str(e)}"
                    
            if is_bad:
                bad_count += 1
                print(f"\n[Corrupted File {bad_count}]")
                print(f"List File: {list_path}")
                print(f"Line Number: {line_idx + 1}")
                print(f"Line Content: {line}")
                print(f"Local Path: {full_path}")
                print(f"Reason: {reason}")
                
    if bad_count == 0:
        print("-> Check complete: No corrupted videos found!")
    else:
        print(f"-> Check complete: Found {bad_count} corrupted videos.")

if __name__ == "__main__":
    # Configure the tasks to check (txt path, local root directory of the dataset)
    check_tasks = [
        # ("data/hmdb51-mini/hmdb51_train.txt", "data/hmdb51-mini"),
        # ("data/hmdb51-mini/hmdb51_test.txt", "data/hmdb51-mini"),
        ("data/arid-mini/arid_train.txt", "data/arid-mini"),
        ("data/arid-mini/arid_test.txt", "data/arid-mini")
    ]

    for txt_file, root_dir in check_tasks:
        find_bad_videos(txt_file, root_dir)