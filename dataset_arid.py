import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class VideoDatasetARID(Dataset):
    def __init__(self, list_file, data_root, clip_len=16, sample_size=112, is_train=True, apply_gamma=True, gamma=0.4):
        self.data_root = data_root
        self.clip_len = clip_len
        self.sample_size = sample_size
        self.is_train = is_train
        
        self.apply_gamma = apply_gamma
        self.gamma = gamma
        
        # Build Gamma Correction Lookup Table for fast execution
        invGamma = 1.0 / self.gamma
        self.gamma_table = np.array([((i / 255.0) ** invGamma) * 255
                                     for i in np.arange(0, 256)]).astype("uint8")
        
        self.video_list = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                     label = int(parts[1])
                     rel_path = parts[2]
                     full_path = os.path.join(data_root, rel_path)
                     self.video_list.append((full_path, label))
                     
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path, label = self.video_list[idx]
        frames = self._load_video(video_path)
        
        # Spatial Augmentation
        if self.is_train:
            frames = self._random_crop(frames, self.sample_size)
            # Add random horizontal flip for robustness
            if np.random.rand() > 0.5:
                frames = frames[:, :, ::-1, :]
        else:
            frames = self._center_crop(frames, self.sample_size)
            
        # Normalize and to Tensor (C, T, H, W)
        frames = frames / 255.0
        frames = frames.transpose((3, 0, 1, 2))
        
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1, 1))
        frames = (frames - mean) / std
        
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply Gamma Correction to enhance brightness of dark pixels
            if self.apply_gamma:
                frame = cv2.LUT(frame, self.gamma_table)
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            return np.zeros((self.clip_len, self.sample_size, self.sample_size, 3), dtype=np.float32)
            
        indices = np.linspace(0, len(frames) - 1, self.clip_len).astype(int)
        sampled_frames = [cv2.resize(frames[i], (self.sample_size + 16, self.sample_size + 16)) for i in indices]
        
        return np.array(sampled_frames)

    def _random_crop(self, frames, size):
        h, w = frames.shape[1], frames.shape[2]
        if w == size and h == size: return frames
        x1 = np.random.randint(0, w - size)
        y1 = np.random.randint(0, h - size)
        return frames[:, y1:y1+size, x1:x1+size, :]
        
    def _center_crop(self, frames, size):
        h, w = frames.shape[1], frames.shape[2]
        if w == size and h == size: return frames
        x1 = int(round((w - size) / 2.))
        y1 = int(round((h - size) / 2.))
        return frames[:, y1:y1+size, x1:x1+size, :]