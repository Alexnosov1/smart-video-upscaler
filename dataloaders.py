import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class VideoDataset(Dataset):
    """
    Загружает последовательности кадров из видеодатасета для тренировки модели.
    """
    def __init__(self, video_dir, sequence_length, crop_size):
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.crop_size = crop_size
        self.video_files = [fr"{os.path.join(video_dir, f)}" for f in os.listdir(video_dir) if f.endswith('.mp4') or f.endswith('.avi')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.random_crop(frame)
            frame_tensor = ToTensor()(frame)
            frames.append(frame_tensor)

        cap.release()

        if len(frames) < self.sequence_length:
            raise ValueError(f"Video {video_path} is too short for the required sequence length.")

        return torch.stack(frames)  # Возвращаем тензор размера [sequence_length, C, H, W]

    def random_crop(self, frame):
        h, w, _ = frame.shape
        top = np.random.randint(0, h - self.crop_size)
        left = np.random.randint(0, w - self.crop_size)
        return frame[top:top + self.crop_size, left:left + self.crop_size, :]


class train_dali_loader:
    """
    Загружает данные в формате PyTorch DataLoader вместо DALI.
    """
    def __init__(self, batch_size, file_root, sequence_length, crop_size, epoch_size=-1, random_shuffle=True, temp_stride=-1):
        dataset = VideoDataset(video_dir=file_root, sequence_length=sequence_length, crop_size=crop_size)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=random_shuffle, num_workers=4, drop_last=True)

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return iter(self.data_loader)
