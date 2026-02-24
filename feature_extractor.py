import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import sys
sys.path.append('pytorch-i3d')
from pytorch_i3d import InceptionI3d

import numpy as np
import cv2
import os
from glob import glob

class VideoDataset(Dataset):
    def __init__(self, video_dir, window_size=64, stride=16, resize=(224, 224)):
        self.video_paths = sorted(glob(os.path.join(video_dir, '*.mp4')))
        self.window_size = window_size
        self.stride = stride
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.resize)
            frames.append(frame)
        cap.release()

        frames = np.array(frames)
        
        windows = []
        for i in range(0, len(frames) - self.window_size + 1, self.stride):
            window = frames[i:i + self.window_size]
            window = np.array([self.transform(frame).numpy() for frame in window])
            windows.append(window)
        
        if not windows:
            # Handle videos shorter than window_size
            # Pad with zeros if necessary
            window = np.zeros((self.window_size, 3, self.resize[0], self.resize[1]), dtype=np.float32)
            if frames.shape[0] > 0:
                processed_frames = np.array([self.transform(frame).numpy() for frame in frames])
                window[:frames.shape[0]] = processed_frames
            windows.append(window)

        return torch.from_numpy(np.array(windows, dtype=np.float32)), video_path

def extract_features(video_dir, output_dir, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load I3D model
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt')) # Make sure to have the model weights
    i3d.to(device)
    i3d.eval()

    dataset = VideoDataset(video_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2) # batch_size=1 for simplicity with variable windows

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for i, (windows, video_path) in enumerate(dataloader):
            video_path = video_path[0]
            print(f"Processing video: {os.path.basename(video_path)}")
            
            windows = windows.squeeze(0).to(device) # (Num_Windows, T, C, H, W) -> (Num_Windows, C, T, H, W)
            windows = windows.permute(0, 2, 1, 3, 4)

            features = []
            for j in range(0, windows.size(0), batch_size):
                batch_windows = windows[j:j + batch_size]
                # The I3D model expects input of shape (B, C, T, H, W)
                batch_features = i3d.extract_features(batch_windows)
                # The output is (B, 1024, T', H', W') - average pool spatially to get (B, 1024, T')
                # Then transpose to match How2Sign format: (T, 1024)
                batch_features = batch_features.mean(dim=[3, 4])  # Average pool spatial dims -> (B, 1024, T')
                batch_features = batch_features.permute(0, 2, 1)  # -> (B, T', 1024)
                features.append(batch_features.cpu().numpy())

            if features:
                features = np.concatenate(features, axis=0)  # (num_windows, T', 1024)
                # Reshape to (total_T, 1024) to match How2Sign format
                features = features.reshape(-1, 1024)
                
                output_filename = os.path.splitext(os.path.basename(video_path))[0] + '.npy'
                output_path = os.path.join(output_dir, output_filename)
                np.save(output_path, features.astype(np.float32))
                print(f"Saved features to {output_path} with shape {features.shape}")

if __name__ == '__main__':
    # Create dummy video files for testing
    if not os.path.exists('test_videos'):
        os.makedirs('test_videos')
        for i in range(3):
            # Create a dummy video file
            width, height = 256, 256
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = f'test_videos/video_{i}.mp4'
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            for _ in range(100): # 100 frames
                frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            print(f"Created dummy video: {video_path}")

    # Create a models directory and notify user to place weights
    if not os.path.exists('models'):
        os.makedirs('models')
    print("\nIMPORTANT: Please download the I3D model weights (e.g., 'rgb_imagenet.pt') and place them in the 'models' directory.\n")


    video_directory = 'test_videos' # Change this to your video directory
    output_feature_dir = 'output_features'
    
    extract_features(video_directory, output_feature_dir)
