import os
import json
from glob import glob

import torch
from torch.utils.data import Dataset
from PIL import Image

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, clip_length=16, transform=None):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform

        self.clips = []  # (clip_frame_paths, label)のリスト

        video_dirs = sorted(glob(os.path.join(self.root_dir, 'frames', '2018*')))

        for video_dir in video_dirs:
            video_id = os.path.basename(video_dir)
            json_path = os.path.join(self.root_dir, 'metadata', f"{video_id}.json")
            with open(json_path, 'r') as f:
                label_data = json.load(f)
            frame_labels = label_data['frames']

            frame_label_dict = {frame_info['number']: int(frame_info['label']) - 1 for frame_info in frame_labels}

            frame_paths = sorted(glob(os.path.join(video_dir, '*.jpg')))
            num_frames = len(frame_paths)

            for i in range(0, num_frames - self.clip_length + 1):
                clip_frame_paths = frame_paths[i:i+self.clip_length]
                middle_frame_num = i + self.clip_length // 2
                label = frame_label_dict.get(middle_frame_num, -1)
                if label == -1:
                    continue
                self.clips.append((clip_frame_paths, label))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_frame_paths, label = self.clips[idx]
        clip = []
        for frame_path in clip_frame_paths:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            clip.append(image)
        clip = torch.stack(clip, dim=0)  # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
        return clip, label
