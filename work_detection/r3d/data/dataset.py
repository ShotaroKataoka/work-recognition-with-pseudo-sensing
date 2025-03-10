import os
import json
from glob import glob

import torch
from torch.utils.data import Dataset
from PIL import Image

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, data_split='train', clip_length=16, clip_skip=7, transform=None):
        self.root_dir = root_dir
        self.data_split = data_split  # 'train', 'val', 'test'
        self.clip_length = clip_length
        self.transform = transform

        self.frame_paths = []
        self.labels = []

        # データスプリットに応じたパスを設定
        split_dir = os.path.join(self.root_dir, self.data_split)
        video_dirs = sorted(glob(os.path.join(split_dir, 'frames', '2018*')))

        for video_dir in video_dirs:
            video_id = os.path.basename(video_dir)
            json_path = os.path.join(split_dir, 'metadata', f"{video_id}.json")
            with open(json_path, 'r') as f:
                label_data = json.load(f)
            frame_labels = label_data['frames']

            # フレーム番号とラベルの辞書を作成
            frame_label_dict = {frame_info['number']: int(frame_info['label']) - 1 for frame_info in frame_labels}

            # フレームパスの取得
            video_frame_paths = sorted(glob(os.path.join(video_dir, '*.jpg')))
            num_frames = len(video_frame_paths)

            for i in range(num_frames):
                frame_path = video_frame_paths[i]
                label = frame_label_dict.get(i, -1)
                if label == -1:
                    continue
                self.frame_paths.append(frame_path)
                self.labels.append(label)

        self.frame_paths = self.frame_paths[::clip_skip]
        self.labels = self.labels[::clip_skip]
        self.num_frames = len(self.frame_paths)

    def __len__(self):
        return self.num_frames - self.clip_length + 1

    def __getitem__(self, idx):
        clip_frame_paths = self.frame_paths[idx:idx+self.clip_length]
        clip_labels = self.labels[idx:idx+self.clip_length]

        clip = []
        frame_ids = []
        for frame_path in clip_frame_paths:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            clip.append(image)
            frame_ids.append(os.path.basename(frame_path))  # フレーム識別子を取得
        clip = torch.stack(clip, dim=0)
        clip = clip.permute(1, 0, 2, 3)
        labels = torch.tensor(clip_labels)

        return clip, labels, frame_ids