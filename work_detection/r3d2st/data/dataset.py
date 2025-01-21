import os
import json
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, data_split='train', clip_length=16, clip_skip=1, transform=None):
        self.root_dir = root_dir
        self.data_split = data_split  # 'train', 'val', 'test'
        self.clip_length = clip_length
        self.transform = transform

        self.frame_paths = []
        self.labels = []
        self.sensors = []

        # データスプリットに応じたパスを設定
        split_dir = os.path.join(self.root_dir, self.data_split)
        video_dirs = sorted(glob(os.path.join(split_dir, 'frames', '2018*')))

        SENSOR_MAX_VALUE = 98
        sensor_files = sorted(glob(os.path.join('../../pseudo_sensing/fudagami', self.data_split, '2018*')))

        for video_dir, sensor_file in zip(video_dirs, sensor_files):
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

            frame_paths = []
            labels = []
            for i in range(num_frames):
                if i % 15 == 7 or i % 15 == 0:
                    frame_path = video_frame_paths[i]
                    label = frame_label_dict.get(i, -1)
                    if label == -1:
                        continue
                    frame_paths.append(frame_path)
                    labels.append(label)
            
            sensors = np.load(sensor_file).astype(np.float32)
            mean = SENSOR_MAX_VALUE / 2
            sensors = (sensors[:, 60:80, 100:125] - mean) / mean
            if sensors.shape[0] > len(frame_paths):
                sensors = sensors[:len(frame_paths)]
            if sensors.shape[0] < len(frame_paths):
                frame_paths = frame_paths[:sensors.shape[0]]
                labels = labels[:sensors.shape[0]]
            self.frame_paths += frame_paths[::clip_skip]
            self.labels += labels[::clip_skip]
            self.sensors.append(sensors[::clip_skip])
        self.sensors = np.concatenate(self.sensors, axis=0)

    def __len__(self):
        return len(self.frame_paths) - self.clip_length + 1

    def __getitem__(self, idx):
        clip_frame_paths = self.frame_paths[idx:idx+self.clip_length]
        clip_labels = self.labels[idx:idx+self.clip_length]
        clip_sensors = self.sensors[idx:idx+self.clip_length]

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
        sensors = torch.tensor(clip_sensors)

        return clip, labels, frame_ids, sensors