import os
from glob import glob
import json

from tqdm import tqdm
import numpy as np
import scipy
from sklearn.utils import shuffle
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from mypath import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Fudagami():
    def __init__(self, base_dir=Path.db_root_dir('fudagami'), split='train', model='feat', is_dev=False):
        self.sensor_class_num = 5
        labels_ = sorted(glob(os.path.join(base_dir, split, "labels", "*.json")))
        labels = []
        for label in labels_:
            t = int(label.split('/')[-1].split('.')[0].split('_')[1])
            if t > 73000 and t < 170000:
                labels += [label]

        if is_dev:
            labels = labels[:2]

        self.model = model

        self.metadatas = []
        skip = 2
        for label in labels:
            with open(label, 'r') as f:
                metadatas = json.load(f)
                self.metadatas += metadatas[::skip]
        self.labels = np.array([metadata['label'] for metadata in self.metadatas], dtype=np.float16) - 1
        self.population = np.bincount(self.labels.astype(np.int64)) / self.labels.shape[0]

        self.feats = np.zeros((len(self.metadatas), 256, 30, 40), dtype=np.float16)
        if self.model=='feat-sensor':
            self.sensor = np.zeros((len(self.metadatas), 1, 20, 30), dtype=np.float16)

        i = 0
        for label in tqdm(labels):
            id = label.split('/')[-1].split('.')[0]
            feats = np.load(os.path.join(base_dir, split, "feats", f"{id}.npy"))[::skip].astype(np.float16)
            num = feats.shape[0]
            self.feats[i:i+num] = feats
            if self.model == 'feat-sensor':
                SENSOR_MAX_VALUE = 98
                sensors = np.load(os.path.join(base_dir, split, "sensor_masks", f"{id}.npy"))[::skip].astype(np.float16) / SENSOR_MAX_VALUE
                if sensors.shape[0] > num:
                    sensors = sensors[:num]
                self.sensor[i:i+num, 0] = sensors
            i += num

        self.labels = torch.from_numpy(self.labels).float()
        self.feats = torch.from_numpy(self.feats).float()

        if self.model == 'feat-sensor':
            self.sensor = torch.from_numpy(self.sensor).float()

    def __getitem__(self, index):
        sample = {}
        sample['label'] = self.labels[index]
        sample['feat'] = self.feats[index]
        if self.model == 'feat-sensor':
            sample['sensor'] = self.sensor[index]
        sample['metadata'] = self.metadatas[index]
        return sample

    def encode_sensor(self, sensor):
        encode = np.zeros(sensor.shape+(self.sensor_class_num,), dtype=np.uint8)
        for i in range(self.sensor_class_num):
            index = sensor == i
            encode[:, :, :, i] = index
        return encode.transpose((0, 3, 1, 2))

    def __len__(self):
        return len(self.labels)

class FudagamiDataset():
    def __init__(self, args, base_dir=Path.db_root_dir('fudagami'), split='train'):
        self.base_dir = base_dir
        self.split = split
        self.window = args.window_size
        self.model = args.model

        self.fudagami = Fudagami(base_dir, split, self.model, is_dev=args.is_dev)
        self.population = self.fudagami.population

    def __getitem__(self, index):
        if type(index) is int and index >= len(self):
            raise IndexError("list index out of range")
        data = self.fudagami[index:index + self.window]
        labels = data['label']
        feats = data['feat']
        if self.model == 'feat-sensor':
            sensors = data['sensor']
        metadatas = data['metadata']
        if len(labels) < self.window:
            padding = self.window - len(labels)

            labels_ = torch.zeros((self.window)).float()
            labels_[:len(labels)] = labels
            labels_[len(labels):] = labels[-1]
            labels = labels_

            feats_ = torch.zeros((self.window, *feats.size()[1:])).float()
            feats_[:len(feats)] = feats
            feats_[len(feats):] = feats[-1]
            feats = feats_

            if self.model == 'feat-sensor':
                sensors_ = torch.zeros((self.window, *sensors.size()[1:])).float()
                sensors_[:len(sensors)] = sensors
                sensors_[len(sensors):] = sensors[-1]
                sensors = sensors_

            metadatas = metadatas + [metadatas[-1]]*padding
        sample = {}
        sample['labels'] = labels
        sample['feats'] = feats
        if self.model == 'feat-sensor':
            sample['sensors'] = sensors
        sample['metadatas'] = metadatas

        return sample
    

    def __len__(self):
        return len(self.fudagami)

