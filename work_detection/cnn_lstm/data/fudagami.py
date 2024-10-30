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
from dataloaders.padding_times import padding_times, aopa1st, padding_times_labels

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Fudagami():
    def __init__(self, base_dir=Path.db_root_dir('fudagami'), split='train', model='feat', is_dev=False):
        self.mask_class_num = 5
        labels_ = sorted(glob(os.path.join(base_dir, split, "labels", "*.json")))
        labels = []
        for label in labels_:
            t = int(label.split('/')[-1].split('.')[0].split('_')[1])
            if t > 73000 and t < 170000:
                labels += [label]

        if is_dev:
            #labels = labels[18:22]
            labels = labels[0:2]

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
        if self.model=='feat-mask':
            self.masks = np.zeros((len(self.metadatas), 5, 120, 160), dtype=np.float16)
        if self.model=='feat-mask-sensor':
            self.masks = np.zeros((len(self.metadatas), 6, 120, 160), dtype=np.float16)
        if self.model=='feat-sensor':
            self.masks = np.zeros((len(self.metadatas), 1, 120, 160), dtype=np.float16)
        if self.model=='feat-sensor-small':
            self.masks = np.zeros((len(self.metadatas), 1, 20, 25), dtype=np.float16)

        i = 0
        for label in tqdm(labels):
            id = label.split('/')[-1].split('.')[0]
            feats = np.load(os.path.join(base_dir, split, "feats", f"{id}.npy"))[::skip].astype(np.float16)
            num = feats.shape[0]
            self.feats[i:i+num] = feats
            if self.model in ['feat-mask', 'feat-mask-sensor']:
                masks = self.encode_mask(np.load(os.path.join(base_dir, split, "masks_npy", f'{id}.npy'))[::skip])
                self.masks[i:i+num, :5] = masks
            if self.model in ['feat-mask-sensor', 'feat-sensor', 'feat-sensor-small']:
                SENSOR_MAX_VALUE = 98
                sensors = np.load(os.path.join(base_dir, split, "sensor_masks", f"{id}.npy"))[::skip].astype(np.float16) / SENSOR_MAX_VALUE
                if sensors.shape[0] > num:
                    sensors = sensors[:num]
                if self.model == 'feat-mask-sensor':
                    self.masks[i:i+num, 5] = sensors
                elif self.model == 'feat-sensor':
                    self.masks[i:i+num, 0] = sensors
                elif self.model == 'feat-sensor-small':
                    SENSOR_MAX_VALUE = 0.5
                    self.masks[i:i+num, 0] = (sensors[:, 60:80, 100:125] - SENSOR_MAX_VALUE) / SENSOR_MAX_VALUE
            i += num

        self.indexdict = {}
        index = 0
        padding_times_list = [time.split(':') for time in padding_times]
        padding_times_list = [int(time[0])*3600 + int(time[1])*60 + int(time[2]) for time in padding_times_list]
        for i, meta in enumerate(self.metadatas):
            time = meta['time'].split(':')
            second = int(time[0])*3600 + int(time[1])*60 + int(time[2])
            if len(padding_times_list) > 0:
                while second > padding_times_list[0]:
                    self.indexdict[index] = {'index': i-1, 'time': padding_times[0], 'label': int(padding_times_labels[0])}
                    index += 1
                    padding_times_list.pop(0)
                    padding_times_labels.pop(0)
                    padding_times.pop(0)
                    if len(padding_times_list) == 0:
                        break
            self.indexdict[index] = {'index': i, 'time': meta['time'], 'label': int(meta['label'])}
            index += 1

        self.labels = torch.from_numpy(self.labels).float()
        self.feats = torch.from_numpy(self.feats).float()

        if self.model in ['feat-mask', 'feat-mask-sensor', 'feat-sensor', 'feat-sensor-small']:
            self.masks = torch.from_numpy(self.masks).float()

    def __getitem__(self, index):
        if isinstance(index, slice):
            num = len(self)
            times = [self.indexdict[i]['time'] for i in range(index.start, index.stop) if i < num]
            index = [self.indexdict[i]['index'] for i in range(index.start, index.stop) if i < num]
            #labels = [aopa1st[t] for i, t in enumerate(times) if t in aopa1st.keys() else self.labels[index[i]]]
            labels = [aopa1st[t] if t in aopa1st.keys() else self.labels[index[i]] for i, t in enumerate(times)]
        else:
            index = self.indexdict[index]
        sample = {}
        #sample['label'] = self.labels[index]
        sample['label'] = torch.from_numpy(np.array(labels)).float()
        sample['feat'] = self.feats[index]
        if self.model in ['feat-mask', 'feat-mask-sensor', 'feat-sensor', 'feat-sensor-small']:
            sample['mask'] = self.masks[index]
        sample['metadata'] = [self.metadatas[i] for i in index]
        for i, t in enumerate(times):
            try:
                label = str(int(labels[i].item())+1)
            except:
                label = str(labels[i]+1)
            sample['metadata'][i] = {
                'label': label,
                'label_name': sample['metadata'][i]['label_name'],
                'time': t
              }
        return sample

    def encode_mask(self, mask):
        encode = np.zeros(mask.shape+(self.mask_class_num,), dtype=np.uint8)
        for i in range(self.mask_class_num):
            index = mask == i
            encode[:, :, :, i] = index
        return encode.transpose((0, 3, 1, 2))

    def __len__(self):
        return len(list(self.indexdict.keys()))

class FudagamiDataset():
    def __init__(self, args, base_dir=Path.db_root_dir('fudagami'), split='train', use_all_window=False):
        self.base_dir = base_dir
        self.split = split
        self.window = args.window_size
        self.model = args.model
        self.use_all_window = use_all_window

        self.fudagami = Fudagami(base_dir, split, self.model, is_dev=args.is_dev)
        self.population = self.fudagami.population

    def __getitem__(self, index):
        if not self.use_all_window:
            index = index * self.window + np.random.randint(0, np.min((self.window, len(self.fudagami)-index*self.window)))
        if type(index) is int and index >= len(self.fudagami):
            raise IndexError("list index out of range")
        data = self.fudagami[index:index + self.window]
        labels = data['label']
        feats = data['feat']
        if self.model in ['feat-mask', 'feat-mask-sensor', 'feat-sensor', 'feat-sensor-small']:
            masks = data['mask']
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

            if self.model in ['feat-mask', 'feat-mask-sensor', 'feat-sensor', 'feat-sensor-small']:
                masks_ = torch.zeros((self.window, *masks.size()[1:])).float()
                masks_[:len(masks)] = masks
                masks_[len(masks):] = masks[-1]
                masks = masks_

            metadatas = metadatas + [metadatas[-1]]*padding
        sample = {}
        sample['labels'] = labels
        sample['feats'] = feats
        if self.model in ['feat-mask', 'feat-mask-sensor', 'feat-sensor', 'feat-sensor-small']:
            sample['masks'] = masks
        sample['metadatas'] = metadatas
        return sample

    def __len__(self):
        if self.use_all_window:
            return len(self.fudagami)
        else:
            return len(self.fudagami) // self.window

