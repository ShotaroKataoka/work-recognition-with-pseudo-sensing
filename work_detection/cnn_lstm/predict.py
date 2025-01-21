import os
import argparse

import tensorboardX as tbx
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloaders import make_data_loader
from utils.saver import Saver
from utils.metrics import Evaluator
from modeling.cnn_lstm import CNN_LSTM
from dataloaders import fudagami


class Predictor(object):
    def __init__(self, args):
        self.args = args

        self.output_dir = f"./outputs/{args.model}/{self.args.checkname}/"

        batch_size = args.batch_size
        args.train_mode = False

        self.cuda = args.cuda
        if self.cuda:
            print('use cuda.')
            device = torch.device("cuda")
        else:
            print('do not use cuda.')
            device = torch.device("cpu")
        self.device = device
        
        if 'b16' in args.checkname:
            c_stream2_large = False
        elif 'b8' in args.checkname:
            c_stream2_large = True
        else:
            raise RuntimeError("=> batch size not found in '{}'" .format(args.checkname))
        self.model = CNN_LSTM(12, model=args.model, c_stream2_large=c_stream2_large)
        self.model = self.model.to(device)

        self.evaluator = Evaluator(12)

        checkpoint = os.path.join(self.output_dir, 'weights', f'weight_epoc{args.checkpoint_epoch}.pth')
        if not os.path.isfile(checkpoint):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(checkpoint))
        checkpoint = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(checkpoint)

    def pred(self, split):
        self.model.eval()
        device = self.device
        self.evaluator.reset()
        tbar = tqdm(DataLoader(fudagami.FudagamiDataset(self.args, split=split), 
                                 batch_size=self.args.batch_size, shuffle=False, pin_memory=True))
        outputs = {}
        for i, sample in enumerate(tbar):
            feats, target = sample['feats'], sample['labels']
            if self.args.use_mask_stream:
                masks = sample['masks']
            else:
                masks = 0
            if self.cuda:
                feats = feats.to(device)
                if self.args.use_mask_stream:
                    masks = masks.to(device)

            if self.args.use_mask_stream:
                inputs = {'feats': feats, 'masks': masks}
            else:
                inputs = {'feats': feats}

            with torch.no_grad():
                output = self.model(inputs)

            pred = output.data.cpu().numpy()
            target = target.data.numpy().astype(np.int16)
            
            batch = pred.shape[0]
            reshape_pred = np.zeros((batch, self.args.window_size),dtype=int)
            for index in range(batch):
                reshape_pred[index] = np.argmax(pred[index], axis = 1)
            labels = np.array([m['label'] for m in sample['metadatas']]).transpose(1, 0).astype(np.int16) - 1
            times = np.array([m['time'] for m in sample['metadatas']]).transpose(1, 0)

            reshape_pred = [str(i) for i in reshape_pred.reshape(-1)]
            labels = [str(i) for i in labels.reshape(-1)]
            times = list(times.reshape(-1))
            try:
                outputs["times"]  += times
                outputs["pred"]   += reshape_pred
                outputs["labels"] += labels
            except:
                outputs["times"]  = times
                outputs["pred"]   = reshape_pred
                outputs["labels"] = labels

            ## Add batch sample into evaluator
            #self.evaluator.add_batch(target, reshape_pred)

        ## Fast test during the training
        #Acc = self.evaluator.Pixel_Accuracy()
        #Acc_class = self.evaluator.Pixel_Accuracy_Class()
        #mIoU = self.evaluator.Mean_Intersection_over_Union()
        #FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        times = outputs["times"]
        preds = outputs["pred"]
        labels = outputs["labels"]
        outputs = ["times, preds, labels\n"]
        outputs += [f"{time}, {pred}, {label}\n" for time, pred, label in zip(times, preds, labels)]
        pred_dir = os.path.join(self.output_dir, 'outputs', f'ep{self.args.checkpoint_epoch}')
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        with open(os.path.join(pred_dir, f'{split}.csv'), 'w') as f:
            f.writelines(outputs)

def main():
    parser = argparse.ArgumentParser(description="PyTorch CNN-LSTM Predicting")
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                training')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--dataset', type=str, default='fudagami',
                        choices=['fudagami'],
                        help='dataset name (default: fudagami)')
    parser.add_argument('--window-size', type=int, default=100,
                        metavar='N', help='Time window size.')
    parser.add_argument('--model', type=str, default='feat',
                        choices=['feat', 'feat-mask', 'feat-mask-sensor', 'feat-sensor', 'feat-sensor-small'],
                        help='model name (default: feat)')
    parser.add_argument('--is-dev', action='store_true', default=
                        False, help='restrict dataset num')
    parser.add_argument('--checkpoint-epoch', type=int, default=None,
                        metavar='N', help='The epochs of checkpoint loaded.')


    args = parser.parse_args()
    assert args.checkname is not None, "Please set checkname."
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.use_mask_stream = args.model in ['feat-mask', 'feat-mask-sensor', 'feat-sensor', 'feat-sensor-small']
    print(args)

    predictor = Predictor(args)

    print("start predicting.")

    predictor.pred('train')
    predictor.pred('val')
    predictor.pred('test')


if __name__ == "__main__":
   main()

