import os
import argparse

import tensorboardX as tbx
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import make_data_loader
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
from modeling.cnn_lstm import CNN_LSTM


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.output_dir = f"./outputs/{args.model}/{self.args.checkname}/"
        self.writer = tbx.SummaryWriter(os.path.join(self.output_dir, 'logs'))

        batch_size = args.batch_size
        args.train_mode = True
        self.train_loader, self.val_loader, self.test_loader, self.nclass, population = make_data_loader(args, pin_memory=True)

        self.best_pred = 0
        self.best_loss = np.inf

        self.cuda = args.cuda
        if self.cuda:
            print('use cuda.')
            device = torch.device("cuda")
        else:
            print('do not use cuda.')
            device = torch.device("cpu")
        self.device = device
        
        self.model = CNN_LSTM(12, model=args.model)
        self.model = self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.lr)
        
        if args.use_weighted_loss:
            loss_weights = population * 100 + 0.001
            loss_weights = loss_weights.mean() / loss_weights
            print('use weited loss', loss_weights)
            loss_weights = torch.tensor(loss_weights).float().to(device)
        else:
            loss_weights = None
            print('do not use weighted loss')
        self.criterion = SegmentationLosses(weight=loss_weights, cuda=self.cuda).build_loss(mode="ce")
        
        self.evaluator = Evaluator(12)
        self.scheduler = LR_Scheduler("poly", args.lr, args.epochs, len(self.train_loader))

    def training(self, epoch):
        best_pred = self.best_pred
        device = self.device

        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            feats, target = sample['feats'], sample['labels']
            if self.args.use_mask_stream:
                masks = sample['masks']
            if self.cuda:
                feats, target = feats.to(device), target.to(device)
                if self.args.use_mask_stream:
                    masks = masks.to(device)

            if self.args.use_mask_stream:
                inputs = {'feats': feats, 'masks': masks}
            else:
                inputs = {'feats': feats}

            self.scheduler(self.optimizer, i, epoch, best_pred)
            self.optimizer.zero_grad()

            output = self.model(inputs)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            pred = output.data.cpu().numpy()
            batch = pred.shape[0]
            reshape_pred = np.zeros((batch, self.args.window_size),dtype=int)
            for index in range(batch):
                result =  np.argmax(pred[index], axis = 1)
                reshape_pred[index] =result
            # Add batch sample into evaluator
            target = target.cpu().numpy()
            self.evaluator.add_batch(target, reshape_pred)
            tbar.set_description('Train loss: %.7f' % (train_loss / (i + 1)))

        self.writer.add_scalar("train/loss", train_loss /(i + 1) , epoch) 
        Acc = self.evaluator.Pixel_Accuracy()
        self.writer.add_scalar("train/acc", Acc, epoch)

    def validation(self, epoch):
        self.model.eval()
        device = self.device
        self.evaluator.reset()
        tbar = tqdm(self.val_loader)
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            feats, target = sample['feats'], sample['labels']
            if self.args.use_mask_stream:
                masks = sample['masks']
            else:
                masks = 0
            if self.cuda:
                feats, target = feats.to(device), target.to(device)
                if self.args.use_mask_stream:
                    masks = masks.to(device)

            if self.args.use_mask_stream:
                inputs = {'feats': feats, 'masks': masks}
            else:
                inputs = {'feats': feats}

            with torch.no_grad():
                output = self.model(inputs)

            loss = self.criterion(output, target)
            test_loss += loss.item()
            
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            # Batches are set one at a time to argmax because 3D arrays cannot be converted to 2D arrays with argmax.
            
            batch = pred.shape[0]
            reshape_pred = np.zeros((batch, self.args.window_size),dtype=int)
            for index in range(batch):
                reshape_pred[index] = np.argmax(pred[index], axis = 1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, reshape_pred)

            tbar.set_description('Test loss: %.7f' % (test_loss / (i + 1)))
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar("Validation/loss", test_loss/(i + 1), epoch)
        self.writer.add_scalar("Validation/acc", Acc, epoch)
        self.writer.add_scalar("Validation/Acc_class", Acc_class, epoch)
        self.writer.add_scalar("Validation/mIoU", mIoU, epoch)
        self.writer.add_scalar("Validation/fwIoU", FWIoU, epoch)

        if test_loss/(i+1) < self.best_loss:
            self.best_loss = test_loss/(i+1)
        
        checkname = os.path.join(self.output_dir, 'weights')
        if not os.path.exists(checkname):
            os.makedirs(checkname)
        checkname = os.path.join(checkname, f'weight_epoc{epoch}.pth')
        torch.save(self.model.state_dict(), checkname)

def main():
    parser = argparse.ArgumentParser(description="CNN-LSTM for factory dataset.")
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                training')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--dataset', type=str, default='fudagami',
                        choices=['fudagami'],
                        help='dataset name (default: fudagami)')
    parser.add_argument('--window-size', type=int, default=100,
                        metavar='N', help='Time window size.')
    parser.add_argument('--model', type=str, default='feat',
                        choices=['basemodel', 'feat-sensor'],
                        help='model name (default: feat)')
    parser.add_argument('--use-weighted-loss', action='store_true', default=
                        False, help='use weighted loss')
    parser.add_argument('--is-dev', action='store_true', default=
                        False, help='restrict dataset num')

    args = parser.parse_args()
    assert args.checkname is not None, "Please set checkname."
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.use_mask_stream = args.model == 'feat-sensor'
    print(args)

    trainer = Trainer(args)

    print("start learning.")

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)


if __name__ == "__main__":
   main()

