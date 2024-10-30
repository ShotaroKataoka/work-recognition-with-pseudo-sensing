import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CnnModule(nn.Module):
    def __init__(self, in_c=256, out_c=256, is_mask_stream=False, model=''):
        super(CnnModule, self).__init__()
        self.conv = []
        self.norm = []
        self.relu = nn.ReLU(inplace=True)
        if is_mask_stream == True:
            if model == 'feat-sensor-small':
                self.conv += [nn.Conv2d(in_c, 32, 3, stride=2)]
                self.conv += [nn.Conv2d(32, 32, 3, stride=1)]
                self.conv += [nn.Conv2d(32, 32, 3, stride=1)]
                self.conv += [nn.Conv2d(32, out_c, 3, stride=1)]
                self.norm += [nn.BatchNorm2d(32)]
                self.norm += [nn.BatchNorm2d(32)]
                self.norm += [nn.BatchNorm2d(32)]
                self.norm += [nn.BatchNorm2d(out_c)]
            else:
                self.conv += [nn.Conv2d(in_c, 32, 3, stride=2)]
                self.conv += [nn.Conv2d(32, 32, 3, stride=2)]
                self.conv += [nn.Conv2d(32, 64, 3, stride=2)]
                self.conv += [nn.Conv2d(64, out_c, 3, stride=2)]
                self.norm += [nn.BatchNorm2d(32)]
                self.norm += [nn.BatchNorm2d(32)]
                self.norm += [nn.BatchNorm2d(64)]
                self.norm += [nn.BatchNorm2d(out_c)]
        else:
            self.conv += [nn.Conv2d(in_c, 256, 3, stride=2)]
            self.conv += [nn.Conv2d(256, out_c, 3, stride=2)]
            self.conv += [nn.Conv2d(out_c, out_c, 3, stride=2)]
            self.norm += [nn.BatchNorm2d(256)]
            self.norm += [nn.BatchNorm2d(out_c)]
            self.norm += [nn.BatchNorm2d(out_c)]

        self.conv = nn.ModuleList(self.conv)
        self.norm = nn.ModuleList(self.norm)

    def forward(self, x):
        """
        Parameters
        ----------
        x: todo
            torch.Size([1600, 256, 30, 40])
            4-D Tensor either of shape (b * sequence_length, c,h, w)

        Returns
        -------
        last_state_list, layer_output
        """
        
        for conv, norm in zip(self.conv, self.norm):
            x = self.relu(norm(conv(x)))

        # ↓ Global average pooling
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]) # in: torch.Size([1600, 512, 28, 38]) out: torch.Size([1600, 512, 1, 1])

        sequence_length , c ,h,w = x.shape 
        x = x.view(sequence_length, c)  # in: torch.Size([1600, 512, 1, 1]) out: torch.Size([1600, 512])
        # ↑ Global average pooling
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, nclass, dropout=0.2, model='feat', c_stream2_large=False, stream2_output_channel=None):
        super(CNN_LSTM, self).__init__()
        stream1_output_channel = 256
        if stream2_output_channel is None:
            if c_stream2_large:
                stream2_output_channel = 64
            else:
                stream2_output_channel = 32
        hidden_dim = 64

        self.model = model
        if model == 'feat':
            stream2_output_channel = 0
        elif model == 'feat-mask':
            self.decoder_cnn = CnnModule(in_c=5, out_c=stream2_output_channel, is_mask_stream=True)
        elif model == 'feat-mask-sensor':
            self.decoder_cnn = CnnModule(in_c=6, out_c=stream2_output_channel, is_mask_stream=True)
        elif model == 'feat-sensor':
            self.decoder_cnn = CnnModule(in_c=1, out_c=stream2_output_channel, is_mask_stream=True)
        elif model == 'feat-sensor-small':
            self.decoder_cnn = CnnModule(in_c=1, out_c=stream2_output_channel, is_mask_stream=True, model=model)

        self.feat_cnn = CnnModule(out_c=stream1_output_channel)
        lstm_input_channel = stream1_output_channel + stream2_output_channel
        self.lstm = nn.LSTM(lstm_input_channel, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, nclass)
        #self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, inputs):
        """
        Parameters
        ----------
        x: todo
            torch.Size([batch, 200 , 256, 30, 40])
            5-D Tensor either of shape (b, sequence_length, c,h, w)

        Returns
        -------
        last_state_list, layer_output
        """
        x = inputs['feats']
        b, sequence_length, c, h, w = x.shape
        x = x.view(b * sequence_length, c,h,w)
        cnn_result = self.feat_cnn(x)

        if self.model in ['feat-mask', 'feat-mask-sensor', 'feat-sensor', 'feat-sensor-small']:
            x = inputs['masks']
            b, sequence_length, c, h, w = x.shape
            x = x.view(b * sequence_length, c, h, w)
            cnn_result2 = self.decoder_cnn(x)
            cnn_result = torch.cat((cnn_result, cnn_result2), 1)

        cnn_result = cnn_result.view(b, sequence_length, -1)
        lstm_out, (h_n, c_n) = self.lstm(cnn_result)
        lstm_out = self.dropout(lstm_out)
        tag_score = self.hidden2tag(lstm_out)

        return tag_score

