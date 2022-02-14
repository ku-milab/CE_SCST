import copy
import math
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class Model(nn.Module):
    def __init__(self, x):
        super(Model, self).__init__()
        self.FE = Feature_Encoder()
        with torch.no_grad(): feature_size = self.FE(x).shape[-1]

        self.cls_1st = Classifier(feature_size, 5)
        self.project_att = nn.Linear(5, feature_size)

        embedding_size = feature_size//2
        self.ctx = BiLSTLM(feature_size, embedding_size)    # IP = [B, L, F]
        with torch.no_grad(): feature_size2 = self.ctx(self.FE(x)).shape[-1]
        self.project_f = nn.Linear(feature_size, feature_size2)
        self.dropout = nn.Dropout()
        self.cls_st = Classifier(feature_size2, 2)
        self.cls = Classifier(feature_size2, 5)

    def stage_confusion_estimator(self, x):
        l_1 = self.cls_1st(x)
        w = nn.Sigmoid()(self.project_att(F.softmax(l_1.flatten(start_dim=2), dim=-1)))
        x_att = x * w
        return x_att, l_1

    def CE(self, x, x_att):
        h = self.ctx(x_att)
        l_2_t = self.cls_st(h)
        h = self.dropout(self.project_f(x) + h)
        l_2 = self.cls(h)
        return l_2, l_2_t

class Classifier(nn.Module):
    def __init__(self, in_f, out_f):
        '''
        Classify FE feature
        '''
        super(Classifier, self).__init__()
        self.linear_1 = nn.Linear(in_f, out_f)

    def forward(self, x):
        x = self.linear_1(x)
        return x

class Feature_Encoder(nn.Module):
    def __init__(self):
        super(Feature_Encoder, self).__init__()
        # Define Conv, SepConv
        conv = lambda in_f, out_f, kernel, s=1: nn.Sequential(nn.Conv1d(in_f, out_f, (kernel,), stride=s), nn.BatchNorm1d(out_f), nn.LeakyReLU())
        sepconv_same = lambda in_f, out_f, kernel: nn.Sequential(nn.Conv1d(in_f, out_f, (kernel,), padding=(int(kernel/2),), groups=in_f),
                                                            nn.Conv1d(out_f, out_f, (1,)), nn.BatchNorm1d(out_f), nn.LeakyReLU())

        self.conv_T_0 = conv(1, 4, 50, args.stride)
        self.sepconv_T_1 = sepconv_same(4, 16, 15)
        self.sepconv_T_2 = sepconv_same(16, 32, 9)
        self.sepconv_T_3 = sepconv_same(32, 64, 5)

        self.conv_S_0 = conv(1, 4, 200, args.stride)
        self.sepconv_S_1 = sepconv_same(4, 16, 11)
        self.sepconv_S_2 = sepconv_same(16, 32, 7)
        self.sepconv_S_3 = sepconv_same(32, 64, 3)

        self.gap = nn.AdaptiveAvgPool1d(1)

    def one_way(self, conv_0, sepconv_1, sepconv_2, sepconv_3, x):
        b, l, t = x.shape
        x = x.view(-1, t).unsqueeze(1)    # [B*L, 1, T]
        x = conv_0(x)
        x = sepconv_1(x)
        x1 = x
        x = sepconv_2(x)
        x2 = x
        x = sepconv_3(x)
        x3 = x

        x = self.gap(torch.cat([x1, x2, x3], 1))    # [B, F, 1]
        x = x.reshape(b, l, -1)
        return x

    def forward(self, x):
        x_T = self.one_way(self.conv_T_0, self.sepconv_T_1, self.sepconv_T_2, self.sepconv_T_3, x)
        x_S = self.one_way(self.conv_S_0, self.sepconv_S_1, self.sepconv_S_2, self.sepconv_S_3, x)
        x = torch.cat((x_T, x_S), dim=-1)    # [B, L, F]
        return x

class BiLSTLM(nn.Module):
    def __init__(self, f, h):
        '''
        Temporal Encoder [Qu et al., 2020]
        Transformer
        '''
        super(BiLSTLM, self).__init__()
        if args.lstm_layers == 1: self.ctx = nn.LSTM(f, h, num_layers=args.lstm_layers, bidirectional=True)
        else: self.ctx = nn.LSTM(f, h, num_layers=args.lstm_layers, dropout=0.5, bidirectional=True)
        self.dropout_1 = nn.Dropout()
        self.dropout_2 = nn.Dropout()

    def forward(self, x):    # [B, L, F]
        h, _ = self.ctx(x.transpose(0, 1))
        h = self.dropout_1(h.transpose(0, 1))
        return h


