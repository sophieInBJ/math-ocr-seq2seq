"""v1 is general 方法  for train.py"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet18

import sys
import numpy as np

# baseline
class AttnAndRnn(nn.Module):

    def __init__(self, hidden_size, embedding_size, class_num, dropout_p=0.1, max_length=60*3):
        super(AttnAndRnn, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.dropout_p = dropout_p
        # max_length是encoder的时间步长
        self.max_length = max_length

        self.embedding = nn.Embedding(self.class_num, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.general = nn.Linear(embedding_size, hidden_size)
        self.reback = nn.Linear(2*hidden_size, hidden_size)
        self.out = nn.Linear(self.hidden_size, self.class_num)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden, att_feature):
        # batchsize ->batchsize*embeddingsize，batch里每个label被扩展成embedding向量
        # hidden 是上一时刻隐藏层状态（1，batchsize，embeddingsize）

        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        embedded = embedded.unsqueeze(0)

        # 之前val准确率上不去的原因就在于这里，没给gru传hidden
        output, hidden = self.gru(embedded, hidden)

        # general attention的方法
        t, b, c = att_feature.size()
        att_feature = att_feature.reshape(t*b, c)
        att_feature = self.general(att_feature)
        _, lc = att_feature.size()
        att_feature = att_feature.reshape(t, b, lc)

        # att_feature (timesteps, batchsize, hidden_size)  hidden  (1, batchsize, hidden_size)
        att_feature = att_feature.permute(1, 0, 2)   # b, t, hs
        hidden_temp = hidden.permute(1, 2, 0)  # b, hs, 1
        sorce = torch.bmm(att_feature, hidden_temp)
        sorce = sorce.squeeze(2)
        weight = F.softmax(sorce, dim=1)
        weight = weight.unsqueeze(2).permute(0, 2, 1)
        attention = torch.bmm(weight, att_feature)
        attention = attention.permute(1, 0, 2)

        # attention.size()  1,bs,hs

        output = torch.cat((attention, output), 2)

        # 增加concat之后降维激活
        output = output.squeeze(0)
        output = self.reback(output)
        output = F.relu(output)

        output = F.log_softmax(self.out(output), dim=1)

        return output, hidden

class Encoder(nn.Module):

    def __init__(self, strides, input_size, hidden_size, embedding_size):
        super(Encoder, self).__init__()
        # self.cnn = resnet.ResNet50(strides, compress='three')
        # self.cnn = resnet.ResNet34(strides, compress='three')
        self.cnn = resnet18.ResNet18(strides)
        self.lstm1 = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True)
        self.liner = nn.Linear(hidden_size*2, embedding_size)
    def forward(self, input):
        # conv features
        conv = self.cnn(input)

        # 64 512 3 60
        b, c, h, w = conv.size()

        # 先转置，再铺平
        conv = conv.permute(0, 1, 3, 2)
        conv = conv.reshape(b, c, h*w)
        conv = conv.permute(2, 0, 1)  # [b, c, t] -> [t, b, c]
        output, _ = self.lstm1(conv)
        output, _ = self.lstm2(output)
        # T步长数，b Batchsize，h 隐藏元数
        T, b, h = output.size()
        temp = output.view(T * b, h)
        output = self.liner(temp)
        output = output.view(T, b, -1)
        return output


class Decoder(nn.Module):

    def __init__(self, hidden_size, embedding_size, class_num, dropout_p=0.1, max_length=120):
        super(Decoder, self).__init__()
        self.decoder = AttnAndRnn(hidden_size, embedding_size, class_num, dropout_p, max_length)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

