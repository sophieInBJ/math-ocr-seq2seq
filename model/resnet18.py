"""support resnet34 resnet50 and for different feature's height"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision.models.resnet
import numpy as np 

def conv1x1(inplane, outplane, stride=1):
    return nn.Conv2d(inplane, outplane, stride=stride, kernel_size=1, bias=False)


def conv3x3(inplane, outplane, stride=1):
    return nn.Conv2d(inplane, outplane, stride=stride, kernel_size=3, bias=False, padding=1)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock, self).__init__()
         #这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, strides):
        super(ResNet, self).__init__()
        self.inplane = 64
        self.conv1 = conv3x3(1, self.inplane, stride=strides[0])
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        # self.compress = compress

        self.layer1 = self._make_layer(block, 64,  2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=1)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
       
        # pytorch 源码的初始化方式
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, channels, stride))
            self.inplane = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
       
        out = F.avg_pool2d(out, (2,1),stride=(2, 1)) 
        out_feature = self.relu(out)
        return out_feature

def ResNet18(strides):
    model = ResNet(BasicBlock, [3, 4, 6, 3], strides)
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = np.random.rand(1, 64, 240, 1)
    images = torch.from_numpy(images)
    images = images.permute(0, 3, 1, 2)
    images = images.to(device, dtype=torch.float)
    strides = [(2,1), (2,2), (2,1), (2,2), (1,1)]
    encoder = ResNet18(strides)
    outs = encoder(images)
    print(outs.size())


# (batchsize, 16, 60, 512*4)
# strides = [(2,1), (2,2), (2,1), (2,2), (1,1)]
# 64,240->3,60

# resnet50
# strides=[(2, 2), (2, 2), (2, 1), (2, 1), (1, 1)]