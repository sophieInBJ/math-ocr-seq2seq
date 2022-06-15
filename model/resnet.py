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

    def __init__(self, inplane, outplane, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplane, outplane, stride)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(outplane, outplane)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, inplane, outplane, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.conv1 = conv1x1(inplane, outplane)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(outplane, outplane, stride)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.conv3 = conv3x3(outplane, outplane*self.expansion)
        self.bn3 = nn.BatchNorm2d(outplane*self.expansion)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, strides, compress='three'):
        super(ResNet, self).__init__()
        self.inplane = 64
        self.conv1 = conv3x3(1, self.inplane, stride=strides[0])
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.compress = compress

        self.layer1 = self._make_layer(block, 64, layers[0], strides[1])
        self.layer2 = self._make_layer(block, 128, layers[1], strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], strides[3])
        self.layer4 = self._make_layer(block, 512, layers[3], strides[4])
        self.conv_end3 = nn.Conv2d(512, 512, (2, 1), (1, 1), padding=0)
        # for resnet50
        # self.conv_end3 = nn.Conv2d(2048, 512, (2, 1), (1, 1), padding=0)
        self.conv_end2 = nn.Conv2d(512, 512, (2, 1), (2, 1), padding=0)
        self.conv_end1 = nn.Conv2d(512, 512, (4, 1), (1, 1), padding=0)

        # pytorch 源码的初始化方式
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, plane, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplane!=plane*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplane, plane*block.expansion, stride=stride),
                nn.BatchNorm2d(plane*block.expansion),
            )

        layers = []
        layers.append(block(self.inplane, plane, stride, downsample))
        self.inplane = plane*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplane, plane))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.compress == 'three':
            out = self.conv_end3(out)
        elif self.compress == 'two':
            out = self.conv_end2(out)
        elif self.compress == 'one':
            out = self.conv_end1(out)
        out_feature = self.relu(out)
        return out_feature

def ResNet34(strides, compress):
    model = ResNet(BasicBlock, [3, 4, 6, 3], strides, compress)
    return model


def ResNet50(strides, compress):
    model = ResNet(BottleNeck, [3, 4, 6, 3], strides, compress)
    return model



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# images = np.random.rand(1, 64, 240, 1)
# images = torch.from_numpy(images)
# images = images.permute(0, 3, 1, 2)
# images = images.to(device, dtype=torch.float)
# strides = [(2,1), (2,2), (2,1), (2,2), (1,1)]
# encoder = ResNet34(strides,'three')
# outs = encoder(images)
# print(outs.size())


# (batchsize, 16, 60, 512*4)
# strides = [(2,1), (2,2), (2,1), (2,2), (1,1)]
# 64,240->3,60

# resnet50
# strides=[(2, 2), (2, 2), (2, 1), (2, 1), (1, 1)]