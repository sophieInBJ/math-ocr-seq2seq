from torch.utils.data import Dataset, DataLoader
import torch
import sys, os
import cv2
import collections
import numpy as np
from PIL import Image
import my_aug

import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


SOS_TOKEN = 0  # special token for start of sentence
EOS_TOKEN = 1  # special token for end of sentence
KONG = 2  #增加空格label用于占位对齐(补长到最大字符)


class ConvertBetweenStringAndLabel(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
        增加KONG占位符
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

        self.dict = {}
        self.dict['SOS_TOKEN'] = SOS_TOKEN
        self.dict['EOS_TOKEN'] = EOS_TOKEN
        self.dict['KONG'] = KONG
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 3
    def encode(self, text):
        """
        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor targets:（max_length +2）× batch_size
            eg：一个batch里有64个label，先求最长的label有多长。每个label编码后开头都是0，中间是label的内容，
            然后以1作为结束符，然后往后面补2使这个list长度到max_length+2
        """
        if isinstance(text, str):
            text = [self.dict[item] if item in self.dict else 2 for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]    #一个装batch_size个list的list
            max_length = max([len(x) for x in text])   #单条label最大长度
            nb = len(text)   #其实就是batch_size
            targets = torch.ones(nb, max_length + 2) * 2
            for i in range(nb):
                targets[i][0] = 0
                targets[i][1:len(text[i]) + 1] = text[i]
                targets[i][len(text[i]) + 1] = 1
            text = targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)

    def decode(self, t):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """

        texts = list(self.dict.keys())[list(self.dict.values()).index(t)]
        return texts
    def decode_numpy(self, nums):
        out = ''
        for n in nums:
            if n == 0 or n==2:
                continue
            if n == 1:
                return out
            out += self.alphabet[n-3] 
        return out
        
class OralDataset(torch.utils.data.Dataset):

    def __init__(self, text_line_file=None, transform=None, target_transform=None):
        self.text_line_file = text_line_file
        with open(text_line_file, 'r', encoding='utf-8') as fp:
            self.lines = fp.readlines()
        self.half = len(self.lines)
        self.lines = self.lines + self.lines
        self.nSamples = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        line_splits = self.lines[index].strip().split(' ')
        img_path = line_splits[0]
        try:
            if 'train' in self.text_line_file:
                # img = cv2.imread(img_path, 0)
                img = Image.open(img_path).convert('L')
            else:
                # img = cv2.imread(img_path, 0)
                img = Image.open(img_path).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if index == len(self) - 1:
            temp = self.lines[0].strip().split(' ')
            img_next_path = temp[0]
            # img_next = cv2.imread(img_next_path, 0)
            img_next = Image.open(img_next_path).convert('L')
        else:
            temp = self.lines[index + 1].strip().split(' ')
            img_next_path = temp[0]
            # img_next = cv2.imread(img_next_path, 0)
            img_next = Image.open(img_next_path).convert('L')

        if index >= self.half:
            img = my_aug.doEageAug(np.array(img))
            img = Image.fromarray(np.uint8(img))

        img_aug = my_aug.aug_random(img, img_next, 1)
        img_pad = my_aug.resize_pil_img(img_aug)

        # img = resize_padding(img)
        img = np.array(img_pad)/255.0

        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        # img = img.permute(2, 0, 1)

        if self.transform is not None:
            img = self.transform(img)

        label = line_splits[1]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

class OralDataset_val(torch.utils.data.Dataset):

    def __init__(self, text_line_file=None, transform=None, target_transform=None):
        self.text_line_file = text_line_file
        with open(text_line_file, 'r', encoding='utf-8') as fp:
            self.lines = fp.readlines()
            self.nSamples = len(self.lines)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        line_splits = self.lines[index].strip().split(' ')
        img_path = line_splits[0]
        try:
            if 'train' in self.text_line_file:
                # img = cv2.imread(img_path, 0)
                img = Image.open(img_path).convert('L')
            else:
                # img = cv2.imread(img_path, 0)
                img = Image.open(img_path).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        img_pad = my_aug.resize_pil_img(img)
        # img = resize_padding(img)
        img = np.array(img_pad)/255.0

        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        # img = img.permute(2, 0, 1)

        if self.transform is not None:
            img = self.transform(img)

        label = line_splits[1]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

def add_stripe(img_in):

    h, w, _ = img_in.shape
    num = h // 4
    for ind in range(num):
        if ind % 2 ==0:
            img_in[ind*4:ind*4+4,:,:] += 255
    return img_in 

def resize_padding(img, IMAGE_HEIGHT=64, IMAGE_WIDTH=240, strip=False):
    # cv2 imread 灰度图会把最后那一个通道去掉
    h, w = img.shape
    # new_img = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1)) * 127

    if strip:
        new_img = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,1))
        new_img = add_stripe(new_img)
    else:
        new_img = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1)) * 127

    if w * 1.0 / h > (IMAGE_WIDTH * 1.0 / IMAGE_HEIGHT):  # w长
        h_hat = int(IMAGE_WIDTH * h * 1.0 / w)
        img = cv2.resize(img, (IMAGE_WIDTH, h_hat), interpolation=cv2.INTER_AREA)
        r0 = int((IMAGE_HEIGHT - h_hat) / 2)
        r1 = r0 + h_hat
        img = np.expand_dims(img, axis=2)
        new_img[r0:r1, :, :] = img
    else:  # h高
        w_hat = int(IMAGE_HEIGHT * w * 1.0 / h)
        img = cv2.resize(img, (w_hat, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        c0 = int((IMAGE_WIDTH - w_hat) / 2)
        c1 = c0 + w_hat
        img = np.expand_dims(img, axis=2)
        new_img[:, c0:c1, :] = img
    return new_img