import sys, os
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import seq2seq

import utils
import cfgs

from DecodeSearch import beam_decode
import argparse


chars = cfgs.CHAR_LIST
nums_class = len(chars)+3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 实例化字符转换的类
converter = utils.ConvertBetweenStringAndLabel(chars)


def load_model(encoder_path, decoder_path, nums_class):
    input_size = 512
    e_h_size = 256
    d_h_size = 256
    embedding_size = 256
    dropout_p = 0.1
    max_length = 180
    nums_class = nums_class

    encoder = seq2seq.Encoder(strides=[(2, 2), (2, 2), (2, 1), (2, 1), (1, 1)], input_size=input_size,
                              hidden_size=e_h_size, embedding_size=embedding_size)
    decoder = seq2seq.Decoder(hidden_size=d_h_size, embedding_size=embedding_size,
                              class_num=nums_class, dropout_p=dropout_p, max_length=max_length)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.to(device)
    decoder.to(device)

    return encoder, decoder

def get_image(img_path):
    img = np.array(Image.open(img_path).convert('L'))
    img = utils.resize_padding(img)   
    return [img]

def predictPerPic(img_path, batchszie=1):
    """返回每张整图的预测值 -> list"""

    # 后面images会变，这里保存到一个中间变量里
    images =  get_image(img_path)
    pr_list,score_list = predict(images)
  
    out_list = mathDecode(pr_list, score_list)
    return out_list


def mathDecode(decoded_batch, decoded_scores):
 
    decoded, decoded_score = decoded_batch[0], decoded_scores[0] # 输入单张图
 
    prs = []
    for ax in decoded:
        pr = []
        for bx in ax:
            if bx.item() == 0 or bx.item() == 2:
                continue
            if bx == 1:
                break
            temp = converter.decode(bx.item())
            pr.append(temp)
        pr = ''.join(pr)
        prs.append(pr) 
    return prs, decoded_score


def convertToTensor(images):
    """将图片转为可送入pytorch模型的格式"""
    # b, h, w, c
    images = np.array(images, dtype=np.float) / 255.0
    images = torch.from_numpy(images)
    images = images.permute(0, 3, 1, 2)
    images = images.to(device, dtype=torch.float)
    return images


def predict(images):
    """返回每个batch的预测值 -> list"""
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        batchsize = len(images)
        images = convertToTensor(images)

        encoder_outputs = encoder(images)
        decoder_hidden = encoder_outputs[-1].unsqueeze(0)

        # 解码
        decoded_batch, decoded_scores = beam_decode(decoder, batchsize, \
                                                     decoder_hidden, encoder_outputs)
       

    return decoded_batch, decoded_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='attention_val')
    parser.add_argument('--encoder', type=str, default='./saves/0203/Encoder-0203-epoch17.pth')
    parser.add_argument('--decoder', type=str, default='./saves/0203/Decoder-0203-epoch17.pth')
    parser.add_argument('--batchsize', type=int, default=64)
    args = parser.parse_args()

    print(args)
    encoder, decoder = load_model(args.encoder, args.decoder, nums_class)

    path = './datasets/data_001/images/15551.jpg' 
    # val_path = './datasets/jie_test/labels.txt'
    outs, scores = predictPerPic(path)
    print(outs, scores)

