import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model import seq2seq

from LabelSmoothingLoss import LabelSmoothSoftmaxCEV1
import utils
import cfgs


# 加载参数
chars = cfgs.CHAR_LIST
nums_class = len(chars)+3
num_epochs = cfgs.NUM_EPOCHS
encoder_resume = cfgs.ENCODER_RESUME
decoder_resume = cfgs.DECODER_RESUME
train_txt = cfgs.TRAIN_TXT
test_txt = cfgs.TEST_TXT
DATE = cfgs.NAME
RESUME = cfgs.RESUME
ENCODELR = cfgs.ENCODE_LEARNING_RATE
DECODELR = cfgs.DECODE_LEARNING_RATE
beta1 = cfgs.BETA1


# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化字符转换的类
converter = utils.ConvertBetweenStringAndLabel(chars)


def build_model(input_size=512, e_h_size=256, d_h_size=256, \
                embedding_size=256, dropout_p=0.1, max_length=180, \
                nums_class=nums_class, resume=False, encoder_resume=encoder_resume,
                decoder_resume=decoder_resume, device=device):
    """input_size:feature map 's channel nums
        e_h_size: Encoder LSTM's hidden_size
        d_h_size: Decoder GRU's hidden_size
        dropout_p: embedding's dropout
        max_length: Encoder LSTM's time steps"""
    # if backbone == 'common':
    encoder = seq2seq.Encoder(strides=[(2, 2), (2, 2), (2, 1), (2, 1), (1, 1)], input_size=input_size,
                                hidden_size=e_h_size, embedding_size=embedding_size)
    decoder = seq2seq.Decoder(hidden_size=d_h_size, embedding_size=embedding_size,
                                class_num=nums_class, dropout_p=dropout_p, max_length=max_length)
    
    if resume:
        encoder.load_state_dict(torch.load(encoder_resume))
        decoder.load_state_dict(torch.load(decoder_resume))

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    return encoder, decoder

def train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, epoch):
    print('\nEpoch: %d  train_start ===================>' % epoch)
    encoder.train()
    decoder.train()
    epoch_allloss = 0.0
    for indx, data in enumerate(train_loader):
        image, label = data
        image = image.to(device, dtype=torch.float)
        # label = label.to(device)
        batch_size = image.size(0)

        encoder_outputs = encoder(image)
        decoder_hidden = encoder_outputs[-1].unsqueeze(0)

        label = converter.encode(label).to(device)

        decoder_input = label[utils.SOS_TOKEN].to(device)
   
        loss = 0.0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # print(label.shape)

        for di in range(1, label.shape[0]):
          
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)      
            loss += criterion(decoder_output, label[di])
            decoder_input = label[di]

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        step_loss = loss.item()/batch_size
        # steps_losses.append(steps_loss)
        epoch_allloss += step_loss

        print('Epoch: {}/{}=========>Batch {}/{} Loss: {:.5f}'.format(epoch, num_epochs, indx, len(train_loader),
                                                                     step_loss))
    epoch_avgloss = epoch_allloss / len(train_loader)
    print('Epoch: {}/{}===========>train_avgloss: {:.5f}'.format(epoch, num_epochs, epoch_avgloss))

def evaluate(test_loader, encoder, decoder, criterion, device, epoch):
    print('\nEpoch: %d  evaluate_start ===================>' % epoch)
    encoder.eval()
    decoder.eval()
    n_correct = 0
    epoch_loss = 0
    with torch.no_grad():
        for indx, data in enumerate(test_loader):
            image, text_label = data
            image = image.to(device, dtype=torch.float)
            # text_label = label.to(device)
            # batch_size = image.size(0)

            encoder_outputs = encoder(image)
            decoder_hidden = encoder_outputs[-1].unsqueeze(0)

            num_label = converter.encode(text_label).to(device)

            decoder_input = num_label[utils.SOS_TOKEN].to(device)
            # decoder_hidden = torch.zeros(1, batch_size, embedding_size).to(device)

            loss = 0.0
            decoder_txts = []
            for di in range(1, num_label.shape[0]):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, num_label[di])
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze(1)
                decoder_input = ni

                if ni == utils.EOS_TOKEN:
                    break
                elif ni == utils.KONG:
                    continue
                else:
                    decoder_txts.append(converter.decode(ni))
            pr = ''.join(decoder_txts)
            gt = text_label[0]
            if pr == gt:
                n_correct += 1
            epoch_loss += loss.item()
            if indx % 10 == 0:
                print('Epoch: {}/{}=======> Batch {}/{} Loss: {:.5f}'.format(epoch, num_epochs, indx, len(test_loader),
                                                                             loss.item()))
    epoch_avgloss = epoch_loss/len(test_loader)
    evaluate_accuray = n_correct/len(test_loader)

    print('Epoch: {}/{}=======> val_accuray: {:.4f}  val_avgloss: {:.5f}'.format(epoch, num_epochs, evaluate_accuray,
                                                                                 epoch_avgloss))

def main():
    try:
        os.makedirs('./saves/%s' % DATE)
    except:
        pass

    os.system('cp cfgs.py ./saves/%s' % DATE)

    train_set = utils.OralDataset(train_txt)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
 
    test_set = utils.OralDataset_val(test_txt)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)

    encoder, decoder = build_model(resume=RESUME)
    print(encoder)
    print(decoder)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=ENCODELR, betas=(beta1, 0.999))
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=DECODELR, betas=(beta1, 0.999))
  
    criterion = LabelSmoothSoftmaxCEV1(lb_smooth=0.1, reduction='mean', ignore_index=-100)

    for epoch in range(1, num_epochs+1):
        train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, epoch)

        evaluate(test_loader, encoder, decoder, criterion, device, epoch)

        encoder_save = 'Encoder-{}-epoch{}.pth'.format(DATE, epoch)
        decoder_save = 'Decoder-{}-epoch{}.pth'.format(DATE, epoch)

        torch.save(encoder.state_dict(), os.path.join('./saves/%s' % DATE, encoder_save))
        torch.save(decoder.state_dict(), os.path.join('./saves/%s' % DATE, decoder_save))

if __name__=='__main__':
    main()