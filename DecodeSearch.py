import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
import utils
import baby_math
import sys
import cfgs
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
KONG = 2
MAX_LENGTH = 60

chars = cfgs.CHAR_LIST
nums_class = len(chars)+3
converter = utils.ConvertBetweenStringAndLabel(chars)



class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.score = self.logp / float(self.leng - 1 + 1e-6)

    def __lt__(self, other):
        return self.score > other.score


def beam_decode(decoder, batchsize, decoder_hiddens, encoder_outputs=None):
    '''
    :param batchsize
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 5
    topk = 5 # how many sentence do you want to generate
    decoded_batch = []
    decoded_scores = []

    # decoding goes sentence by sentence
    for idx in range(batchsize):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)

        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)
        '''pooling方法的改动'''
        # encoder_output = encoder_outputs[idx, :, :, :].unsqueeze(0)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([utils.SOS_TOKEN]).to(device)

        # Number of sentence to generate
        endnodes = []


        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put(node)
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            # qisize代表解码次数，对每个节点解码，qisize要加beam_width-1的数量
            # qisize是一个经验阈值，需要寻找一个准确且解码准确率高的阈值
            if qsize > 150:
                break

            # fetch the best node
            # 每次拿出来的一定是分数最高的
            n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((n.score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= topk:
                    break
                else:
                    continue

            # output维度为 [batch_size, vocab_size]
            # hidden维度为 [num_layers * num_directions, batch_size, hidden_size]
            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            # log_prov, indexes维度为 [batch_size, beam_width] = [1, beam_width]
            log_prob, indexes = torch.topk(decoder_output, beam_width, dim=1)

            # nextnodes是从当前概率最高的节点作为输入，解码出的概率最高前beam_width个节点，概率高的在前面
            nextnodes = []

            for new_k in range(beam_width):
                # decoded_t: [1,1],通过view(1,-1)将数字tensor变为维度为[1,1]的tensor
                # decoded_t = indexes[0][new_k].view(1, -1)
                decoded_t = indexes[0][new_k].unsqueeze(0)
                # log_p, int
                log_p = log_prob[0][new_k].item() # item()将tensor数字变为int

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                nextnodes.append(node)
            # put them into queu
            # 把那些节点入队
            for i in range(len(nextnodes)):
                nn = nextnodes[i]
                nodes.put(nn)
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest
        # paths, back trace them
        # 每个解码的序列都没有终止符
        if len(endnodes) == 0:
            cur_nodes_num = nodes.qsize()
            # 如果队列里没有topk个值，只输出队列里有的
            if topk >= cur_nodes_num:
                for _ in range(cur_nodes_num):
                    temp = nodes.get()
                    endnodes.append((temp.score, temp))
            else:
                for _ in range(topk):
                    temp = nodes.get()
                    endnodes.append((temp.score, temp))

        utterances = []
        scores = []
        # print('长度:', len(endnodes))
        # print('解码次数:', qsize)
        for score, n in sorted(endnodes, key=operator.itemgetter(0), reverse=True):
            utterance = []
            utterance.append(n.wordid)
            scores.append(score)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            # 反转，从头输出解码结果
            utterance = utterance[::-1]
            utterances.append(utterance)


        decoded_batch.append(utterances)
        decoded_scores.append(scores)

    return decoded_batch, decoded_scores

def greedy_decode(decoder, decoder_hidden, encoder_outputs, batch_size):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    seq_len = 180
    decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
    decoder_input = torch.LongTensor([utils.SOS_TOKEN for _ in range(batch_size)]).to(device)

    decoder_input = torch.LongTensor([utils.SOS_TOKEN]*(batch_size)).to(device)

    for t in range(MAX_LENGTH):
  
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.data.topk(1)  # get candidates
        topi = topi.view(-1)
        decoded_batch[:, t] = topi
        decoder_input = topi

    return decoded_batch


if __name__ == "__main__":
    # str_list = ['1=1=1','79=79=79','34>33>35','33>30<35']
    # str_list = ['1+1=1','70+9=79=79','34>33>35','33>30<35']
    # str_list = ['1+1=1','70+8=79=79','34>33>32','33>30<35']
    # out0, out, i = select_right2(str_list)
    # print(out)
    a = '2+3=(945)…(88)'
    b = before_check_ox(a)
    print(b)
