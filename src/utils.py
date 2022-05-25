import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector

        return attn_pool, alpha

class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
            #import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim

        return attn_pool, alpha

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss

class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


def emotion_shift(filename):
    all_conv = pd.read_json(filename)
    list_all_conv = []
    for i in range(all_conv.shape[0]):
        list_all_conv.append(pd.json_normalize(all_conv.iloc[i, :].dropna()))

    emotion_shift = []
    for conv in list_all_conv:
        conv["no_shift"] = conv.emotion == conv.emotion.shift(2)

        # making sure first utternaces are considered as not shift
        conv.loc[0:2, "no_shift"] = True
        # emotion_shift.append(conv.shape[0] - conv["no_shift"].sum() - 2)
        emotion_shift.append(conv.shape[0] - conv["no_shift"].sum())

    return list_all_conv, emotion_shift


def stat_shift(shift):
    print(
        f"average : {np.mean(shift):.2f} ,median : {np.median(shift):.2f}, std : {np.std(shift):.2f}, length : {np.size(shift)}")


def length_conv(list_conv):
    length_conv = []
    for conv in list_conv:
        length_conv.append(conv.shape[0])
    return length_conv


def show_stat(list, shift):
    print("emotion shift in a conversation ")
    stat_shift(shift)
    print("Conversation length")
    stat_shift(length_conv(list))
    plt.figure()
    plt.xlabel('Number of emotion shift in a conversation')
    plt.ylabel('Counts')
    plt.hist(shift)
    plt.show()
    print("\n")


def meld_emotion_shift(filename):
    all_conv = pd.read_json(filename)
    list_all_conv = []
    for i in range(all_conv.shape[0]):
        list_all_conv.append(pd.json_normalize(all_conv.iloc[i, :].dropna()))
    emotion_shift = []
    for conv in list_all_conv:
        conv["no_shift"] = conv.emotion == conv.groupby(["speaker"]).emotion.shift(1)
        emotion_shift.append(conv.shape[0] - conv["no_shift"].sum() - (1 * np.size(np.unique(conv.speaker))))
    return list_all_conv, emotion_shift


class IEMOCAPRobertaCometDataset(Dataset):

    def __init__(self, split):
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open('iemocap/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
            = pickle.load(open('iemocap/iemocap_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor(self.xIntent[vid]), \
               torch.FloatTensor(self.xAttr[vid]), \
               torch.FloatTensor(self.xNeed[vid]), \
               torch.FloatTensor(self.xWant[vid]), \
               torch.FloatTensor(self.xEffect[vid]), \
               torch.FloatTensor(self.xReact[vid]), \
               torch.FloatTensor(self.oWant[vid]), \
               torch.FloatTensor(self.oEffect[vid]), \
               torch.FloatTensor(self.oReact[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.speakers[vid]]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 14 else pad_sequence(dat[i], True) if i < 16 else dat[i].tolist() for i in
                dat]


class MELDRobertaCometDataset(Dataset):

    def __init__(self, split, classify='emotion'):
        '''
        label index mapping =
        '''
        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open('meld/meld_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
            = pickle.load(open('meld/meld_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor(self.xIntent[vid]), \
               torch.FloatTensor(self.xAttr[vid]), \
               torch.FloatTensor(self.xNeed[vid]), \
               torch.FloatTensor(self.xWant[vid]), \
               torch.FloatTensor(self.xEffect[vid]), \
               torch.FloatTensor(self.xReact[vid]), \
               torch.FloatTensor(self.oWant[vid]), \
               torch.FloatTensor(self.oEffect[vid]), \
               torch.FloatTensor(self.oReact[vid]), \
               torch.FloatTensor(self.speakers[vid]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 14 else pad_sequence(dat[i], True) if i < 16 else dat[i].tolist() for i in
                dat]


class DailyDialogueRobertaCometDataset(Dataset):

    def __init__(self, split):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open('dailydialog/dailydialog_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
            = pickle.load(open('dailydialog/dailydialog_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor(self.xIntent[vid]), \
               torch.FloatTensor(self.xAttr[vid]), \
               torch.FloatTensor(self.xNeed[vid]), \
               torch.FloatTensor(self.xWant[vid]), \
               torch.FloatTensor(self.xEffect[vid]), \
               torch.FloatTensor(self.xReact[vid]), \
               torch.FloatTensor(self.oWant[vid]), \
               torch.FloatTensor(self.oEffect[vid]), \
               torch.FloatTensor(self.oReact[vid]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.speakers[vid]]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 14 else pad_sequence(dat[i], True) if i < 16 else dat[i].tolist() for i in
                dat]

# test = "/content/drive/MyDrive/IEMOCAP6/IEMOCAP6_test.json"
# dev = "/content/drive/MyDrive/IEMOCAP6/IEMOCAP6_dev.json"
# train = "/content/drive/MyDrive/IEMOCAP6/IEMOCAP6_train.json"
#
# iemo_test, shift_test = emotion_shift(test)
# iemo_dev, shift_dev = emotion_shift(dev)
# iemo_train, shift_train = emotion_shift(train)
#
# show_stat(iemo_test, shift_test)
# show_stat(iemo_dev, shift_dev)
# show_stat(iemo_train, shift_train)
#
# print(np.sum(shift_test))
# print(np.sum(shift_dev))
# print(np.sum(shift_train))