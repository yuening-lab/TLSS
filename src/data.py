import sys
import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity

class TLData_Loader(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window # 20
        self.h = args.horizon # 1
        self.d = 0
        self.add_his_day = False
        self.rawdat = np.loadtxt(open("/content/drive/MyDrive/Colab Notebooks/Cola_GNN_TransferLearning/COVID19_50states_548.txt".format(args.dataset)), delimiter=',')
        print('data x shape', self.rawdat.shape)
        if args.sim_mat:
            self.load_sim_mat(args)
        if args.vader:
            self.load_vader(args)
        if args.st:
            self.load_st(args)
 
        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape # n_sample, n_group
        self.scale = np.ones(self.m)

        self._pre_train(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        self._split(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        print('x size of train/val/test sets',self.train[0].shape,self.val[0].shape,self.test[0].shape)
        print('x2 size of train/val/test sets',self.train[1].shape,self.val[1].shape,self.test[1].shape)
        print('x3 size of train/val/test sets',self.train[2].shape,self.val[2].shape,self.test[2].shape)

    def load_sim_mat(self, args):
        self.adj = torch.Tensor(np.loadtxt(open("/content/drive/MyDrive/Colab Notebooks/Cola_GNN_TransferLearning/50state-adj.txt".format(args.sim_mat)), delimiter=','))
        self.orig_adj = self.adj
        rowsum = 1. / torch.sqrt(self.adj.sum(dim=0))
        self.adj = rowsum[:, np.newaxis] * self.adj * rowsum[np.newaxis, :]
        self.adj = Variable(self.adj)
        if args.cuda:
            self.adj = self.adj.cuda()
            self.orig_adj = self.orig_adj.cuda()

    def load_vader(self, args):
        self.news = pd.read_csv (r"/content/drive/MyDrive/Colab Notebooks/Cola_GNN_Vader/VaderScoreMean_news_50states_v2.csv")
        self.news = self.news.iloc[:,[0,1,-1]]
        self.news_Bydate = self.news.sort_values(['date', 'state'])
        self.vader = self.create_sequence1(lookback = args.window*50, window = args.window)
        self.vader = torch.tensor(self.vader).float()
        self.orig_vader = self.vader
        #print("vader data shape:", self.vader)

        if args.cuda:
            self.vader = self.vader.cuda()
            self.orig_vader = self.orig_vader.cuda()

    def load_st(self, args):
        self.st = pd.read_csv (r"/content/drive/MyDrive/Colab Notebooks/st_scaled_50states.csv")
        self.st = self.st.iloc[:,1:]
        self.st_Bydate = self.st.sort_values(['date', 'state'])
        self.st = self.create_sequence2(lookback = args.window*50, window = args.window)
        self.st = torch.tensor(self.st).float()
        self.orig_st = self.st
        #print("sentence transformer data shape:", self.st.shape)

        if args.cuda:
            self.st = self.st.cuda()
            self.orig_st = self.orig_st.cuda()

    def create_sequence1(self, lookback, window):
      news_cos = []
      for i in range (int(len(self.news_Bydate)/50) - window): #data length=548
        sequence = self.news_Bydate[i*50:(i*50 + lookback)]
        sequence_re = pd.pivot_table(sequence, values='compound', index=['state'], columns=['date'])
        sequence_cos = cosine_similarity(sequence_re)
        news_cos.append(sequence_cos)
      return np.array(news_cos)

    def create_sequence2(self, lookback, window):
      news_cos = []
      for i in range (int(len(self.st_Bydate)/50) - window): #data length=548
        sequence = self.st_Bydate[i*50:(i*50 + lookback)]
        sequence_re = pd.pivot_table(sequence, index=['state'], columns=['date'])
        sequence_cos = cosine_similarity(sequence_re)
        news_cos.append(sequence_cos)
      return np.array(news_cos)

    def _pre_train(self, train, valid, test):
        self.train_set = train_set = range(self.P+self.h-1, train)
        self.valid_set = valid_set = range(train, valid)
        self.test_set = test_set = range(valid, self.n)
        self.tmp_train = self._batchify(train_set, self.h, useraw=True)
        train_mx = torch.cat((self.tmp_train[0][0], self.tmp_train[2]), 0).numpy() #199, 47
        self.max = np.max(train_mx, 0)
        self.min = np.min(train_mx, 0)
        self.peak_thold = np.mean(train_mx, 0)
        self.dat  = (self.rawdat  - self.min ) / (self.max  - self.min + 1e-12)
        #print(self.dat.shape)

    def _split(self, train, valid, test):
        self.train = self._batchify(self.train_set, self.h) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set, self.h)
        self.test = self._batchify(self.test_set, self.h)
        if (train == valid):
            self.val = self.test
 
    def _batchify(self, idx_set, horizon, useraw=False): ###tonights work
        
        n = len(idx_set)
        Y = torch.zeros((n, self.m))
        X2 = self.vader[:n, :, :]
        X3 = self.st[:n, :, :]
        if self.add_his_day and not useraw:
            X = torch.zeros((n, self.P+1, self.m))
        else:
            X = torch.zeros((n, self.P, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            if useraw: # for narmalization
                X[i,:self.P,:] = torch.from_numpy(self.rawdat[start:end, :])
                Y[i,:] = torch.from_numpy(self.Y[idx_set[i], :])
            else:
                his_window = self.dat[start:end, :]
                if self.add_his_day:
                    if idx_set[i] > 51 : # at least 52
                        his_day = self.dat[idx_set[i]-52:idx_set[i]-51, :]
                    else: # no history day data
                        his_day = np.zeros((1,self.m))
                    his_window = np.concatenate([his_day,his_window])
                    # print(his_window.shape,his_day.shape,idx_set[i],idx_set[i]-52,idx_set[i]-51)
                    X[i,:self.P+1,:] = torch.from_numpy(his_window) # size (window+1, m)
                else:
                    X[i,:self.P,:] = torch.from_numpy(his_window) # size (window, m)
                Y[i,:] = torch.from_numpy(self.normY[idx_set[i], :])
        return [X, X2, X3, Y]
   
    # original
    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]
        inputs2 = data[1]
        inputs3 = data[2]
        targets = data[3]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt,:]
            X2 = inputs2[excerpt,:]
            X3 = inputs3[excerpt,:]
            Y = targets[excerpt,:]
            if (self.cuda):
                X = X.cuda()
                X2 = X2.cuda()
                X3 = X3.cuda()
                Y = Y.cuda()
            model_inputs = Variable(X)
            model_inputs2 = Variable(X2)
            model_inputs3 = Variable(X3)
            data = [model_inputs, model_inputs2, model_inputs3, Variable(Y)]
            yield data
            start_idx += batch_size
