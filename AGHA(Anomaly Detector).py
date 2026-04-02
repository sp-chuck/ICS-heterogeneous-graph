import warnings

import torch
import math
import time
import argparse
import random

import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

#from torch_geometric.nn import SAGEConv
#from torch_geometric.nn import GATConv

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, auc

warnings.filterwarnings ("ignore")

#device = torch.device ("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device ("cpu")

nnodes = 35 + 6 + 1
batch_size = 64

def encode_onehot (labels):

    #标签独热化，例如，有三个标签1，2，3，处理后分别为(1,0,0), (0,1,0), (0,0,1)
    classes = set (labels)
    classes_dict = {c: np.identity (len (classes)) [i, :] for i, c in enumerate (classes)}
    labels_onehot = np.array (list (map (classes_dict.get, labels)), dtype = np.int32)
    return labels_onehot


def normalize (mx):
    
    """Row-normalize sparse matrix"""
    rowsum = np.array (mx.sum (1))
    #r_inv = np.power (rowsum, -1).flatten ()
    r_inv = np.power (rowsum, -0.5).flatten ()   #拉普拉斯对称归一化
    r_inv [np.isinf (r_inv)] = 0.
    r_mat_inv = sp.diags (r_inv)
    mx = r_mat_inv.dot (mx)
    mx = mx.dot (r_mat_inv)    #拉普拉斯对称归一化
    return mx


def sparse_mx_to_torch_sparse_tensor (sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo ().astype (np.float32)
    indices = torch.from_numpy (np.vstack ((sparse_mx.row, sparse_mx.col)).astype (np.int64))
    values = torch.from_numpy (sparse_mx.data)
    shape = torch.Size (sparse_mx.shape)
    return torch.sparse.FloatTensor (indices, values, shape)

def load_data (path = "./", dataset = "swat", elements = ["nodes_3_pca_all_time_ticks_dev_alt", "edges_learnt_dev_alt"]):

    print ('Loading {} dataset...'.format (dataset))

    idx_features_labels = np.genfromtxt ("{}{}_{}.csv".format (path, dataset, elements [0]), delimiter = ',', dtype=np.dtype (np.float32))

    #从节点的.csv文件读取特征和标签数据
    features = sp.csr_matrix (idx_features_labels [:, 1:-1], dtype = np.float32)
    labels = encode_onehot (idx_features_labels [:, -1])
    nsamples = len (labels)
    nbatches = int (nsamples / (batch_size * nnodes))

    print ("Number of samples: %d" % nsamples)
    print ("Number of batches: %d" % nbatches)

    #build graph
    idx = np.array (idx_features_labels [:, 0], dtype = np.int32)
    idx_map = {j: i for i, j in enumerate (idx)}

    #从边的.csv文件读取边数据
    edges_unordered = np.genfromtxt ("{}{}_{}.csv".format (path, dataset, elements [1]), delimiter = ',', dtype=np.int32)
    #print (edges_unordered)
    #print (edges_unordered.shape [0])
    
    #edges = np.array (list (map (idx_map.get, edges_unordered.flatten ())), dtype=np.int32).reshape (edges_unordered.shape)
    edges = edges_unordered
    #通过边数据构建邻接矩阵adj
    adj = sp.coo_matrix ((np.ones (edges.shape [0]), (edges [:, 0], edges [:, 1])), shape = (batch_size * nnodes, batch_size * nnodes), dtype = np.float32)

    #邻接矩阵对称化和归一化
    #build symmetric adjacency matrix
    adj = adj + adj.T.multiply (adj.T > adj) - adj.multiply (adj.T > adj)

    #features = normalize (features)
    adj = normalize (adj + sp.eye (adj.shape [0]))

    #构建mini-batches列表
    batches = []
    for i in range (int (nbatches)):

        batches.append ([features [i * batch_size * nnodes : (i + 1) * batch_size * nnodes, :], labels [i * batch_size * nnodes : (i + 1) * batch_size * nnodes]])

    #分配训练、验证和测试集，总共64个时刻的数据，以4个时刻为一个mini-batch，总共16个batches，前10个batches为训练集，第11至第13个为验证集，第14至第16个为测试集
    idx_train = range (int (0.05 * nbatches))
    idx_val = range (int (0.05 * nbatches), int (0.06 * nbatches))
    idx_test = range (int (0.06 * nbatches), int (nbatches))    

    #将特征、标签、邻接矩阵、训练集、验证集和测试集处理成tensor，作为函数返回值
    features = torch.FloatTensor (np.array (features.todense ()))
    labels = torch.LongTensor (np.where (labels) [1])

    for batch in batches:
        batch [0] = torch.FloatTensor (np.array (batch [0].todense ())).to (device)
        batch [1] = torch.LongTensor (np.where (batch [1]) [1]).to (device)
    
    adj = sparse_mx_to_torch_sparse_tensor (adj)
    adj = adj.to (device)

    return adj, batches, idx_train, idx_val, idx_test

def accuracy (output, labels):
    #preds = output.max (1)[1].type_as (labels)
    preds = output
    #print ("Preds: ", preds)
    correct = preds.eq (labels).double ()
    correct = correct.sum ()
    return correct / len (labels)


class GraphConvolution (Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__ (self, in_features, out_features, bias = True):
        super (GraphConvolution, self).__init__ ()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter (torch.FloatTensor (in_features, out_features))
        if bias:
            self.bias = Parameter (torch.FloatTensor (out_features))
        else:
            self.register_parameter ('bias', None)
        self.reset_parameters ()

    def reset_parameters (self):
        stdv = 1. / math.sqrt (self.weight.size (1))
        self.weight.data.uniform_ (-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_ (-stdv, stdv)

    def forward (self, input, adj):
        support = torch.mm (input, self.weight)
        output = torch.spmm (adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__ (self):
        return self.__class__.__name__ + ' (' \
               + str (self.in_features) + ' -> ' \
               + str (self.out_features) + ')'

class NN (nn.Module):
    def __init__ (self, nfeat, nhid, nclass, dropout):
        super (NN, self).__init__ ()
        self.gc1 = GraphConvolution (nfeat, nhid)
        self.gc2 = GraphConvolution (nhid, nhid)
        self.gc3 = GraphConvolution (nhid, nclass)
        self.fc = nn.Linear (nhid, nclass)
        self.dropout = dropout

    def forward (self, x, adj):
        x = F.relu (self.gc1 (x, adj))
        x = F.dropout (x, self.dropout, training = self.training)
        x = F.relu (self.gc2 (x, adj))
        x = F.dropout (x, self.dropout, training = self.training)
        x = F.relu (self.gc2 (x, adj))
        x = F.dropout (x, self.dropout, training = self.training)
        x = F.relu (self.gc2 (x, adj))
        x = self.fc (x)
        return F.log_softmax (x, dim = 1)


# Load data
adj, batches, idx_train, idx_val, idx_test = load_data ()
print (adj)


# Model and optimizer
model = NN (nfeat = batches [0][0].shape [1], nhid = 128, nclass = batches [0][1].max ().item () + 1, dropout = 0.5).to (device)
optimizer = optim.Adam (model.parameters (), lr = 0.01, weight_decay = 5e-6)

def train (epoch):
    #t = time.time()
    model.train ()
    i = 0
    
    for j in idx_train:
        batch = batches [j]
        i += 1
        optimizer.zero_grad ()
        output = model.forward (batch [0], adj)
        loss_train = F.nll_loss (output, batch [1])
        acc_train = accuracy (output.max (1) [1], batch [1])
        loss_train.backward ()
        optimizer.step ()
    '''
    labels, logits, preds = [], [], []
    
    for j in idx_val:
        batch = batches [j]
        output = model.forward (batch [0], adj)        
        labels.append (batch [1])
        logits.append (output [:, 1])
        preds.append (output.max (1) [1])

    labels = torch.stack (labels, dim = 0)
    logits = torch.stack (logits, dim = 0)
    preds = torch.stack (preds, dim = 0)

    labels = torch.reshape (labels, (-1,))
    logits = torch.reshape (logits, (-1,))
    preds = torch.reshape (preds, (-1,))

    #print ("labels_val: \n", labels.shape)
    #print ("preds_val: \n", preds.shape)
    
    acc_val = accuracy (preds, labels)
    recall_val = recall_score (labels.cpu (), preds.cpu (), average = 'macro')
    precision_val = precision_score (labels.cpu (), preds.cpu (), average = 'macro')
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
    auroc_val = roc_auc_score (labels.cpu ().detach().numpy(), logits.cpu ().detach().numpy())
    
    pr, re, thresholds = precision_recall_curve (labels.cpu ().detach().numpy(), logits.cpu ().detach().numpy())
    auprc_val = auc (re, pr)
    
    if epoch % 100 == 0:
        print ('Epoch: {:04d}'.format (epoch + 1),
               'acc_val: {:.4f}'.format (acc_val.item ()),
               'recall_val: {:.4f}'.format (recall_val.item ()),
               "precision = {:.4f}".format (precision_val.item ()),
               "f1 = {:.4f}".format (f1_val.item ()),
               "auroc = {:.4f}".format (auroc_val.item ()),
               "auprc = {:.4f}".format (auprc_val.item ())
               )
    '''

def test ():
    
    model.eval ()
    labels, logits, preds = [], [], []
    
    for j in idx_test:
        batch = batches [j]
        output = model.forward (batch [0], adj)
        #loss_test = F.nll_loss (output, batch [1])
        labels.append (batch [1])
        logits.append (output [:, 1])
        preds.append (output.max (1) [1])

    labels = torch.stack (labels, dim = 0)
    logits = torch.stack (logits, dim = 0)
    preds = torch.stack (preds, dim = 0)

    labels = torch.reshape (labels, (-1,))
    logits = torch.reshape (logits, (-1,))
    preds = torch.reshape (preds, (-1,))
    
    acc_test = accuracy (preds, labels)
    recall_test = recall_score (labels.cpu (), preds.cpu (), average = 'macro')
    precision_test = precision_score (labels.cpu (), preds.cpu (), average = 'macro')
    f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)
    auroc_test = roc_auc_score (labels.cpu ().detach().numpy(), logits.cpu ().detach().numpy())
    
    pr, re, thresholds = precision_recall_curve (labels.cpu ().detach().numpy(), logits.cpu ().detach().numpy())
    auprc_test = auc (re, pr)
    
    print ("Test set results:",
           "accuracy = {:.4f}".format (acc_test.item ()),
           "recall = {:.4f}".format (recall_test.item ()),
           "precision = {:.4f}".format (precision_test.item ()),
           "f1 = {:.4f}".format (f1_test.item ()),
           "auroc = {:.4f}".format (auroc_test.item ()),
           "auprc = {:.4f}".format (auprc_test.item ()),
           )

ntimes = 1
    
for i in range (ntimes):
    # Train model
    nepoches = 300
    print ("nepoches:", nepoches)
    t = time.time ()
    for epoch in range (nepoches):
        train (epoch)
    print ("Training time: ", time.time () - t)

    #print ("Optimization Finished!")
    #print ("Total time elapsed: {:.4f}s".format (time.time () - t_total))

    # Testing
    t = time.time ()
    test ()
    print ("Test time: ", time.time () - t)
    
    model = NN (nfeat = batches [0][0].shape [1], nhid = 128, nclass = batches [0][1].max ().item () + 1, dropout = 0.5).to (device)
    optimizer = optim.Adam (model.parameters (), lr = 0.01, weight_decay = 5e-6)
