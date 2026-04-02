import pandas as pd
import numpy as np
import torch
import time

import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
#from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

data_node = np.loadtxt ('node.csv', delimiter = ',')
data_node = data_node.tolist ()
print ("#nodes: %d, #features: %d" % (len (data_node), len (data_node [0])))
#print (data_node)

x = torch.tensor(data_node, dtype=torch.float)
#print (x)
print ("Node features loaded successfully!")

data_edge = np.loadtxt ('link.csv', delimiter = ',')
data_edge = data_edge.tolist ()
print ("#edges: %d" % len (data_edge))

for edge in data_edge:
        edge [0], edge [1] = int (edge [0]), int (edge [1])
#print (data_edge)

edge_index = torch.tensor (data_edge, dtype = torch.long)
#print (edge_index)
print ("All positive links loaded successfully!")

data = Data (x = x, edge_index = edge_index.t ().contiguous ())
print (data)
print ("Data object created successfully!\n")


class Net (nn.Module):
        
	def __init__ (self, in_channels, out_channels, dropout = 0.5):
		super (Net, self).__init__ ()
		self.conv1 = GCNConv (in_channels, 128)
		self.conv = GCNConv (128, 128)
		self.conv2 = GCNConv (128, out_channels)
		self.dropout = dropout

	def encode (self, x, edge_index): # 节点表征学习
		x = self.conv1 (x, edge_index)   # 第一层
		#x = x.relu ()
		x = F.leaky_relu(x, negative_slope = 0.2)
		x = F.dropout (x, self.dropout, training = self.training)

		x = self.conv (x, edge_index)    # 第二层，适用于层数为3和4的情况，层数为2时应注释掉
		x = F.leaky_relu(x, negative_slope = 0.2)
		x = F.dropout (x, self.dropout, training = self.training)

		x = self.conv (x, edge_index)    # 第三层，适用于层数为4的情况，层数为2或3时应注释掉
		x = F.leaky_relu(x, negative_slope = 0.2)
		x = F.dropout (x, self.dropout, training = self.training)
		
		x = self.conv2 (x, edge_index)   # 最后一层
		return x

	def decode (self, z, pos_edge_index, neg_edge_index): # z传入经过表征学习的所有节点特征矩阵
		edge_index = torch.cat ([pos_edge_index, neg_edge_index], dim=-1) # dim=-1, 2维就是1
		return (z [edge_index [0]] * z [edge_index [1]]).sum (dim=-1) # 头尾节点属性对应相乘后求和
		# 返回一个 [(正样本数+负样本数),1] 的向量

	def decode_all (self, z):
		prob_adj = z @ z.t () # 头节点属性和尾节点属性对应相乘后求和，[节点数，节点数]
		return (prob_adj > 0).nonzero (as_tuple = False).t () # [2,m], 列存储有边的nodes的序号


data.train_mask = data.val_mask = data.test_mask = data.y = None
#data = train_test_split_edges (data, val_ratio = 0.3, test_ratio = 0.3)
random_link_split = RandomLinkSplit (num_val = 0.3, num_test = 0.3, is_undirected = False, split_labels = True, add_negative_train_samples = True, neg_sampling_ratio = 1)
train_data, val_data, test_data = random_link_split (data)
print ("Datasets defined successfully!")

'''
print ("\n\n")

print ("Training data: \n", train_data)
print ("Validation data: \n", val_data)
print ("Test data: \n", test_data)

print ("\n\n")

print ("Edge indexes: \n")
print ("Training data: ", train_data.edge_index)
print ("Validaiton data: ", val_data.edge_index)
print ("Test data", test_data.edge_index)

print ("\n\n")

print ("Edge labels: \n")
print ("Training data: \n\tPositive samples: ")
print ("\t\t", train_data.pos_edge_label)
print ("\tNegative samples: ")
print ("\t\t", train_data.neg_edge_label)

print ("Validaiton data: \n\tPositive samples: ")
print ("\t\t", val_data.pos_edge_label)
print ("\tNegative samples: ")
print ("\t\t", val_data.neg_edge_label)

print ("Test data: \n\tPositive samples: ")
print ("\t\t", test_data.pos_edge_label)
print ("\tNegative samples: ")
print ("\t\t", test_data.neg_edge_label)

print ("\n\n")

print ("Edge label indexes: \n")
print ("Training data: \n\tPositive samples: ")
print ("\t", train_data.pos_edge_label_index)
print ("\tNegative samples: ")
print ("\t", train_data.neg_edge_label_index)

print ("Validaiton data: \n\tPositive samples: ")
print ("\t", val_data.pos_edge_label_index)
print ("\tNegative samples: ")
print ("\t", val_data.neg_edge_label_index)

print ("Test data: \n\tPositive samples: ")
print ("\t", test_data.pos_edge_label_index)
print ("\tNegative samples: ")
print ("\t", test_data.neg_edge_label_index)

print ("\n\n")
'''

device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')
model = Net (data.num_features, 64).to (device)
data = data.to (device)
optimizer = torch.optim.Adam (params = model.parameters (), lr = 0.01)
print ("Model initialized successfully!")

# 训练
def train (train_data):        
	model.train ()

	# 负采样，只对训练集中不存在边的节点采样
	#neg_edge_index = negative_sampling (edge_index = data.train_pos_edge_index, num_nodes = data.num_nodes, num_neg_samples = data.train_pos_edge_index.size (1))
	
	optimizer.zero_grad ()
	
	# 节点表征学习
	z = model.encode (train_data.x, train_data.edge_index) 
	# 有无边的概率计算
	link_logits = model.decode (z, train_data.pos_edge_label_index, train_data.neg_edge_label_index)
	# 真实边情况[0,1]，调用get_link_labels
	link_labels = get_link_labels (train_data.pos_edge_label_index, train_data.neg_edge_label_index).to (device)
	# 损失计算
	loss = F.binary_cross_entropy_with_logits (link_logits, link_labels)
	# 反向求导
	loss.backward ()
	# 迭代
	optimizer.step ()
	
	return loss

# 生成正负样本边的标记
def get_link_labels (pos_edge_index, neg_edge_index):
	num_links = pos_edge_index.size (1) + neg_edge_index.size (1)
	link_labels = torch.zeros (num_links, dtype = torch.float) # 向量
	link_labels [:pos_edge_index.size (1)] = 1
	return link_labels
									

# 测试
@torch.no_grad()
def test (data, test_flag):
        model.eval ()
        results = []
        t = time.time ()

        # 计算所有的节点表征
        z = model.encode (data.x, data.pos_edge_label_index)
    
        # 有无边的概率预测
        link_logits = model.decode (z, data.pos_edge_label_index, data.neg_edge_label_index)
        link_probs = link_logits.sigmoid ()
        #print ("Time consumed: ", time.time () - t)
        
        # 真实情况
        link_labels = get_link_labels (data.pos_edge_label_index, data.neg_edge_label_index)
        # 存入准确率
        #results.append (roc_auc_score (link_labels.cpu (), link_probs.cpu ()))
        results.append (link_labels)
        results.append (link_probs)
        
        if test_flag:
                output_edge_index_T = []
                f = open ("swat_edge_index_dev_alt.csv", "w")
                corr_index = model.decode_all (z)
                #print ("Edge index size: ", corr_index.shape)
                #print ("Edge index: ", corr_index)

                for i in range (corr_index.shape [1]):
                        e = [corr_index [0][i].tolist (), corr_index [1][i]. tolist ()]
                        #print ("\t", e)

                        if e [0] < 0 or e [1] < 0:
                                continue
                        output_edge_index_T.append (e)

                #output_edge_index_T = output_edge_index_T + topology_index    #base + correlation
                
                #print ("Output Edge Index: \n", output_edge_index_T)

                length = 42

                for j in range (64):
                        for e in output_edge_index_T:
                                row = str (e [0] + length * j) + "," + str (e [1] + length * j) + "\n"
                                f.write (row)
                
                f.close ()

        return results


#best_val_auc = test_auc = 0
v_epoch = []
v_loss = []
v_val = []
v_test = []
nepochs = 100

for epoch in range (nepochs):
        loss = train (train_data)
        val_results = test (val_data, 0) # 训练一次计算一次验证准确率
        predicted_val_labels = []
        for i in val_results [1]:
                if i > 0.5:
                        predicted_val_labels.append (1)
                else:
                        predicted_val_labels.append (0)
        recall_val = recall_score (val_results [0], predicted_val_labels)
        precision_val = precision_score (val_results [0], predicted_val_labels)
        f1_val = f1_score (val_results [0], predicted_val_labels)
        accuracy_val = accuracy_score (val_results [0], predicted_val_labels)
        #print (val_results)
        auc_val = roc_auc_score (val_results [0].cpu(), val_results [1].cpu())
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Recall_val: {recall_val:.4f}, Precision_val: {precision_val:.4f}, F1_val: {f1_val:.4f}, Accuracy_val: {accuracy_val:.4f}, AUC_val: {auc_val:.4f}') # 03d，不足3位前面补0，大于3位照常输出
        v_epoch.append (epoch + 1)
        v_loss.append (float (loss))
        v_val.append (auc_val)
        #v_test.append (test_auc)

test_results = test (test_data, 1)
print ("test_results [0](link_labels): ", test_results [0])
print ("test_results [1](link_probs): ", test_results [1])

predicted_labels = []

for i in test_results [1]:
        if i > 0.5:
                predicted_labels.append (1)
        else:
                predicted_labels.append (0)
        
print (predicted_labels)
#print (test_results [1])

recall_test = recall_score (test_results [0], predicted_labels)
precision_test = precision_score (test_results [0], predicted_labels)
f1_test = f1_score (test_results [0], predicted_labels)
accuracy_test = accuracy_score (test_results [0], predicted_labels)
auc_test = roc_auc_score (test_results [0].cpu(), test_results [1].cpu())

print ("Test results:\n\tRecall: ", recall_test, "Precision: ", precision_test, "F1 score: ", f1_test, "Accuracy: ", accuracy_test, "AUC: ", auc_test)

'''
csv_data = np.vstack ((v_epoch, v_loss, v_val, v_test)).T
Coordinates = pd.DataFrame (np.mat (csv_data))
headers = ['epoch','loss','val_auc','test_auc']
Coordinates.to_csv ("GCN_2_D20.csv", header=headers, index = 0)
'''

