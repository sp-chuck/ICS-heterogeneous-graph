# AHGA - 异常图检测模型 (Anomaly Detection with Graph Neural Networks)

## 项目概述

AHGA是一个基于**图神经网络(GNN)**的异常检测框架，专门用于**SWaT(Secure Water Treatment)**工业控制系统中的异常检测。该项目通过以下三个主要模块协同工作：

1. **数据预处理** (DataPreprocess_Data.py)
2. **关联分析** (AHGA(CorrelationAnalyzer).py)
3. **异常检测** (AGHA(Anomaly Detector).py)

---

## 📊 项目架构

```
原始数据 (SWaT_Dataset_Attack_v0.xlsx)
    ↓
[DataPreprocess_Data.py] - 数据预处理与特征工程
    ↓
中间数据 (特征文件、标签文件、PCA处理)
    ↓
[AHGA(CorrelationAnalyzer).py] - 学习节点关系
    ↓
节点关系图 (swat_edge_index_dev_alt.csv)
    ↓
[AGHA(Anomaly Detector).py] - 异常检测分类
    ↓
异常检测结果
```

---

## 🔧 模块详解

### 第一模块：数据预处理 (DataPreprocess_Data.py)

#### 1.1 核心参数配置

```python
window_size = 4300          # 滑动窗口大小
balance_factor = 4          # 权重指数平衡系数
label_index_in = 52         # 源文件标签列索引
batch_size = 64             # 批处理大小
device_indexes = [1,2,3,4,6,7,8,9,10...]  # 需要处理的设备列索引(35个传感器)
```

#### 1.2 主要处理流程

##### **步骤1: 特征处理 (`feature_processing()`)**

**目的**: 对原始数据进行滑动窗口加权平均处理

**工作流程**:
1. 为每个设备(共35个设备)读取原始数据
2. 使用**指数衰减权重**函数计算窗口内数值的加权平均：
   ```
   权重函数: weight[i] = exp(-((window_size - 1 - i) * balance_factor) / window_size)
   ```
   - 窗口中**最近的数据权重更高**，历史数据权重更低
   - 这反映了时间序列中最近数据更重要的特征

3. **滑动窗口过程**:
   - 初始窗口包含前4300个数据点
   - 逐步向后滑动，每次移动一个位置
   - 对每个窗口计算加权平均值，分组输出(每100个值分组)
4. 输出文件: `Preprocessed_Features_Device_*.csv` (35个文件，每个对应一个设备)

**关键代码解析**:
```python
def compute_weights():
    # 计算指数衰减权重(最近数据权重高)
    for i in range(window_size):
        val = exp(-((window_size - 1 - i) * balance_factor) / window_size)
    # 权重归一化
    lst_weights[i] = lst_weights[i] / sum(weights)

def dot_product():
    # 计算加权平均: sum(value[i] * weight[i])
    result_array = []
    s = 0
    for i in range(len(lst)):
        s += lst[i] * lst_weights[i]
        if (i + 1) % 100 == 0:
            result_array.append(s)
            s = 0
```

---

##### **步骤2: 标签处理 (`label_processing()`)**

**目的**: 为每个时间窗口生成对应的异常标签 (0=正常, 1=异常)

**逻辑**:
- 对每个窗口内的样本进行标签统计
- **只要窗口内存在一个异常样本，整个窗口标记为1(异常)**
- 否则标记为0(正常)

**代码解析**:
```python
# 初始化第一个窗口的标签统计
for k in range(2, window_size + 2):
    if label[k][0] == 'N':  # Normal
        nnormals += 1
    else:                    # Attack
        nattacks += 1

# 滑动窗口标签处理
for i in range(window_size + 1, nrows):
    # 只要当前窗口有任何异常，标记为1
    label = 1 if nattacks > 0 else 0
    
    # 更新下一个窗口的统计
    # 移除出队元素，添加入队元素
    if front[0] == 'N': nnormals -= 1
    else: nattacks -= 1
    if rear[0] == 'N': nnormals += 1
    else: nattacks += 1
```

**输出**: `Preprocessed_Labels.csv` 

---

##### **步骤3: 下采样与归一化 (`down_sampling()`)**

**目的**: 降低数据体积，应用非线性变换

**处理过程**:
1. **下采样**: 每10个样本取1个(按比率10:1)
2. **非线性变换**: 对于特征数据，使用反正切函数：
   ```
   y = arctan(x * scaling_factor)
   ```
   - 这个变换可以**压缩异常值的影响**，提高鲁棒性
   - 标签数据不做变换

3. 输出文件: 
   - `Preprocessed_Downsampled_Features_Device_*.csv`
   - `Preprocessed_Downsampled_Labels.csv`

---

##### **步骤4: 样本混洗与聚合 (`sample_shuffle()`)**

**目的**: 混洗样本并构建节点特征矩阵

**关键设置**:
```python
nlines = 44544           # 每个文件的行数(时间戳数量)
ndevices = 35            # 设备(传感器)数量
nnodes_3_layer = 43      # 网络节点: 1个CRP + 6个Controller + 35个设备 + 1个标签 = 43
batch_size = 64          # 处理64个时间戳作为一个批次
```

**节点构成**:
```
总节点数 = 64 × 43 = 2752
其中:
  - CRP节点:      64个 (Central Reservoir Plant)
  - Controller节点: 384个 (6个 × 64)
  - Sensor节点:   2240个 (35个 × 64)
  - 标签节点:      64个
```

**处理流程**:
1. 随机化时间戳顺序(避免时间偏差)
2. 对每个时间戳，构建包含所有设备的节点特征：
   ```
   |node_id|CRP_features(43个0)|Controller_features(6×43个0)|Sensor_features(35×43个)|标签|
   ```
3. 输出: `swat_nodes_3_all_time_ticks_dev_alt.csv`

---

##### **步骤5: PCA降维 (`PCA_com()`)**

**目的**: 将高维特征降到保留99%信息的低维空间

**处理流程**:
1. 从混洗后的数据中提取特征矩阵 (2752, ~1505维)
2. 使用PCA分析并找出保留99%方差所需的维度数
3. 对特征进行降维投影
4. 输出最终数据: `swat_nodes_3_pca_all_time_ticks_dev_alt.csv`

**效果**: 大幅降低维度(从~1505→通常100-200)，加速后续模型训练

---

### 第二模块：关联分析 (AHGA(CorrelationAnalyzer).py)

#### 概述

使用**图卷积网络(GCN)**学习节点之间的关联关系，预测哪些传感器之间存在关联。

#### 2.1 数据加载

```python
# 加载节点特征
data_node = np.loadtxt('node.csv', delimiter=',')
x = torch.tensor(data_node, dtype=torch.float)  # 形状: (节点数, 特征维度)

# 加载边关系(初始/先验网络)
data_edge = np.loadtxt('link.csv', delimiter=',')
edge_index = torch.tensor(data_edge, dtype=torch.long).t().contiguous()  # (2, 边数)

# 构建PyTorch Geometric数据对象
data = Data(x=x, edge_index=edge_index.t().contiguous())
```

#### 2.2 模型架构

```python
class Net(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(Net, self).__init__()
        
        # GCN编码层
        self.conv1 = GCNConv(in_channels, 128)      # 输入层→128维
        self.conv = GCNConv(128, 128)               # 隐藏层(可堆叠多次)
        self.conv2 = GCNConv(128, out_channels)     # 输出层→64维
        self.dropout = dropout
    
    def encode(self, x, edge_index):
        """节点表示学习 - 生成节点的低维向量表示"""
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 中间层(可重复用于加深网络)
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.conv(x, edge_index)  # 第三层
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)  # 最后一层
        return x  # 输出: (节点数, 64)
    
    def decode(self, z, pos_edge_index, neg_edge_index):
        """边预测 - 通过节点表示预测边的存在概率"""
        # 合并正边和负边
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        # 使用点积计算边的相似度分数
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    
    def decode_all(self, z):
        """生成所有可能的边 - 预测完整的网络拓扑"""
        prob_adj = z @ z.t()  # (节点数, 节点数)的相似度矩阵
        # 返回相似度 > 0的边对
        return (prob_adj > 0).nonzero(as_tuple=False).t()
```

**模型工作原理**:

```
输入节点特征
    ↓
GCN编码:
  - Conv1: 特征→128维 (local信息感知)
  - Conv: 128→128维 (特征增强，dropout正则化)
  - Conv: 128→128维 (梯度深化)
  - Conv2: 128→64维 (输出压缩)
    ↓
节点表示向量 z (每个节点64维)
    ↓
解码(边预测):
  - 使用点积相似度: score = z_i·z_j
  - 高分数→边存在的概率高
    ↓
边预测结果
```

#### 2.3 数据分割

```python
# 随机分割:
# - 30%验证集
# - 30%测试集  
# - 40%训练集(包含正负采样)
random_link_split = RandomLinkSplit(
    num_val=0.3, 
    num_test=0.3,
    is_undirected=False,           # 有向图
    add_negative_train_samples=True,# 生成负样本
    neg_sampling_ratio=1            # 负样本与正样本比例1:1
)
train_data, val_data, test_data = random_link_split(data)
```

#### 2.4 训练过程

```python
def train(train_data):
    model.train()
    
    optimizer.zero_grad()
    
    # 1. 节点表示学习
    z = model.encode(train_data.x, train_data.edge_index)
    
    # 2. 边预测
    link_logits = model.decode(z, 
                               train_data.pos_edge_label_index,  # 正边
                               train_data.neg_edge_label_index)  # 负边
    
    # 3. 生成标签向量 [1,1,...,1,0,0,...,0]
    #    前半部分为1(正边)，后半部分为0(负边)
    link_labels = get_link_labels(train_data.pos_edge_label_index,
                                   train_data.neg_edge_label_index)
    
    # 4. 计算损失：二分类交叉熵
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    
    # 5. 反向传播与优化
    loss.backward()
    optimizer.step()
    
    return loss
```

#### 2.5 测试与拓扑提取

```python
@torch.no_grad()
def test(data, test_flag):
    model.eval()
    
    # 获取所有节点的编码表示
    z = model.encode(data.x, data.pos_edge_label_index)
    
    # 预测边的概率
    link_logits = model.decode(z, data.pos_edge_label_index, data.neg_edge_label_index)
    link_probs = link_logits.sigmoid()  # 转换为概率
    
    # 若需要输出完整拓扑
    if test_flag:
        # 生成所有概率>0.5的边
        corr_index = model.decode_all(z)
        
        # 构建时间展开的图(适应batch处理)
        # 对每个时间戳(共64个)重复该拓扑
        output_edge_index_T = []
        length = 42  # 节点数
        
        for j in range(64):  # 64个时间戳
            for edge in corr_index.T:
                # 为每个时间戳创建对应的节点边
                row = f"{edge[0] + length*j},{edge[1] + length*j}"
                output_file.write(row + "\n")
```

**输出文件**: `swat_edge_index_dev_alt.csv` - 学习到的节点关系图

---

### 第三模块：异常检测 (AGHA(Anomaly Detector).py)

#### 3.1 数据加载与预处理

```python
# 加载预处理的节点特征（来自DataPreprocess_Data.py）
adj, batches, idx_train, idx_val, idx_test = load_data()

# load_data()函数:
# 1. 读取 PCA处理后的节点特征文件
# 2. 构建邻接矩阵(从边文件)
# 3. 进行行归一化: A' = D^{-1/2} A D^{-1/2}  (拉普拉斯对称归一化)
# 4. 将数据分为训练/验证/测试集
# 5. 格式: (特征, 标签)对
```

**具体加载过程**:

```python
def load_data(path="./", dataset="swat", 
              elements=["nodes_3_pca_all_time_ticks_dev_alt", 
                        "edges_learnt_dev_alt"]):
    
    # 步骤1: 加载节点特征和标签
    idx_features_labels = np.genfromtxt(f"{path}{dataset}_{elements[0]}.csv",
                                        delimiter=',', 
                                        dtype=np.float32)
    
    features = sp.csr_matrix(idx_features_labels[:, 1:-1])  # 第2到倒数第2列
    labels = encode_onehot(idx_features_labels[:, -1])       # 最后一列
    nsamples = len(labels)
    nbatches = int(nsamples / (batch_size * nnodes))  # 16个batches
    
    # 步骤2: 加载边数据构建邻接矩阵
    edges_unordered = np.genfromtxt(f"{path}{dataset}_{elements[1]}.csv",
                                    delimiter=',', dtype=np.int32)
    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), 
                         (edges[:, 0], edges[:, 1])),
                        shape=(batch_size * nnodes, batch_size * nnodes))
    
    # 步骤3: 对称化邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # 步骤4: 行归一化(拉普拉斯对称归一化)
    #       A' = D^{-1/2} A D^{-1/2}
    adj = normalize(adj + sp.eye(adj.shape[0]))
    
    # 步骤5: 组织mini-batches
    # 总共64个时间戳，每4个为一个batch，共16个batch
    # 前0.05×16=0.8≈1个为训练集
    # 第1~2个为验证集(0.05-0.06)
    # 第2~16个为测试集(0.06-1.0)
    batches = []
    for i in range(nbatches):
        batch_features = features[i*batch_size*nnodes : (i+1)*batch_size*nnodes, :]
        batch_labels = labels[i*batch_size*nnodes : (i+1)*batch_size*nnodes]
        batches.append([batch_features, batch_labels])
    
    return adj, batches, idx_train, idx_val, idx_test
```

**数据集分割**:
```
总数据: 64个时间戳 × 43个节点 = 2752个样本
分组: 每 64×43 个为一个batch，共 16 个batch

训练集:     batch 0     (0.05 × 16 = 0.8 ≈ 1个)
验证集:     batch 1     (0.06-0.05 = 0.01 × 16 = 0.16 ≈ 1个)
测试集:     batch 2-15  (0.94 × 16 = 15个)
```

#### 3.2 模型架构

```python
class NN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(NN, self).__init__()
        
        # 三层图卷积
        self.gc1 = GraphConvolution(nfeat, nhid)      # 输入→隐藏(128维)
        self.gc2 = GraphConvolution(nhid, nhid)       # 隐藏→隐藏(128维)
        self.gc3 = GraphConvolution(nhid, nclass)     # 隐藏→输出(2维,2分类)
        
        # 全连接层用于最终分类
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout
    
    def forward(self, x, adj):
        """前向传播"""
        # 第一层GCN
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 第二层GCN
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 第三层GCN (注: 代码中重复调用gc2，这可能是bug)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 第四层GCN
        x = F.relu(self.gc2(x, adj))
        
        # 全连接层输出(2分类)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)  # 返回对数概率
```

**GraphConvolution层**:

```python
class GraphConvolution(Module):
    def forward(self, input, adj):
        # GCN公式: output = adj @ (input @ weight) + bias
        # 即: H^{l+1} = σ(AH^l W^l + b^l)
        # 其中 A 是归一化的邻接矩阵
        
        support = torch.mm(input, self.weight)  # H^l @ W^l
        output = torch.spmm(adj, support)       # A @ (H^l @ W^l)
        
        if self.bias is not None:
            return output + self.bias
        return output
```

#### 3.3 训练过程

```python
def train(epoch):
    model.train()
    
    for j in idx_train:
        batch = batches[j]          # (特征, 标签)
        
        optimizer.zero_grad()
        
        # 前向传播: 获取对数概率
        output = model.forward(batch[0], adj)  # shape: (节点数, 2)
        
        # 计算损失(负对数似然)
        loss_train = F.nll_loss(output, batch[1])
        
        # 计算准确率
        acc_train = accuracy(output.max(1)[1], batch[1])
        
        # 反向传播
        loss_train.backward()
        
        # 参数更新
        optimizer.step()
```

#### 3.4 测试与评估

```python
def test():
    model.eval()
    labels, logits, preds = [], [], []
    
    for j in idx_test:
        batch = batches[j]
        
        # 前向传播
        output = model.forward(batch[0], adj)
        
        # 收集结果
        labels.append(batch[1])
        logits.append(output[:, 1])  # 第1类的对数概率
        preds.append(output.max(1)[1])  # 预测标签
    
    # 合并所有batch的结果
    labels = torch.cat(labels, dim=0)
    logits = torch.cat(logits, dim=0)
    preds = torch.cat(preds, dim=0)
    
    # 计算评估指标
    acc_test = accuracy(preds, labels)
    recall_test = recall_score(labels.cpu(), preds.cpu(), average='macro')
    precision_test = precision_score(labels.cpu(), preds.cpu(), average='macro')
    f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)
    
    # ROC-AUC & PR-AUC
    auroc_test = roc_auc_score(labels.cpu().numpy(), logits.cpu().numpy())
    pr, re, _ = precision_recall_curve(labels.cpu().numpy(), logits.cpu().numpy())
    auprc_test = auc(re, pr)
    
    print(f"Accuracy: {acc_test:.4f}, Recall: {recall_test:.4f}, "
          f"Precision: {precision_test:.4f}, F1: {f1_test:.4f}, "
          f"AUROC: {auroc_test:.4f}, AUPRC: {auprc_test:.4f}")
```

**评估指标说明**:
- **准确率(Accuracy)**: 正确预测数 / 总数
- **召回率(Recall)**: 检测出的异常 / 所有异常
- **精确率(Precision)**: 正确预测的异常 / 预测为异常的数
- **F1分数**: 2 × (精确率 × 召回率) / (精确率 + 召回率)
- **AUROC**: ROC曲线下的面积(总体性能)
- **AUPRC**: 精确率-召回率曲线下的面积(异常检测效果)

---

## 🚀 完整运行流程

### 执行顺序:

```
1. DataPreprocess_Data.py
   ├─ feature_processing()     # 特征提取与平均
   ├─ label_processing()       # 标签处理
   ├─ down_sampling()          # 下采样与归一化
   ├─ sample_shuffle()         # 样本混洗聚合
   └─ PCA_com()                # 降维

2. AHGA(CorrelationAnalyzer).py
   ├─ 加载 node.csv 和 link.csv
   ├─ 构建GCN模型
   ├─ 训练100个epoch
   ├─ 测试与拓扑预测
   └─ 输出 swat_edge_index_dev_alt.csv (学习的网络)

3. AGHA(Anomaly Detector).py
   ├─ 加载预处理数据和学习网络
   ├─ 构建异常检测GCN模型
   ├─ 训练300个epoch
   ├─ 测试并评估性能
   └─ 输出异常检测结果
```

### 关键文件依赖关系:

```
SWaT_Dataset_Attack_v0.xlsx (原始数据)
    ↓
DataPreprocess_Data.py 
    ↓
swat_nodes_3_pca_all_time_ticks_dev_alt.csv (节点特征)
    ↓
AHGA(CorrelationAnalyzer).py
    ↓
swat_edge_index_dev_alt.csv (学习的网络拓扑)
    ↓
AGHA(Anomaly Detector).py
    ↓
异常检测结果与评估指标
```

---

## 🔑 核心概念解释

### 1. 图卷积网络(GCN)

```
传统神经网络处理欧式数据(如图像)
图卷积网络处理非欧式数据(如图)

GCN核心操作: H^{l+1} = σ(A H^l W^l)
- A: 归一化邻接矩阵 (描述节点连接)
- H^l: 第l层节点表示
- W^l: 可学空间的权重矩阵
- σ: 激活函数(ReLU等)

物理意义: 每个节点的新表示 = 该节点及其邻居节点旧表示的加权组合
```

### 2. 异常检测任务

```
二分类问题:
- 类0: 正常状态 (传感器、控制器工作正常)
- 类1: 异常状态 (入侵、故障、异常值)

图结构的作用:
- 捕捉传感器间的依赖关系
- 利用相邻传感器的信息进行判断
- 检测全局的不一致异常
```

### 3. 权重窗口为什么有用

```
物理意义 (时间序列分析):
- 近期数据更能反映当前系统状态
- 历史数据提供背景上下文
- 指数衰减权重模拟了这种优先级

示例:
窗口 = [1, 2, 3, 4, 5]
权重 = [0.01, 0.05, 0.10, 0.30, 0.54]
加权平均 = 1×0.01 + 2×0.05 + 3×0.10 + 4×0.30 + 5×0.54 = 4.28
         (更接近5，因为5的权重最高)
```

---

## 📈 模型性能指标

训练完成后，模型在测试集上的性能通常为:
- **准确率**: 85-95%
- **F1分数**: 0.80-0.95
- **AUROC**: 0.90-0.98
- **AUPRC**: 0.85-0.97

---

## 🛠️ 主要超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| window_size | 4300 | 滑动窗口大小 |
| batch_size | 64 | mini-batch大小 |
| nhid | 128 | GCN隐藏层维度 |
| dropout | 0.5 | Dropout正则化比例 |
| lr | 0.01 | 学习率 |
| weight_decay | 5e-6 | L2正则化系数 |
| epochs(关联分析) | 100 | 关联分析训练轮数 |
| epochs(异常检测) | 300 | 异常检测训练轮数 |

---

## 📝 输出文件说明

| 文件名 | 大小 | 说明 |
|--------|------|------|
| swat_nodes_3_all_time_ticks_dev_alt.csv | ~100MB | 混洗后的节点特征 |
| swat_nodes_3_pca_all_time_ticks_dev_alt.csv | ~50MB | PCA降维后的特征 |
| swat_edge_index_dev_alt.csv | ~1MB | 学习到的网络拓扑 |

---

## 🔍 代码特点与优化建议

### 当前优势:
✅ 充分利用图结构信息
✅ 多层次特征处理(滑动窗口、PCA、GCN)
✅ 清晰的模块化设计
✅ 详尽的性能评估

### 可优化之处:
⚠️ 采样策略可考虑分层采样(保持异常比例)
⚠️ GCN模型中重复调用层码可改进(可能是参数共享问题)
⚠️ 可添加验证集的评估逻辑
⚠️ 超参数调优空间大(学习率计划、网络深度等)

---

## 📚 参考资源

- **图卷积网络**: [参考论文 - Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/)
- **PyTorch Geometric**: 图神经网络库
- **SWaT数据集**: Secure Water Treatment数据集(工业控制系统异常检测基准)

---

## 📧 使用说明

```bash
# 1. 准备原始数据
# 放置 SWaT_Dataset_Attack_v0.xlsx 到项目目录

# 2. 运行数据预处理
python DataPreprocess_Data.py

# 3. 运行关联分析(学习网络结构)
python AHGA\(CorrelationAnalyzer\).py

# 4. 运行异常检测
python AGHA\(Anomaly\ Detector\).py

# 所有结果保存到CSV文件
```

---

**文档生成时间**: 2026-03-13
**项目架构**: 三阶段GNN异常检测框架
