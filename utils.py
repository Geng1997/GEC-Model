import numpy as np
from sklearn.metrics import auc, roc_curve
from graphnnSiamese import graphnn
import json
import matplotlib.pyplot as plt

def get_f_name(DATA, SF, CM, OP, VS):
    F_NAME = []
    for sf in SF:
        for cm in CM:
            for op in OP:
                for vs in VS:
                    F_NAME.append(DATA+sf+cm+op+vs+".json")
    return F_NAME


def get_f_dict(F_NAME):
    name_num = 0
    name_dict = {}
    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                if (g_info['fname'] not in name_dict):
                    name_dict[g_info['fname']] = name_num
                    name_num += 1
    return name_dict

class graph(object):
    def __init__(self, node_num=0, label=None, name=None):
        self.node_num = node_num
        self.label = label
        self.name = name
        self.features = []
        self.succs = []
        self.preds = []
        if (node_num > 0):
            for i in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])
                
    def add_node(self, feature=[]):
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])
        
    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)

    def toString(self):
        ret = '{} {}\n'.format(self.node_num, self.label)
        for u in range(self.node_num):
            for fea in self.features[u]:
                ret += '{} '.format(fea)
            ret += str(len(self.succs[u]))
            for succ in self.succs[u]:
                ret += ' {}'.format(succ)
            ret += '\n'
        return ret


"""
    生成存储图和图对应的函数列表
"""
def read_graph(F_NAME, FUNC_NAME_DICT, FEATURE_DIM):
    graphs = []
    classes = []
    if FUNC_NAME_DICT != None:
        for f in range(len(FUNC_NAME_DICT)):
            classes.append([])
    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                # 数据集筛选，保存函数体块数大于等于5的作为数据集
                # if g_info['n_num'] >= 2 and g_info['n_num'] <= 10:
                if g_info['n_num'] > 10:
                    label = FUNC_NAME_DICT[g_info['fname']]  # 函数名组成的字典对应的标签值
                    classes[label].append(len(graphs))
                    cur_graph = graph(g_info['n_num'], label, g_info['src'])
                    for u in range(g_info['n_num']):
                        cur_graph.features[u] = np.array(g_info['features'][u])
                        for v in g_info['succs'][u]:
                            cur_graph.add_edge(u, v)
                    graphs.append(cur_graph)
    return graphs, classes


"""
    划分数据集，形成训练集、验证集和测试集
"""
def partition_data(Gs, classes, partitions, perm):
    C = len(classes)
    st = 0.0
    ret = []
    for part in partitions:
        cur_g = []
        cur_c = []
        ed = st + part * C
        for cls in range(int(st), int(ed)):
            prev_class = classes[perm[cls]]  # 获取当前未知所对应的函数集
            cur_c.append([])
            # 将目标函数对应的图加入cur_g
            for i in range(len(prev_class)):
                cur_g.append(Gs[prev_class[i]])
                cur_g[-1].label = len(cur_c)-1
                cur_c[-1].append(len(cur_g)-1)

        ret.append(cur_g)
        ret.append(cur_c)
        st = ed

    return ret


"""
    获取数据对pair
    X1、X2分别是数据对中的两个对应的数据
    m1、m2分别是与数据对中对应的邻接矩阵
    y是标签,1表示相似,-1表示不相似
    pos_id, neg_id是正例对集和反例对集
"""
# Gs是图,classes是function,M是batch size
def generate_epoch_pair(Gs, classes, M, output_id = False, load_id = None):
    epoch_data = []
    id_data = []   # [ ([(G0,G1),(G0,G1), ...], [(G0,H0),(G0,H0), ...]), ... ]

    if load_id is None:
        st = 0
        while st < len(Gs):
            if output_id:
                X1, X2, m1, m2, y, pos_id, neg_id = get_pair(Gs, classes,
                        M, st=st, output_id=True)
                id_data.append( (pos_id, neg_id) )
            else:
                X1, X2, m1, m2, y = get_pair(Gs, classes, M, st=st)
            epoch_data.append((X1,X2,m1,m2,y))
            st += M
    else:   ## Load from previous id data
        id_data = load_id
        for id_pair in id_data:
            X1, X2, m1, m2, y = get_pair(Gs, classes, M, load_id=id_pair)
            epoch_data.append((X1,X2,m1,m2,y))

    if output_id:
        return epoch_data, id_data
    else:
        return epoch_data


"""
    get_pair()用来获取数据对，形成正例和反例
    X1_input和X2_input是ACFG对应的特征
        例子:正例[(G_0, G_1)]、反例[(G_0, H_0)]，X1_input对应的是数据对中的G_0，X2_input对应的是数据对中的G_1和H_0
    node1_mask和node2_mask是ACFG中的节点，即邻接矩阵
    y_input存储的是数据对的标签，1表示相似，-1表示不相似
    pos_ids和neg_ids分别存放正例对和反例对
"""
def get_pair(Gs, classes, M, st = -1, output_id = False, load_id = None):
    if load_id is None:
        # 无现成数据集，需要制作
        C = len(classes)

        if (st + M > len(Gs)):
            M = len(Gs) - st
        ed = st + M

        pos_ids = []  # [(G_0, G_1)]  # 正例positive
        neg_ids = []  # [(G_0, H_0)]  # 反例negative

        for g_id in range(st, ed):
            g0 = Gs[g_id]
            cls = g0.label
            tot_g = len(classes[cls])
            if (len(classes[cls]) >= 2):
                g1_id = classes[cls][np.random.randint(tot_g)]
                while g_id == g1_id:
                    g1_id = classes[cls][np.random.randint(tot_g)]
                pos_ids.append( (g_id, g1_id) )  # 添加正例

            cls2 = np.random.randint(C)
            while (len(classes[cls2]) == 0) or (cls2 == cls):
                cls2 = np.random.randint(C)

            tot_g2 = len(classes[cls2])
            h_id = classes[cls2][np.random.randint(tot_g2)]
            neg_ids.append( (g_id, h_id) )  # 添加反例
    else:
        pos_ids = load_id[0]  # 存在现成的数据集，无需重新制作，[0]中存储所有正例,[1]中存储所有反例
        neg_ids = load_id[1]
        
    M_pos = len(pos_ids)
    M_neg = len(neg_ids)
    M = M_pos + M_neg

    # maxN1 = 0
    # maxN2 = 0
    # # 选取最大的长度构建数据集
    # for pair in pos_ids:
    #     maxN1 = max(maxN1, Gs[pair[0]].node_num)
    #     maxN2 = max(maxN2, Gs[pair[1]].node_num)
    # for pair in neg_ids:
    #     maxN1 = max(maxN1, Gs[pair[0]].node_num)
    #     maxN2 = max(maxN2, Gs[pair[1]].node_num)

    maxN1 = 50
    maxN2 = 50

    feature_dim = len(Gs[0].features[0])
    X1_input = np.zeros((M, maxN1, feature_dim))
    X2_input = np.zeros((M, maxN2, feature_dim))
    # node1_mask = np.zeros((M, 50, 50))
    # node2_mask = np.zeros((M, 50, 50))
    node1_mask = np.zeros((M, maxN1, maxN1))
    node2_mask = np.zeros((M, maxN2, maxN2))
    y_input = np.zeros((M))

    for i in range(M_pos):
        y_input[i] = 1
        g1 = Gs[pos_ids[i][0]]
        g2 = Gs[pos_ids[i][1]]
        for u in range(g1.node_num):
            try:
                X1_input[i, u, :] = np.array( g1.features[u] )
                for v in g1.succs[u]:
                    node1_mask[i, u, v] = 1
            except:
                continue
        for u in range(g2.node_num):
            try:
                X2_input[i, u, :] = np.array( g2.features[u] )
                for v in g2.succs[u]:
                    node2_mask[i, u, v] = 1
            except:
                continue
        
    for i in range(M_pos, M_pos + M_neg):
        y_input[i] = -1
        g1 = Gs[neg_ids[i-M_pos][0]]
        g2 = Gs[neg_ids[i-M_pos][1]]
        for u in range(g1.node_num):
            try:
                X1_input[i, u, :] = np.array( g1.features[u] )
                for v in g1.succs[u]:
                    node1_mask[i, u, v] = 1
            except:
                continue
        for u in range(g2.node_num):
            try:
                X2_input[i, u, :] = np.array( g2.features[u] )
                for v in g2.succs[u]:
                    node2_mask[i, u, v] = 1
            except:
                continue
    if output_id:
        return X1_input,X2_input,node1_mask,node2_mask,y_input,pos_ids,neg_ids
    else:
        return X1_input,X2_input,node1_mask,node2_mask,y_input


def train_epoch(model, graphs, classes, batch_size, load_data=None):
    if load_data is None:
        epoch_data = generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    perm = np.random.permutation(len(epoch_data))   # Random shuffle 随机排列一个数组

    cum_loss = 0.0
    for index in perm:
        cur_data = epoch_data[index]
        X1, X2, mask1, mask2, y = cur_data
        loss = model.train(X1, X2, mask1, mask2, y)
        cum_loss += loss

    return cum_loss / len(perm)


def get_auc_epoch(model, graphs, classes, batch_size, load_data=None):
    tot_diff = []
    tot_truth = []

    if load_data is None:
        epoch_data= generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    for cur_data in epoch_data:
        X1, X2, m1, m2, y = cur_data
        diff = model.calc_diff(X1, X2, m1, m2)
    #    print diff
        tot_diff += list(diff)
        tot_truth += list(y > 0)
    diff = np.array(tot_diff)
    truth = np.array(tot_truth)

    fpr, tpr, thres = roc_curve(truth, (1-diff)/2)
    model_auc = auc(fpr, tpr)

    return model_auc, fpr, tpr, thres


# 获取最佳阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def get_best_auc_epoch(model, graphs, classes, batch_size, load_data=None):
    tot_diff = []
    tot_truth = []

    if load_data is None:
        epoch_data= generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    print(len(epoch_data))
    cur_num = 1
    for cur_data in epoch_data:
        print(cur_num)
        X1, X2, m1, m2, y = cur_data
        diff = model.calc_diff(X1, X2, m1, m2)
        tot_diff += list(diff)
        tot_truth += list(y > 0)
        cur_num += 1
    diff = np.array(tot_diff)
    truth = np.array(tot_truth)

    fpr, tpr, thres = roc_curve(truth, (1-diff)/2)
    optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, thres)
    print(optimal_threshold, point)
    model_auc = auc(fpr, tpr)

    return diff, truth, model_auc, fpr, tpr, thres, optimal_threshold


def get_all_embed(model, graphs, classes, batch_size, load_data=None):
    all_embed = []

    if load_data is None:
        epoch_data= generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    print(len(epoch_data))
    cur_num = 1
    for cur_data in epoch_data:
        print(cur_num)
        X1, X2, m1, m2, y = cur_data
        diff = model.calc_diff(X1, X2, m1, m2)
        embed = model.get_embed(X1, m1)
        print(embed)
        cur_num += 1
        all_embed.append(embed)

    return all_embed


def plot_roc_curve(fpr, tpr):
    plt.figure()
    lw = 2
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(r'D:\Gemini-Repetition\AUC\CNN_small\CNN_big.pdf')
    plt.show()