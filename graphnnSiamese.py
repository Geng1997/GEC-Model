import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score
import CBAM
from TestNewModel import GlobalPooling


def graph_embed(X, msg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
    # X -- affine(W1) -- ReLU -- (Message -- affine(W2) -- add (with aff W1)
    # -- ReLU -- )* MessageAll  --  output
    # [-1, N_x]中-1表示不用指定该维度大小,N_x是节点特征的维度，Wnode是W1[N_x,N_embed],N_embed是嵌入维度
    node_val = tf.reshape(tf.matmul(tf.reshape(X, [-1, N_x]), Wnode), [tf.shape(X)[0], -1, N_embed])
    
    cur_msg = tf.nn.relu(node_val)   # [batch, node_num, embed_dim] 作为第0层不用考虑连接节点，所以直接relu
    for t in range(iter_level):
        # Message convey
        Li_t = tf.matmul(msg_mask, cur_msg)  # [batch, node_num, embed_dim]
        # Complex Function
        cur_info = tf.reshape(Li_t, [-1, N_embed])
        for Wi in Wembed:
            if (Wi == Wembed[-1]):
                cur_info = tf.matmul(cur_info, Wi)
            else:
                cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
        neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
        # Adding
        tot_val_t = node_val + neigh_val_t
        # Nonlinearity
        tot_msg_t = tf.nn.tanh(tot_val_t)
        cur_msg = tot_msg_t   # [batch, node_num, embed_dim]

    g_embed = tf.reduce_sum(cur_msg, 1)   # [batch, embed_dim] tensorflow中0表示列，1表示行,所以上面是对cur_msg的行求和
    output = tf.matmul(g_embed, W_output) + b_output

    return output


def graph_CNN(input_x, N_embed):
    input_x_images = tf.reshape(input_x, [-1, 50, 50, 1])

    # 隐藏层,输出变成了 [50*50*32]
    conv1 = tf.layers.conv2d(
        inputs=input_x_images,
        filters=32,
        kernel_size=[3, 3],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    conv1 = CBAM.cbam_block(conv1)

    # 输出变成了[?,25,25,32]
    pool1 = tf.layers.average_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # 输出变成了  [?,25,25,64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    conv2 = CBAM.cbam_block(conv2)

    # 输出变成了[?,12,12,64]
    pool2 = tf.layers.average_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    conv3 = CBAM.cbam_block(conv3)

    # 输出变成了[?,6,6,128]
    pool3 = tf.layers.average_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=2
    )

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    conv4 = CBAM.cbam_block(conv4)

    # 输出变成了[?,3,3,256]
    pool4 = tf.layers.average_pooling2d(
        inputs=conv4,
        pool_size=[2, 2],
        strides=2
    )

    # flat(平坦化)
    # flat = tf.reshape(pool3, [-1, 6*6*128])
    flat = tf.reshape(pool4, [-1, 3 * 3 * 256])

    # densely-connected layers 全连接层,输出变成了[?,1024]
    dense = tf.layers.dense(
        inputs=flat,
        units=1024,
        activation=tf.nn.relu
    )

    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.5,
    )

    # 输出层，不用激活函数（本质就是一个全连接层）
    logits = tf.layers.dense(
        inputs=dropout,
        units=N_embed
    )
    # 输出形状[?,64]
    return logits


def FeatureFusion(x, weight, bias):
    layer1 = tf.add(tf.matmul(x, weight['h1']), bias['h1'])
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.add(tf.matmul(layer1, weight['h2']), bias['h2'])
    layer2 = tf.nn.relu(layer2)
    layer3 = tf.add(tf.matmul(layer2, weight['h3']), bias['h3'])
    layer3 = tf.nn.relu(layer3)
    out_layer = tf.add(tf.matmul(layer3, weight['out']), bias['out'])
    return out_layer


class graphnn(object):
    def __init__(self,
                    N_x,
                    Dtype, 
                    N_embed,
                    depth_embed,
                    N_o,
                    ITER_LEVEL,
                    lr,
                    device='/gpu:0'
                    # device='/cpu:0'
                ):

        self.NODE_LABEL_DIM = N_x

        tf.reset_default_graph()  # 用于清除默认图形堆栈并重置全局默认图形
        with tf.device(device):

            # 定义特征融合所需的元素
            n_input = 128
            n_hidden_1 = 1024
            n_hidden_2 = 1024
            n_hidden_3 = 1024
            n_class = 64

            # MLP隐藏层
            weight = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
                'out': tf.Variable(tf.random_normal([n_hidden_3, n_class]))
            }
            bias = {
                'h1': tf.Variable(tf.random_normal([n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_2])),
                'h3': tf.Variable(tf.random_normal([n_hidden_3])),
                'out': tf.Variable(tf.random_normal([n_class]))
            }

            Wnode = tf.Variable(tf.truncated_normal(
                shape = [N_x, N_embed], stddev = 0.1, dtype = Dtype))  # 变量的定义和初始化
            Wembed = []
            for i in range(depth_embed):
                Wembed.append(tf.Variable(tf.truncated_normal(
                    shape = [N_embed, N_embed], stddev = 0.1, dtype = Dtype)))

            W_output = tf.Variable(tf.truncated_normal(
                shape = [N_embed, N_o], stddev = 0.1, dtype = Dtype))
            b_output = tf.Variable(tf.constant(0, shape = [N_o], dtype = Dtype))
            
            X1 = tf.placeholder(Dtype, [None, None, N_x]) #[B, N_node, N_x]
            msg1_mask = tf.placeholder(Dtype, [None, None, None])
                                            #[B, N_node, N_node]
            self.X1 = X1
            self.msg1_mask = msg1_mask
            # 图嵌入
            embed1 = graph_embed(X1, msg1_mask, N_x, N_embed, N_o, ITER_LEVEL,
                    Wnode, Wembed, W_output, b_output)  #[B, N_x]

            # 图卷积
            g1_cnn = graph_CNN(msg1_mask, N_embed)
            # 特征融合
            embed1 = tf.concat((embed1, g1_cnn), 1)
            # embed1 = FeatureFusion(embed1, weight, bias)
            # embed1 = graph_CNN(msg1_mask, N_embed)

            X2 = tf.placeholder(Dtype, [None, None, N_x])
            msg2_mask = tf.placeholder(Dtype, [None, None, None])
            self.X2 = X2
            self.msg2_mask = msg2_mask
            embed2 = graph_embed(X2, msg2_mask, N_x, N_embed, N_o, ITER_LEVEL,
                    Wnode, Wembed, W_output, b_output)

            g2_cnn = graph_CNN(msg2_mask, N_embed)
            embed2 = tf.concat((embed2, g2_cnn), 1)
            # embed2 = FeatureFusion(embed2, weight, bias)
            # embed2 = graph_CNN(msg2_mask, N_embed)

            label = tf.placeholder(Dtype, [None, ])  # same: 1; different:-1
            self.label = label
            self.embed1 = embed1

            cos = tf.reduce_sum(embed1*embed2, 1) / tf.sqrt(tf.reduce_sum(
                embed1**2, 1) * tf.reduce_sum(embed2**2, 1) + 1e-10)

            diff = -cos
            self.diff = diff
            loss = tf.reduce_mean((diff + label) ** 2)
            self.loss = loss

            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            self.optimizer = optimizer
    
    def say(self, string):
        print(string)
        if self.log_file != None:
            self.log_file.write(string+'\n')
    
    def init(self, LOAD_PATH, LOG_PATH):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        self.sess = sess
        self.saver = saver
        self.log_file = None
        if (LOAD_PATH is not None):
            if LOAD_PATH == '#LATEST#':
                checkpoint_path = tf.train.latest_checkpoint('./')
            else:
                checkpoint_path = LOAD_PATH
            saver.restore(sess, checkpoint_path)
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'a+')
            self.say('{}, model loaded from file: {}'.format(
                datetime.datetime.now(), checkpoint_path))
        else:
            sess.run(tf.global_variables_initializer())
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'w')
            self.say('Training start @ {}'.format(datetime.datetime.now()))
    
    def get_embed(self, X1, mask1):
        vec, = self.sess.run(fetches=[self.embed1],
                feed_dict={self.X1:X1, self.msg1_mask:mask1})
        return vec

    def calc_loss(self, X1, X2, mask1, mask2, y):
        cur_loss = self.sess.run(fetches=[self.loss], feed_dict={self.X1:X1,
            self.X2:X2,self.msg1_mask:mask1,self.msg2_mask:mask2,self.label:y})
        return cur_loss
        
    def calc_diff(self, X1, X2, mask1, mask2):
        diff, = self.sess.run(fetches=[self.diff], feed_dict={self.X1:X1,
            self.X2:X2, self.msg1_mask:mask1, self.msg2_mask:mask2})
        return diff
    
    def train(self, X1, X2, mask1, mask2, y):
        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict={self.X1:X1,
            self.X2:X2,self.msg1_mask:mask1,self.msg2_mask:mask2,self.label:y})
        return loss
    
    def save(self, path, epoch=None):
        checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
        return checkpoint_path