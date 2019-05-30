# coding=utf-8
'''
Created on 2017.5.3

@author: Ann

Att----Transformer
'''

# import time
# import sys
# import subprocess
# import os
# import random
from keras.models import Model
from keras.layers import Dense, Activation, Input, merge, Conv2D, MaxPooling2D, LSTM, GRU, Embedding,GlobalMaxPool1D,Conv1D,MaxPooling1D
from keras.layers.core import Dropout,initializers
from keras.layers.core import Flatten, Lambda, RepeatVector, Reshape,Permute
# from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate,multiply
from keras.layers import Multiply
from keras import optimizers

from keras.preprocessing import sequence
from keras.engine.topology import Layer

from optparse import OptionParser

import pickle as pkl
import gzip
from keras import backend as K
# from keras import activations, initializations, regularizers, constraints
import numpy as np
# np.random.seed(345)

# evaluation of DDI extraction results. 4 DDI tpyes
def result_evaluation(y_test,pred_test):

    pred_matrix = np.zeros_like(pred_test, dtype=np.int8)

    y_matrix = np.zeros_like(y_test, dtype=np.int8)
    pred_indexs = np.argmax(pred_test, 1)
    y_indexs = np.argmax(y_test, 1)

    for i in range(len(pred_indexs)):
        pred_matrix[i][pred_indexs[i]] = 1
        y_matrix[i][y_indexs[i]] = 1

    count_matrix=np.zeros((4,3))
    for class_idx in range(4):

        count_matrix[class_idx][0] = np.sum(np.array(pred_matrix[:, class_idx]) * np.array(y_matrix[:, class_idx]))#tp
        count_matrix[class_idx][1] = np.sum(np.array(pred_matrix[:, class_idx]) * (1 - np.array(y_matrix[:, class_idx])))#fp
        count_matrix[class_idx][2] = np.sum((1 - np.array(pred_matrix[:, class_idx])) * np.array(y_matrix[:, class_idx]))#fn

    sumtp=sumfp=sumfn=0

    for i in range(4):
        sumtp+=count_matrix[i][0]
        sumfp+=count_matrix[i][1]
        sumfn+=count_matrix[i][2]

        precision=recall=f1=0

    if (sumtp + sumfp) == 0:
        precision = 0.
    else:
        precision = float(sumtp) / (sumtp + sumfp)

    if (sumtp + sumfn) == 0:
        recall = 0.
    else:
        recall = float(sumtp) / (sumtp + sumfn)

    if (precision + recall) == 0.:
        f1 = 0.
    else:
        f1 = 2 * precision * recall / (precision + recall)

    from sklearn.metrics import classification_report
    print(classification_report(y_indexs, pred_indexs, target_names=["Effect", "Mechanism", "Advice", "Int", "Negative"]))

    return precision,recall,f1


from keras.engine.topology import Layer


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                2 * K.arange(self.size / 2, dtype='float32' \
                                             ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class emb_AttentionLayer(Layer):

    def __init__(self, nb_head, size_per_head, mask_right=False, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        self.mask_right = mask_right
        super(emb_AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(emb_AttentionLayer, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(A[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e12
            A = A - mask
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

#自定义评价函数
def precision(y_true, y_pred):
    print('precision:',y_true.shape[1])
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    print('true_positives:', true_positives)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    print('Fmeasure',fbeta_score(y_true, y_pred, beta=1))
    return fbeta_score(y_true, y_pred, beta=1)



if __name__ == '__main__':

    s = {
        'emb_dimension': 200,  # dimension of word embedding
        'mini_batch': 100,
        'shortest_part_length': 12,
        'epochs': 200,
        'class_type': 5,
        'binary_class':2,
        'first_hidden_layer': 200,
        'lr': 0.001,
        'emb_dropout': 0.7,
        'dense_dropout': 0.5,
        'train_file': "D:/ay/detection-ML/PreprocessData/train.pkl.gz",
        'test_file': "D:/ay/detection-ML/PreprocessData/test.pkl.gz",
        'wordvecfile': "D:/ay/detection-ML/PreprocessData/vec.pkl.gz",

    }

    outpath = '../data/'
    nb_class = s['class_type']
    binary_class = s['binary_class']


    # read train and test data which are pkl files
    f_Train = gzip.open(s['train_file'], 'rb')
    train_labels_vec = pkl.load(f_Train)
    train_word = pkl.load(f_Train)

    train_POS = pkl.load(f_Train)

    train_distances = pkl.load(f_Train)

    train_distances2 = pkl.load(f_Train)

    train_shortest_word = pkl.load(f_Train)

    train_shortest_pos = pkl.load(f_Train)
    train_shortest_dis1 = pkl.load(f_Train)
    train_shortest_dis2 = pkl.load(f_Train)

    train_entity = pkl.load(f_Train)

    f_Train.close()

    f_Test = gzip.open(s['test_file'], 'rb')
    test_labels_vec = pkl.load(f_Test)
    test_word = pkl.load(f_Test)
    test_POS = pkl.load(f_Test)
    test_distances = pkl.load(f_Test)

    test_distances2 = pkl.load(f_Test)

    test_shortest_word = pkl.load(f_Test)

    test_shortest_pos = pkl.load(f_Test)
    test_shortest_dis1 = pkl.load(f_Test)
    test_shortest_dis2 = pkl.load(f_Test)

    test_entity = pkl.load(f_Test)

    f_Test.close()

    f_word2vec = gzip.open(s['wordvecfile'], 'rb')
    vec_table = pkl.load(f_word2vec, encoding='latin1')
    pos_vec_table = pkl.load(f_word2vec, encoding='latin1')
    dis_vec_table = pkl.load(f_word2vec, encoding='latin1')

    f_word2vec.close()

    #binary labels[1,0]positive,[0,1]negative
    test_binary = [[1,0] if test_labels_vec[i][4]==0 else [0,1] for i in range(len(test_labels_vec))]
    train_binary = [[1,0] if train_labels_vec[i][4]==0 else [0,1] for i in range(len(train_labels_vec))]
    train_labels=[]
    for i in range(len(train_labels_vec)):
        temp = [np.array(train_labels_vec[i])]
        if train_labels_vec[i][4]==0:
            temp.append(np.array([1,0]))
        else:
            temp.append(np.array([0,1]))
        temp=np.array(temp)
        train_labels.append(temp)

    train_labels = np.array(train_labels)
    # print test_labels_vec
    answer_y = np.array(test_labels_vec, dtype=np.int8)
    train_y_mulity = np.array(train_labels_vec, dtype=np.int8)

    answer_y_binary = np.array(test_binary, dtype=np.int8)
    train_y_binary = np.array(train_binary, dtype=np.int8)

    # train_y = [train_y_mulity,train_y_binary]
    test_word_bak = test_word

    word_dep_max = 0
    word_max_num = 0
    temp_max = 0

    new_train_word = []
    new_train_pos = []
    new_train_dis = []
    new_train_dis2 = []
    new_test_word = []
    new_test_pos = []
    new_test_dis = []
    new_test_dis2 = []

    word_length = []

    # calculate the max length of each subsequence
    for i in range(3):

        temp_max = 0
        temp_train_word = []
        temp_train_pos = []
        temp_train_dis = []
        temp_train_dis2 = []
        for j in range(len(train_word)):

            assert len(train_word[j][i]) == len(train_POS[j][i])
            assert len(train_POS[j][i]) == len(train_distances[j][i])
            assert len(train_distances[j][i]) == len(train_distances2[j][i])
            temp_train_word.append(train_word[j][i])
            temp_train_pos.append(train_POS[j][i])
            temp_train_dis.append(train_distances[j][i])
            temp_train_dis2.append(train_distances2[j][i])
            if len(train_word[j][i]) > temp_max:
                temp_max = len(train_word[j][i])

        new_train_word.append(temp_train_word)
        new_train_pos.append(temp_train_pos)
        new_train_dis.append(temp_train_dis)
        new_train_dis2.append(temp_train_dis2)

        temp_test_word = []
        temp_test_pos = []
        temp_test_dis = []
        temp_test_dis2 = []
        for j in range(len(test_word)):
            assert len(test_word[j][i]) == len(test_POS[j][i])
            assert len(test_POS[j][i]) == len(test_distances[j][i])
            assert len(test_distances[j][i]) == len(test_distances2[j][i])
            temp_test_word.append(test_word[j][i])
            temp_test_pos.append(test_POS[j][i])
            temp_test_dis.append(test_distances[j][i])
            temp_test_dis2.append(test_distances2[j][i])
            if len(test_word[j][i]) > temp_max:
                temp_max = len(test_word[j][i])

        new_test_word.append(temp_test_word)
        new_test_pos.append(temp_test_pos)
        new_test_dis.append(temp_test_dis)
        new_test_dis2.append(temp_test_dis2)
        word_length.append(temp_max)

    train_word = new_train_word

    train_POS = new_train_pos
    train_distances = new_train_dis
    train_distances2 = new_train_dis2

    test_word = new_test_word

    test_POS = new_test_pos
    test_distances = new_test_dis
    test_distances2 = new_test_dis2

    train_entity_0 = []
    train_entity_1 = []
    test_entity_0 = []
    test_entity_1 = []
    for i in range(len(train_entity)):
        temp_list = []
        temp_list.append(train_entity[i][0])
        train_entity_0.append(temp_list)
        temp_list = []
        temp_list.append(train_entity[i][1])
        train_entity_1.append(temp_list)
    for i in range(len(test_entity)):
        temp_list = []
        temp_list.append(test_entity[i][0])
        test_entity_0.append(temp_list)
        temp_list = []
        temp_list.append(test_entity[i][1])
        test_entity_1.append(temp_list)

    print('Pad sequences (samples x time)')

    train_word_list = []
    test_word_list = []
    train_POS_list = []
    test_POS_list = []

    train_distances_list = []
    test_distances_list = []
    train_distances2_list = []
    test_distances2_list = []

    for i in range(len(train_word)):
        train_word_list.append(
            sequence.pad_sequences(train_word[i], maxlen=word_length[i], truncating='post', padding='post'))
    for i in range(len(test_word)):
        test_word_list.append(sequence.pad_sequences(test_word[i], maxlen=word_length[i], truncating='post',
                                                     padding='post'))

    for i in range(len(train_POS)):
        train_POS_list.append(
            sequence.pad_sequences(train_POS[i], maxlen=word_length[i], truncating='post', padding='post'))
    for i in range(len(test_distances)):
        test_POS_list.append(sequence.pad_sequences(test_POS[i], maxlen=word_length[i], truncating='post',
                                                    padding='post'))

    for i in range(len(train_distances)):
        train_distances_list.append(
            sequence.pad_sequences(train_distances[i], maxlen=word_length[i], truncating='post', padding='post'))
    for i in range(len(test_distances)):
        test_distances_list.append(sequence.pad_sequences(test_distances[i], maxlen=word_length[i], truncating='post',
                                                          padding='post'))

    for i in range(len(train_distances2)):
        train_distances2_list.append(
            sequence.pad_sequences(train_distances2[i], maxlen=word_length[i], truncating='post', padding='post'))
    for i in range(len(test_distances2)):
        test_distances2_list.append(sequence.pad_sequences(test_distances2[i], maxlen=word_length[i], truncating='post',
                                                           padding='post'))

    train_shortest_word = sequence.pad_sequences(train_shortest_word, maxlen=s['shortest_part_length'],
                                                 truncating='post', padding='post')
    test_shortest_word = sequence.pad_sequences(test_shortest_word, maxlen=s['shortest_part_length'], truncating='post',
                                                padding='post')

    train_shortest_pos = sequence.pad_sequences(train_shortest_pos, maxlen=s['shortest_part_length'], truncating='post',
                                                padding='post')
    test_shortest_pos = sequence.pad_sequences(test_shortest_pos, maxlen=s['shortest_part_length'], truncating='post',
                                               padding='post')
    train_shortest_dis1 = sequence.pad_sequences(train_shortest_dis1, maxlen=s['shortest_part_length'],
                                                 truncating='post', padding='post')
    test_shortest_dis1 = sequence.pad_sequences(test_shortest_dis1, maxlen=s['shortest_part_length'], truncating='post',
                                                padding='post')
    train_shortest_dis2 = sequence.pad_sequences(train_shortest_dis2, maxlen=s['shortest_part_length'],
                                                 truncating='post', padding='post')
    test_shortest_dis2 = sequence.pad_sequences(test_shortest_dis2, maxlen=s['shortest_part_length'], truncating='post',
                                                padding='post')

    train_entity_0 = sequence.pad_sequences(train_entity_0, maxlen=1, truncating='post', padding='post')
    train_entity_1 = sequence.pad_sequences(train_entity_1, maxlen=1, truncating='post', padding='post')
    test_entity_0 = sequence.pad_sequences(test_entity_0, maxlen=1, truncating='post', padding='post')
    test_entity_1 = sequence.pad_sequences(test_entity_1, maxlen=1, truncating='post', padding='post')



    # embedding layer
    wordembedding = Embedding(vec_table.shape[0],
                              vec_table.shape[1],
                              weights=[vec_table])

    disembedding = Embedding(dis_vec_table.shape[0],
                             dis_vec_table.shape[1],
                             weights=[dis_vec_table]
                             )

    posembedding = Embedding(pos_vec_table.shape[0],
                             pos_vec_table.shape[1],
                             weights=[pos_vec_table]
                             )

    input_entity_0 = Input(shape=(1,), dtype='int32', name='input_entity_0')
    entity_fea_0 = wordembedding(input_entity_0)
    input_entity_1 = Input(shape=(1,), dtype='int32', name='input_entity_1')
    entity_fea_1 = wordembedding(input_entity_1)

    input_word_0 = Input(shape=(word_length[0],), dtype='int32', name='input_word_0')
    word_fea_0 = wordembedding(input_word_0)  # trainable=False

    input_pos_0 = Input(shape=(word_length[0],), dtype='int32', name='input_pos_0')
    pos_fea_0 = posembedding(input_pos_0)

    input_dis1_0 = Input(shape=(word_length[0],), dtype='int32', name='input_dis1_0')
    dis_fea1_0 = disembedding(input_dis1_0)

    input_dis2_0 = Input(shape=(word_length[0],), dtype='int32', name='input_dis2_0')
    dis_fea2_0 = disembedding(input_dis2_0)

    input_word_1 = Input(shape=(word_length[1],), dtype='int32', name='input_word_1')
    word_fea_1 = wordembedding(input_word_1)

    input_pos_1 = Input(shape=(word_length[1],), dtype='int32', name='input_pos_1')
    pos_fea_1 = posembedding(input_pos_1)

    input_dis1_1 = Input(shape=(word_length[1],), dtype='int32', name='input_dis1_1')
    dis_fea1_1 = disembedding(input_dis1_1)

    input_dis2_1 = Input(shape=(word_length[1],), dtype='int32', name='input_dis2_1')
    dis_fea2_1 = disembedding(input_dis2_1)

    input_word_2 = Input(shape=(word_length[2],), dtype='int32', name='input_word_2')
    word_fea_2 = wordembedding(input_word_2)

    input_pos_2 = Input(shape=(word_length[2],), dtype='int32', name='input_pos_2')
    pos_fea_2 = posembedding(input_pos_2)

    input_dis1_2 = Input(shape=(word_length[2],), dtype='int32', name='input_dis1_2')
    dis_fea1_2 = disembedding(input_dis1_2)

    input_dis2_2 = Input(shape=(word_length[2],), dtype='int32', name='input_dis2_2')
    dis_fea2_2 = disembedding(input_dis2_2)

    emb_merge_0 = concatenate(inputs=[word_fea_0, pos_fea_0, dis_fea1_0, dis_fea2_0])

    emb_merge_1 = concatenate(inputs=[word_fea_1, pos_fea_1, dis_fea1_1, dis_fea2_1])
    emb_merge_2 = concatenate(inputs=[word_fea_2, pos_fea_2, dis_fea1_2, dis_fea2_2])

    input_shortest_word = Input(shape=(s['shortest_part_length'],), dtype='int32', name='input_shortest_word')
    shortest_word_fea = wordembedding(input_shortest_word)

    input_shortest_pos = Input(shape=(s['shortest_part_length'],), dtype='int32', name='input_shortest_pos')

    shortest_pos_fea = posembedding(input_shortest_pos)

    shortest_input_dis1 = Input(shape=(s['shortest_part_length'],), dtype='int32', name='shortest_input_dis1')
    shortest_dis_fea1 = disembedding(shortest_input_dis1)

    shortest_input_dis2 = Input(shape=(s['shortest_part_length'],), dtype='int32', name='shortest_input_dis2')
    shortest_dis_fea2 = disembedding(shortest_input_dis2)

    # emb_merge_shortest = Merge(mode='concat')([shortest_word_fea, shortest_pos_fea,shortest_dis_fea1,shortest_dis_fea2])
    emb_merge_shortest = concatenate(inputs=
                                     [shortest_word_fea, shortest_pos_fea, shortest_dis_fea1, shortest_dis_fea2])


    # dropout layer without att
    emb_merge_0 = Dropout(rate=s['emb_dropout'])(emb_merge_0)
    emb_merge_1 = Dropout(rate=s['emb_dropout'])(emb_merge_1)
    emb_merge_2 = Dropout(rate=s['emb_dropout'])(emb_merge_2)
    emb_merge_shortest = Dropout(rate=s['emb_dropout'])(emb_merge_shortest)

    # bottom RNNs
    left_gru_0 = GRU(units=s['first_hidden_layer'],
                      recurrent_initializer='orthogonal',
                      # recurrent_initializer='xavier',
                      activation='tanh',
                      return_sequences=True,
                      recurrent_activation='sigmoid')(emb_merge_0)

    right_gru_0 = GRU(units=s['first_hidden_layer'],
                       recurrent_initializer='orthogonal',
                       # recurrent_initializer='xavier',
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       go_backwards=True,
                      return_sequences=True)(emb_merge_0)

    # left_gru_0 = emb_AttentionLayer(8,16)(left_gru_0)
    # right_gru_0 = emb_AttentionLayer(8,16)(right_gru_0)
    gru_merge_0 = concatenate([left_gru_0, right_gru_0])

    print('gru',gru_merge_0)
    left_gru_1 = GRU(units=s['first_hidden_layer'],
                      recurrent_initializer='orthogonal',
                      activation='tanh',
                      recurrent_activation='sigmoid',
                      return_sequences=True)(emb_merge_1)

    right_gru_1 = GRU(units=s['first_hidden_layer'],
                       recurrent_initializer='orthogonal',
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       go_backwards=True,
                      return_sequences=True)(emb_merge_1)
    # left_gru_1 = emb_AttentionLayer(8,16)(left_gru_1)
    # right_gru_1 = emb_AttentionLayer(8,16)(right_gru_1)
    gru_merge_1 = concatenate(inputs=[left_gru_1, right_gru_1])

    left_gru_2 = GRU(units=s['first_hidden_layer'],
                      recurrent_initializer='orthogonal',
                      activation='tanh',
                      recurrent_activation='sigmoid',
                      return_sequences=True)(emb_merge_2)

    right_gru_2 = GRU(units=s['first_hidden_layer'],
                       recurrent_initializer='orthogonal',
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       go_backwards=True,
                      return_sequences=True)(emb_merge_2)
    # left_gru_2 = emb_AttentionLayer(8,16)(left_gru_2)
    # right_gru_2 = emb_AttentionLayer(8,16)(right_gru_2)
    gru_merge_2 = concatenate(inputs=[left_gru_2, right_gru_2])

    left_gru_shortest = GRU(units=s['first_hidden_layer'],
                             recurrent_initializer='orthogonal',
                             activation='tanh',
                             recurrent_activation='sigmoid',
                      return_sequences=True)(emb_merge_shortest)

    right_gru_shortest = GRU(units=s['first_hidden_layer'],
                              recurrent_initializer='orthogonal',
                              activation='tanh',
                              recurrent_activation='sigmoid',
                              go_backwards=True,
                      return_sequences=True)(emb_merge_shortest)
    # left_gru_shortest = emb_AttentionLayer(8,16)(left_gru_shortest)
    # right_gru_shortest = emb_AttentionLayer(8,16)(right_gru_shortest)
    gru_merge_shortest = concatenate(inputs=[left_gru_shortest, right_gru_shortest])

    gru_merge_temp = concatenate(
        inputs=[gru_merge_0, gru_merge_1,gru_merge_2])

    # gru_merge_temp = Dropout(0.7)(gru_merge_temp)
    # gru_merge_shortest = Dropout(0.7)(gru_merge_shortest)

    #TOP CNN
    convd_0_0 = Conv1D(filters=64,kernel_size=3,activation="relu")(gru_merge_temp)
    maxp_0_0 = MaxPooling1D(2)(convd_0_0)
    convd_0_1 = Conv1D(32, 3,activation="relu")(convd_0_0)
    maxp_0_1 = MaxPooling1D(2)(convd_0_1)

    convd_0_0_shortest = Conv1D(64, 3,activation="relu")(gru_merge_shortest)
    maxp_0_0_shortest = MaxPooling1D(2)(convd_0_0_shortest)
    convd_0_1_shortest = Conv1D(32, 3,activation="relu")(maxp_0_0_shortest)
    maxp_0_1_shortest = MaxPooling1D(2)(convd_0_1_shortest)

    cnn_flatten = Flatten()(maxp_0_1)
    cnn_flatten_shortest = Flatten()(maxp_0_1_shortest)

    cnn_merge = concatenate(inputs=[cnn_flatten, cnn_flatten_shortest])
    dense_drop_0 = Dropout(s['dense_dropout'])(cnn_merge)

    #多分类
    final_output_2 = Dense(nb_class)(dense_drop_0)
    final_output = Activation('softmax',name="multi_class")(final_output_2)


    model = Model(input=[input_word_0, input_word_1, input_word_2, input_pos_0, input_pos_1, input_pos_2, input_dis1_0,
                         input_dis1_1, input_dis1_2,
                         input_dis2_0, input_dis2_1, input_dis2_2, input_shortest_word, shortest_input_dis1,
                         shortest_input_dis2,
                         input_shortest_pos, input_entity_0, input_entity_1], outputs=[final_output])



    # opt = RMSprop(lr=s['lr'])
    # opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    opt = optimizers.Adam(lr=0.0012, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt = optimizers.Adamax(lr=0.02, epsilon=1e-06)

    model.compile(loss={'multi_class':'categorical_crossentropy'},
                  optimizer=opt,
                  # metrics=['accuracy'],
                  metrics=[fmeasure]
                  # weighted_metrics={'accuracy':1,'fmeasure':0.5}
                  )

    print('\n')

    # model.summary()

    # model.load_weights("./model_weights.h5")

    # inds = list(range(train_word_list[0].shape[0]))
    # np.random.shuffle(inds)

    # batch_num = len(inds) // s['mini_batch']

    totalloss = 0
    # print len(inds),s['mini_batch'],batch_num

    print('-----------------Begin of training-------------------')
    all_precision=[]
    all_recall=[]
    all_F1 =[]
    all_loss=[]
    minloss = 200
    bestepoch = 0
    bestFscore = 0
    import datetime

    #划分训练集合和验证集合
    from sklearn.model_selection import train_test_split
    train_word_list1, dev_word_list1,\
    train_word_list2, dev_word_list2,\
    train_word_list3, dev_word_list3,\
    train_POS_list1,dev_POS_list1,\
    train_POS_list2,dev_POS_list2,\
    train_POS_list3,dev_POS_list3,\
    train_distances_list1,dev_distances_list1, \
    train_distances_list2,dev_distances_list2, \
    train_distances_list3,dev_distances_list3, \
    train_distances2_list1,dev_distances2_list1,\
    train_distances2_list2,dev_distances2_list2,\
    train_distances2_list3,dev_distances2_list3,\
    train_shortest_word,dev_shortest_word, \
    train_shortest_dis1,dev_shortest_dis1, \
    train_shortest_dis2,dev_shortest_dis2, \
    train_entity_0,dev_entity_0, \
    train_entity_1,dev_entity_1,\
    train_shortest_pos,dev_shortest_pos,\
    train_y_mulity, dev_y_mulity = train_test_split(train_word_list[0], train_word_list[1],train_word_list[2],train_POS_list[0],train_POS_list[1],train_POS_list[2],train_distances_list[0],train_distances_list[1],train_distances_list[2],train_distances2_list[0],train_distances2_list[1],train_distances2_list[2],train_shortest_word,train_shortest_dis1,train_shortest_dis2,train_entity_0,train_entity_1,train_shortest_pos,train_y_mulity, test_size=0.2, random_state=0)

    inds = list(range(train_word_list1.shape[0]))
    np.random.shuffle(inds)

    batch_num = len(inds) // s['mini_batch']

    for epoch in range(s['epochs']):

        loss = acc = 0

        print('learning epoch:' + str(epoch))
        # starttime = datetime.datetime.now()

        for minibatch in range(batch_num):

            val = model.train_on_batch([train_word_list1[inds[minibatch::batch_num]], \
                                        train_word_list2[inds[minibatch::batch_num]], \
                                        train_word_list3[inds[minibatch::batch_num]], \
                                        train_POS_list1[inds[minibatch::batch_num]], \
                                        train_POS_list2[inds[minibatch::batch_num]],
                                        train_POS_list3[inds[minibatch::batch_num]], \
                                        train_distances_list1[inds[minibatch::batch_num]], \
                                        train_distances_list2[inds[minibatch::batch_num]],
                                        train_distances_list3[inds[minibatch::batch_num]], \
                                        train_distances2_list1[inds[minibatch::batch_num]],
                                        train_distances2_list2[inds[minibatch::batch_num]],
                                        train_distances2_list3[inds[minibatch::batch_num]],
                                        train_shortest_word[inds[minibatch::batch_num]],
                                        train_shortest_dis1[inds[minibatch::batch_num]],
                                        train_shortest_dis2[inds[minibatch::batch_num]],
                                        train_shortest_pos[inds[minibatch::batch_num]],
                                        train_entity_0[inds[minibatch::batch_num]],
                                        train_entity_1[inds[minibatch::batch_num]]],
                                       train_y_mulity[inds[minibatch::batch_num]]
                                       )

            if minibatch % 20 == 0:
                # print ('=', end = '')
                print('=',)

            loss = loss + val[0]

        # training ended every  epoch
        totalloss = totalloss + loss
        print('<<    ', 'training loss:', str(np.round(loss, 5)))
        # sys.stdout.flush()

        # print ('------------------Begin of testing------------------')
        pred_dev = model.predict(
            [dev_word_list1, dev_word_list2, dev_word_list3, dev_POS_list1, dev_POS_list2,
             dev_POS_list3,
             dev_distances_list1, dev_distances_list2, dev_distances_list3, dev_distances2_list1,
             dev_distances2_list2, dev_distances2_list3, dev_shortest_word, dev_shortest_dis1,
             dev_shortest_dis2, dev_shortest_pos, dev_entity_0, dev_entity_1],
            batch_size=s['mini_batch'])  # ,test_POS
        precision, recall, F1 = result_evaluation(dev_y_mulity, pred_dev)
        # print'testing epochs:' + ' precision:' + str(np.round(precision, 5)) + ' recall:' + str(np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5))
        endtime = datetime.datetime.now()
        print('dev-->epochs:' + str(epoch + 1) + ' precision:' + str(np.round(precision, 5)) + ' recall:' + str(
            np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5)))
        all_loss.append(np.round(loss,3))
        all_F1.append(np.round(F1, 3))
        all_precision.append(np.round(precision,3))
        all_recall.append(np.round(recall,3))

        #判断loss
        if loss<minloss:
            minloss = loss
            bestepoch = epoch
        elif (epoch-bestepoch)>=8:
            break

        #F
        if F1>bestFscore:
            bestFscore = F1
            pred_test = model.predict(
                [test_word_list[0], test_word_list[1], test_word_list[2], test_POS_list[0], test_POS_list[1],
                 test_POS_list[2],
                 test_distances_list[0], test_distances_list[1], test_distances_list[2], test_distances2_list[0],
                 test_distances2_list[1], test_distances2_list[2], test_shortest_word, test_shortest_dis1,
                 test_shortest_dis2, test_shortest_pos, test_entity_0, test_entity_1],
                batch_size=s['mini_batch'])
            precision, recall, F1 = result_evaluation(answer_y, pred_test)
            print('testing epochs:' + str(epoch + 1) + ' precision:' + str(np.round(precision, 5)) + ' recall:' + str(
                np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5)) )


    print('-----------------End of DDI extraction----------------')
    print('best_epoch:',bestepoch,'|min_loss:',minloss,'|bestdevF1Score:',bestFscore)
    print(all_loss)
    print(all_F1)
    print(all_precision)
    print(all_recall)
    precision, recall, F1 = result_evaluation(answer_y, pred_test)

