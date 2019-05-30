# coding=utf-8
'''
Created on 2017.5.3

@author: DUTIRLAB
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
#
from keras import optimizers

from keras.preprocessing import sequence
from keras.engine.topology import Layer

from optparse import OptionParser

import pickle as pkl
import gzip
from keras import backend as K
# from keras import activations, initializations, regularizers, constraints
import numpy as np

np.random.seed(345)


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


# embedding entities attention
class emb_AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(emb_AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[-1],),
                                 initializer='uniform',
                                 trainable=True)
        super(emb_AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # # x = K.permute_dimensions(inputs, (0, 2, 1))
        # x = inputs
        # a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        # outputs = a * x
        # # outputs = K.sum(outputs, axis=1)
        # return outputs

        M = K.tanh(inputs)
        a = K.softmax(K.dot(M,self.W)+self.b)
        outputs = a*inputs
        return outputs


    def compute_output_shape(self, input_shape):
        return input_shape[0],input_shape[1], input_shape[2]






if __name__ == '__main__':

    s = {
        'emb_dimension': 100,  # dimension of word embedding
        'mini_batch': 128,
        'shortest_part_length': 12,
        'epochs': 150,
        'class_type': 5,
        'first_hidden_layer': 230,
        'lr': 0.001,
        'emb_dropout': 0.7,
        'dense_dropout': 0.5,
        'train_file': "D:/ay/detection-ML/PreprocessData/train.pkl.gz",
        'test_file': "D:/ay/detection-ML/PreprocessData/test.pkl.gz",
        'wordvecfile': "D:/ay/detection-ML/PreprocessData/vec.pkl.gz",

    }

    outpath = '../data/'
    nb_class = s['class_type']

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

    # print test_labels_vec

    answer_y = np.array(test_labels_vec, dtype=np.int8)
    train_y = np.array(train_labels_vec, dtype=np.int8)
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

    # emb_merge_0 = Merge(mode='concat')([word_fea_0, pos_fea_0, dis_fea1_0, dis_fea2_0])
    #
    # emb_merge_1 = Merge(mode='concat')([word_fea_1, pos_fea_1, dis_fea1_1, dis_fea2_1])
    # emb_merge_2 = Merge(mode='concat')([word_fea_2, pos_fea_2, dis_fea1_2, dis_fea2_2])

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

    # # attention layer
    # att_emb_merge_0 = emb_AttentionLayer()([word_fea_0, entity_fea_0, entity_fea_1, emb_merge_0])
    # att_emb_merge_1 = emb_AttentionLayer()([word_fea_1, entity_fea_0, entity_fea_1, emb_merge_1])
    # att_emb_merge_2 = emb_AttentionLayer()([word_fea_2, entity_fea_0, entity_fea_1, emb_merge_2])
    # att_emb_merge_shortest = emb_AttentionLayer()([shortest_word_fea, entity_fea_0, entity_fea_1, emb_merge_shortest])
    #
    #
    # # dropout layer
    # emb_merge_0 = Dropout(0.7)(att_emb_merge_0)
    # emb_merge_1 = Dropout(rate=s['emb_dropout'])(att_emb_merge_1)
    # emb_merge_2 = Dropout(rate=s['emb_dropout'])(att_emb_merge_2)
    # emb_merge_shortest = Dropout(rate=s['emb_dropout'])(att_emb_merge_shortest)


    # dropout layer without att
    emb_merge_0 = Dropout(rate=s['emb_dropout'])(emb_merge_0)
    emb_merge_1 = Dropout(rate=s['emb_dropout'])(emb_merge_1)
    emb_merge_2 = Dropout(rate=s['emb_dropout'])(emb_merge_2)
    emb_merge_shortest = Dropout(rate=s['emb_dropout'])(emb_merge_shortest)

    # bottom RNNs
    left_gru_0 = GRU(units=s['first_hidden_layer'],
                      recurrent_initializer='orthogonal',
                      activation='tanh',
                      return_sequences=True,
                      recurrent_activation='sigmoid')(emb_merge_0)

    right_gru_0 = GRU(units=s['first_hidden_layer'],
                       recurrent_initializer='orthogonal',
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       go_backwards=True,
                      return_sequences=True)(emb_merge_0)

    left_gru_0 = emb_AttentionLayer()(left_gru_0)
    right_gru_0 = emb_AttentionLayer()(right_gru_0)

    # gru_merge_0 = Merge(mode='concat')([left_gru_0, right_gru_0])
    gru_merge_0 = concatenate([left_gru_0, right_gru_0])
    # gru_merge_0 = emb_AttentionLayer()(gru_merge_0)

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
    left_gru_1 = emb_AttentionLayer()(left_gru_1)
    right_gru_1 = emb_AttentionLayer()(right_gru_1)
    # gru_merge_1 = Merge(mode='concat')([left_gru_1, right_gru_1])
    gru_merge_1 = concatenate(inputs=[left_gru_1, right_gru_1])
    # gru_merge_1 = emb_AttentionLayer()(gru_merge_1)

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
    left_gru_2 = emb_AttentionLayer()(left_gru_2)
    right_gru_2 = emb_AttentionLayer()(right_gru_2)
    # gru_merge_2 = Merge(mode='concat')([left_gru_2, right_gru_2])
    gru_merge_2 = concatenate(inputs=[left_gru_2, right_gru_2])
    # gru_merge_2 = emb_AttentionLayer()(gru_merge_2)

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
    left_gru_shortest = emb_AttentionLayer()(left_gru_shortest)
    right_gru_shortest = emb_AttentionLayer()(right_gru_shortest)
    # gru_merge_shortest = Merge(mode='concat')([left_gru_shortest, right_gru_shortest])
    gru_merge_shortest = concatenate(inputs=[left_gru_shortest, right_gru_shortest])
    # gru_merge_shortest = emb_AttentionLayer()(gru_merge_shortest)

    # gru_merge_0 = Reshape((60,2 * s['first_hidden_layer'],1))(gru_merge_0)
    # gru_merge_1 = Reshape((60,2 * s['first_hidden_layer'],1))(gru_merge_1)
    # gru_merge_2 = Reshape((60,2 * s['first_hidden_layer'],1))(gru_merge_2)
    # gru_merge_shortest = Reshape((12, 2 * s['first_hidden_layer'],1))(gru_merge_shortest)

    # gru_merge_temp = concatenate(
    #     inputs=[gru_merge_0, gru_merge_1,gru_merge_2,gru_merge_shortest])
    gru_merge_temp = concatenate(
        inputs=[gru_merge_0, gru_merge_1,gru_merge_2])

    # gru_merge_temp = Dropout(0.7)(gru_merge_temp)
    # gru_merge_shortest = Dropout(0.7)(gru_merge_shortest)

    print('gru_merge_temp',gru_merge_temp)
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


    # #bottom PCNN
    # convd_0_0 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), input_shape=[None, 6, 200, ])(gru_merge_0)
    # maxp_0_0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convd_0_0)
    # convd_0_1 = Conv2D(64, (5, 5))(maxp_0_0)
    # maxp_0_1 = MaxPooling2D(pool_size=(2, 2))(convd_0_1)
    # cnn_flatten0 = Flatten()(maxp_0_1)
    # # cnn_flatten0 = Flatten()(maxp_0_0)
    #
    # convd_1_0 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), input_shape=[None, 6, 200, ])(gru_merge_1)
    # maxp_1_0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convd_0_0)
    # convd_1_1 = Conv2D(64, (5, 5))(maxp_1_0)
    # maxp_1_1 = MaxPooling2D(pool_size=(2, 2))(convd_1_1)
    # cnn_flatten1 = Flatten()(maxp_1_1)
    # # cnn_flatten1 = Flatten()(maxp_1_0)
    #
    # convd_2_0 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), input_shape=[None, 6, 200, ])(gru_merge_2)
    # maxp_2_0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convd_2_0)
    # convd_2_1 = Conv2D(64, (5, 5))(maxp_2_0)
    # maxp_2_1 = MaxPooling2D(pool_size=(2, 2))(convd_2_1)
    # cnn_flatten2 = Flatten()(maxp_2_1)
    # # cnn_flatten2 = Flatten()(maxp_2_0)

    # cnn_flatten_shortest = Flatten()(maxp_shortest_0)



    dense_drop_0 = Dropout(s['dense_dropout'])(cnn_merge)

    # final_output_0 = Dense(nb_class)(dense_drop_0)
    # final_output_1 = Dense(nb_class)(drop)
    # final_output_1 = Activation('relu')(final_output_0)
    final_output_2 = Dense(nb_class)(dense_drop_0)
    final_output = Activation('softmax')(final_output_2)

    model = Model(input=[input_word_0, input_word_1, input_word_2, input_pos_0, input_pos_1, input_pos_2, input_dis1_0,
                         input_dis1_1, input_dis1_2,
                         input_dis2_0, input_dis2_1, input_dis2_2, input_shortest_word, shortest_input_dis1,
                         shortest_input_dis2,
                         input_shortest_pos, input_entity_0, input_entity_1], outputs=final_output)



    # opt = RMSprop(lr=s['lr'])
    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    # opt = optimizers.Adam(lr=0.0012, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt = optimizers.Adamax(lr=0.02, epsilon=1e-06)
    print(final_output)


    # 自定义评价函数
    def precision(y_true, y_pred):
        print('precision:', y_true.shape[1])
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
        print('Fmeasure', fbeta_score(y_true, y_pred, beta=1))
        return fbeta_score(y_true, y_pred, beta=1)

    model.compile(loss='categorical_crossentropy', optimizer=opt, \
                  metrics=[fmeasure])

    print('\n')

    model.summary()
    # model.load_weights("./model_weights.h5")

    inds = list(range(train_word_list[0].shape[0]))
    np.random.shuffle(inds)

    batch_num = len(inds) // s['mini_batch']

    totalloss = 0
    # print len(inds),s['mini_batch'],batch_num

    print('-----------------Begin of training-------------------')
    all_precision=[]
    all_recall=[]
    all_F1 =[]
    all_loss=[]
    minloss = 100
    bestepoch = 0
    bestFscore = 0
    import datetime


    for epoch in range(s['epochs']):

        loss = acc = 0

        print('learning epoch:' + str(epoch))

        starttime = datetime.datetime.now()

        for minibatch in range(batch_num):

            val = model.train_on_batch([train_word_list[0][inds[minibatch::batch_num]], \
                                        train_word_list[1][inds[minibatch::batch_num]], \
                                        train_word_list[2][inds[minibatch::batch_num]], \
                                        train_POS_list[0][inds[minibatch::batch_num]], \
                                        train_POS_list[1][inds[minibatch::batch_num]],
                                        train_POS_list[2][inds[minibatch::batch_num]], \
                                        train_distances_list[0][inds[minibatch::batch_num]], \
                                        train_distances_list[1][inds[minibatch::batch_num]],
                                        train_distances_list[2][inds[minibatch::batch_num]], \
                                        train_distances2_list[0][inds[minibatch::batch_num]],
                                        train_distances2_list[1][inds[minibatch::batch_num]],
                                        train_distances2_list[2][inds[minibatch::batch_num]],
                                        train_shortest_word[inds[minibatch::batch_num]],
                                        train_shortest_dis1[inds[minibatch::batch_num]],
                                        train_shortest_dis2[inds[minibatch::batch_num]],
                                        train_shortest_pos[inds[minibatch::batch_num]],
                                        train_entity_0[inds[minibatch::batch_num]],
                                        train_entity_1[inds[minibatch::batch_num]]],
                                       train_y[inds[minibatch::batch_num]])

            if minibatch % 20 == 0:
                # print ('=', end = '')
                print('=',)

            loss = loss + val[0]

        # training ended every  epoch
        totalloss = totalloss + loss
        print('<<    ', 'training loss:', str(np.round(loss, 5)))
        # sys.stdout.flush()

        # print ('------------------Begin of testing------------------')
        pred_test = model.predict(
            [test_word_list[0], test_word_list[1], test_word_list[2], test_POS_list[0], test_POS_list[1],
             test_POS_list[2],
             test_distances_list[0], test_distances_list[1], test_distances_list[2], test_distances2_list[0],
             test_distances2_list[1], test_distances2_list[2], test_shortest_word, test_shortest_dis1,
             test_shortest_dis2, test_shortest_pos, test_entity_0, test_entity_1],
            batch_size=s['mini_batch'])  # ,test_POS

        precision, recall, F1 = result_evaluation(answer_y, pred_test)
        # print'testing epochs:' + ' precision:' + str(np.round(precision, 5)) + ' recall:' + str(np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5))
        endtime = datetime.datetime.now()
        print('testing epochs:' + str(epoch+1) + ' precision:' + str(np.round(precision, 5)) + ' recall:' + str(
            np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5))+ ' Time:' + str(endtime-starttime))
        all_loss.append(np.round(loss,3))
        all_F1.append(np.round(F1, 3))
        all_precision.append(np.round(precision,3))
        all_recall.append(np.round(recall,3))

        #判断loss
        if loss<minloss:
            minloss = loss
            bestepoch = epoch
        elif (epoch-bestepoch)>=5:
            break

        #F
        if F1>bestFscore:
            bestFscore = F1


    print('-----------------End of DDI extraction----------------')
    print('best_epoch:',bestepoch,'|min_loss:',minloss,'bestF1Score:',bestFscore)
    print(all_loss)
    print(all_F1)
    print(all_precision)
    print(all_recall)
