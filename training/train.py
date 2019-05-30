# -*- coding: utf-8 -*-
from itertools import combinations
from nltk import ngrams
# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


dataset_dictionary = None
top_word_pair_features = None
top_syntactic_grammar_list = None

trained_model_pickle_file = './trained_model.pkl'


def get_empty_vector(n):
    return [0 for _ in range(n)]


def get_top_word_dataset_dictionary():
    from feaure_extraction.feature_vector import get_dataset_dictionary

    global dataset_dictionary
    if dataset_dictionary is None:
        dataset_dictionary = get_dataset_dictionary()
    return dataset_dictionary


def get_top_word_pair_features():
    from feaure_extraction.feature_vector import extract_top_word_pair_features

    global top_word_pair_features
    if top_word_pair_features is None:
        top_word_pair_features = extract_top_word_pair_features()
    return top_word_pair_features


def get_top_syntactic_grammar_list():
    from feaure_extraction.feature_vector import extract_top_syntactic_grammar_trio

    global top_syntactic_grammar_list
    if top_syntactic_grammar_list is None:
        top_syntactic_grammar_list = extract_top_syntactic_grammar_trio()
    return top_syntactic_grammar_list


def get_word_feature(normalized_sentence):
    unique_tokens = set(word for word in normalized_sentence.split())
    # exclude duplicates in same line and sort to ensure one word is always before other
    bi_grams = set(ngrams(normalized_sentence.split(), 2))
    words = unique_tokens | bi_grams
    dataset_dictionary = get_top_word_dataset_dictionary()
    #print('dataset_dictionary',dataset_dictionary)
    X = [i if j in words else 0 for i, j in enumerate(dataset_dictionary)]
    return X


def get_frequent_word_pair_feature(normalized_sentence):
    unique_tokens = sorted(set(word for word in normalized_sentence.split()))
    # exclude duplicates in same line and sort to ensure one word is always before other
    combos = combinations(unique_tokens, 2)
    top_word_pair_features = get_top_word_pair_features()
    X = [i if j in combos else 0 for i, j in enumerate(top_word_pair_features)]
    return X


def get_syntactic_grammar_feature(sentence_text):
    from feaure_extraction.feature_vector import extract_syntactic_grammar
    trigrams_list = extract_syntactic_grammar(sentence_text)
    top_syntactic_grammar_list = get_top_syntactic_grammar_list()
    X = [i if j in trigrams_list else 0 for i, j in enumerate(top_syntactic_grammar_list)]
    return X


def make_feature_vector(row):
    normalized_sentence = row.normalized_sentence
    sentence = row.sentence_text

    word_feature = get_word_feature(normalized_sentence)
    frequent_word_feature = get_frequent_word_pair_feature(normalized_sentence)
    syntactic_grammar_feature = get_syntactic_grammar_feature(sentence)

    features = word_feature
    features.extend(frequent_word_feature)
    features.extend(syntactic_grammar_feature)
    return features


def main():
    from sklearn.metrics import classification_report
    from imblearn.under_sampling import RandomUnderSampler



def extract_training_data_from_dataframe(df):
    from dataset.read_dataset import get_training_label
    from feaure_extraction.feature_vector import dataset_vector
    X = dataset_vector(df)
    Y = df.apply(get_training_label, axis=1)
    X = np.array(X)
    Y = np.array(Y.tolist())
    return X, Y

def extract_test_data_from_dataframe(df):
    from dataset.read_dataset import get_training_label
    from feaure_extraction.feature_vector import dataset_test_vector
    X = dataset_test_vector(df)
    Y = df.apply(get_training_label, axis=1)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def train_data():
    from dataset.read_dataset import get_dataset_dataframe
    df = get_dataset_dataframe()
    X, Y = extract_training_data_from_dataframe(df)
    print('trainX: ', (X.shape), 'trainY : ', (Y.shape))
    return X,Y

def test_data():
    import os
    from dataset.read_dataset import get_dataset_dataframe
    dfTest = get_dataset_dataframe(directory=os.path.expanduser(
        'D:/ay/detection-SVM-word2vec/dataset/DDICorpus/Test/test_for_ddi_extraction_task/DrugBank/'))
    X, Y = extract_test_data_from_dataframe(dfTest)
    print('testX: ', (X.shape), 'testY : ', (Y.shape))
    return X, Y
