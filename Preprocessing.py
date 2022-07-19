#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import sklearn
import numpy as np
from nltk.corpus import stopwords
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from sklearn.utils import shuffle
import sys


# this function help remove stop words
def prepare_dataset(df, df_t, target):

    sentences = df['text'].values
    y = df['label'].values

    sentences_t = df_t['text'].values
    y_t = df_t['label'].values

    """ uncomment this section if you would like to remove stopwords
    sentences_pre = df['text'].values
    sentences = []
    for line in sentences_pre:
        line = line.lower()
        text_tokens = word_tokenize(line)
        no_sw = [word for word in text_tokens if not word in STOPWORDS]
        filtered_sentence = (" ").join(no_sw)
        sentences.append(filtered_sentence)

    y = df['label'].values

    sentences_t_pre = df_t['text'].values
    sentences_t = []
    for line_t in sentences_t_pre:
        line_t = line_t.lower()
        text_tokens_t = word_tokenize(line_t)
        no_sw_t = [word for word in text_tokens_t if not word in STOPWORDS]
        filtered_sentence_t = (" ").join(no_sw_t)
        sentences_t.append(filtered_sentence_t)

    y_t = df_t['label'].values
    """

    print("####target - {} ####".format(target))
    print(Counter(y))

    # generate testing data
    sentences_test, y_test = shuffle(sentences, y, random_state=0)

    # generate training data
    sentences_train, y_train = shuffle(sentences_t, y_t,  random_state=0)

    return dict(
        train=sentences_train,
        test=sentences_test,
        y_train=y_train,
        y_test=y_test
    )


# this function tokenizes the sentences
def prepare_input_for_dl(sentences_train, sentences_test):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    print(vocab_size)
    print(sentences_train[2])
    print(X_train[2])

    X_train = pad_sequences(X_train, padding='post', maxlen=MAX_SEN_LENGTH)
    X_test = pad_sequences(X_test, padding='post', maxlen=MAX_SEN_LENGTH)
    print(X_train.shape)
    print(X_train[0, :])

    return X_train, X_test, vocab_size