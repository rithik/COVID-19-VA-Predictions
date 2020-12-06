import os
import heapq
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from itertools import chain
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import sklearn
import keras.backend as K

tf.compat.v1.enable_eager_execution()
# sess = tf.InteractiveSession()
tf.random.set_seed(
    42
)


def soft_acc(y_true, y_pred):
    difference = tf.math.subtract(y_true, y_pred)
    def equaler(t): return 1.0 if t < 1.0 else 0.0
    bools = K.map_fn(equaler, difference)
    val = K.mean(bools)
    return val


def create_mlp(args=None):
    # You can use args to pass parameter values to this method
    model = Sequential()
    # Define model architecture
    model.add(Dense(units=args['hidden_dim'],
                    activation='relu', input_dim=args['x_train_shape'][1]))  # 42
    for k in range(args['hidden_layer']):
        model.add(Dense(units=args['hidden_dim'], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    # Optimizer
    optimizer = keras.optimizers.SGD(
        lr=args['learning_rate']) if args['opt'] == 'sgd' else keras.optimizers.Adam(lr=args['learning_rate'])

    # Compile
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(x_train, y_train, args):
    model = create_mlp(args)
    history = model.fit(
        x_train, y_train, batch_size=args['batch_size'], epochs=args['epoch'], validation_split=0.1, shuffle=True)
    return model, history


def train_and_select_model(x_train, y_train):
    args = {
        # 'batch_size': 16,
        # 'epoch': 5,
        # 'learning_rate': 0.0001,  # 0.00001 20
        # 'hidden_dim': 200,
        # 'hidden_layer': 20,
        # 'opt': 'adam', 
        # 'x_train_shape': x_train.shape
        'batch_size': 16,
        'epoch': 5,
        'learning_rate': 0.005,  # 0.00001 20
        'hidden_dim': 150,
        'hidden_layer': 4,
        'opt': 'adam', 
        'x_train_shape': x_train.shape
    }
    best_acc = 0
    best_model = None
    best_history = None

    # for lr in [0.001, 0.0005, 0.0001]:
    #     for hd in [50, 100, 200]:
    #         for hl in [10, 20, 30]:
    #             args['learning_rate'] = lr
    #             args['hidden_dim'] = hd
    #             args['hidden_layer'] = hl
    model, history = train_model(x_train, y_train, args=args)
    if history.history['val_accuracy'][-1] >= best_acc:
        best_acc = history.history['val_accuracy'][-1]
        best_model = model
        best_history = history
    print(history.history['val_accuracy'])
    return best_model, best_history


def processData(path):
    df = pd.read_csv(path).sort_values(by=['report-date'])
    # x = df[['fips', 'unemployment-percent', 'median-household-income',
    #             'age0-14', 'age15-24', 'age25-34', 'age35-44', 'age45-54', 'age55-64', 'age65-85', 'white', 'black',
    #             'american-indian', 'asian', 'total', 'c2', 'h2', 'd2', 'c3', 'h3', 'd3', 'c4', 'h4', 'd4', 'c5', 'h5',
    #             'd5', 'c6', 'h6', 'd6', 'c7', 'h7', 'd7', 'c8', 'h8', 'd8', 'c9', 'h9', 'd9', 'c10', 'h10', 'd10']].to_numpy().astype('float32')
    x = df[['fips', 'unemployment-percent', 'median-household-income', 'num-days', 
            'age0-14', 'age15-24', 'age25-34', 'age35-44', 'age45-54', 'age55-64', 'age65-85', 'white', 'black',
            'american-indian', 'asian', 'total', 'c2', 'c3',  'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']].to_numpy().astype('float32')
    y = df[['greaterC']].to_numpy().astype('int32')
    # print(y.shape)
    print(x)
    x, y = sklearn.utils.shuffle(x, y)
    train_test_split = int(x.shape[0] * 0.9)
    x_train = x[:train_test_split]
    y_train = y[:train_test_split]
    x_test = x[train_test_split:]
    y_test = y[train_test_split:]
    # print(x_test.shape)
    # for k in range(0, len(x_test)):
    #     if x_test[k][0] == 51059:
    #         print(k)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    path = 'datasetbinary.csv'
    x_train, x_test, y_train, y_test = processData(path)
    model, history = train_and_select_model(x_train, y_train)
    y_predict = model.predict(x_test)
    print(y_predict)
    np.savetxt('output.txt', y_predict)
    print("-------------------------------------------------")
    print(y_test)
    np.savetxt('true.txt', y_test)
    loss_test, acc_test = model.evaluate(x_test, y_test)
    print(loss_test, acc_test)
    df = pd.DataFrame({"test" : y_test.flatten(), "pred" : y_predict.flatten()})
    df.to_csv("submission1.csv")
    # print(model.summary())
