import json
import time
import sys
import os
import pickle

import tensorflow as tf
import numpy as np

import problem
import genboost
import test

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

del mnist
del x_train
del y_train

model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(512, activation=tf.nn.relu),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

def eval_model(weights):
    weight_layer_one = np.array(weights[:784*512]).reshape(784,512)
    weight_layer_two = np.array(weights[784*512:785*512]).reshape(512,)
    weight_layer_three = np.array(weights[785*512:-10]).reshape(512,10)
    weight_layer_four = np.array(weights[-10:]).reshape(10,)

    model.set_weights([
        weight_layer_one,
        weight_layer_two,
        weight_layer_three,
        weight_layer_four
    ])
    return np.array([-1.*model.evaluate(x_test, y_test, verbose=0)[1]])

if __name__ == "__main__":
    tests_path = "tests"
    results_path = "results"
    test_file = "temp.json"
    results_file = "sga_results.json"

    with open(os.path.join(tests_path, test_file)) as json_data:
        params = json.load(json_data)

    my_prob = problem.Problem(fit_func=eval_model, dim=407050, lb=-1., rb=1.)
    gb = genboost.GenBoost(problem=my_prob)
    results = test.test(gb, params, fname = os.path.join(results_path, results_file))