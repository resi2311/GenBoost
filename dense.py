import tensorflow as tf
import numpy as np
import json
import copy
import pickle

from scripts.problem import Problem
from scripts.genboost import GenBoost

GB_PARAMS = 'config.json'
WEIGHTS_SAVE = 'best_weights_dense.bin'
RESULT_SAVE = 'result_dense.json'

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
    return np.array([-1.*model.evaluate(x_test, y_test,verbose=0 )[1]])

with open(GB_PARAMS) as json_data:
    params = json.load(json_data)

my_prob = Problem(fit_func=eval_model, dim=407050, lb=-1., rb=1.)
gb = GenBoost(problem=my_prob)

pop = gb.run(params)
result = copy.copy(params)
result['champion_f'] = pop.champion_f[0]
with open(RESULT_SAVE,'w', encoding="utf-8", newline='\r\n') as json_data:
    json.dump(result, json_data, indent = 4)
pickle.dump(pop.champion_x, open(WEIGHTS_SAVE,'wb'))