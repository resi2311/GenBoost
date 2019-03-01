import numpy as np
import json
import copy
import pickle

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from scripts.problem import Problem
from scripts.genboost import GenBoost

GB_PARAMS = 'config.json'
WEIGHTS_SAVE = 'best_weights_cnn.bin'
RESULT_SAVE = 'result_cnn.json'

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    print('ch_1')
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

del x_train
del y_train

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

def eval_model(weights):
    weight_layer_one = np.array(weights[:3*3*1*32]).reshape(3,3,1,32)
    weight_layer_two = np.array(weights[288:288 + 32]).reshape(32,)
    weight_layer_three = np.array(weights[320:320 + 3*3*32*64]).reshape(3,3,32,64)
    weight_layer_four = np.array(weights[18752:18752 + 64]).reshape(64,)
    weight_layer_five = np.array(weights[18816:18816+9216*64]).reshape(9216,64)
    weight_layer_six = np.array(weights[608640:608640 + 64]).reshape(64,)
    weight_layer_seven = np.array(weights[608704:608704+64*10]).reshape(64,10)
    weight_layer_eight = np.array(weights[-10:]).reshape(10,)
    
    model.set_weights([
        weight_layer_one,
        weight_layer_two,
        weight_layer_three,
        weight_layer_four,
        weight_layer_five,
        weight_layer_six,
        weight_layer_seven,
        weight_layer_eight
    ])
    return np.array([-1.*model.evaluate(x_test, y_test,verbose=0 )[1]])

with open(GB_PARAMS) as json_data:
    params = json.load(json_data)

my_prob = Problem(fit_func=eval_model, dim=609354, lb=-1., rb=1.)
gb = GenBoost(problem=my_prob)

pop = gb.run(params)
result = copy.copy(params)
result['champion_f'] = pop.champion_f[0]
with open(RESULT_SAVE,'w', encoding="utf-8", newline='\r\n') as json_data:
    json.dump(result, json_data, indent = 4)
pickle.dump(pop.champion_x, open(WEIGHTS_SAVE,'wb'))
