import keras

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten,Dense, Activation
from keras.datasets import mnist
from mnistprob import mnist_net
import numpy as np
import json
from genboost import genboost
import time
from ElephantSender import sendNotification
import sys

batch_size = 128
num_classes = 10
epochs = 5
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols,1)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

del x_train
del y_train

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

def eval_model(weights):
    # Rewrite this - 609,354
    weight_layer_one = np.array(weights[:3*3*1*32]).reshape(3,3,1,32)
    # 3*3*1*32 = 288
    weight_layer_two = np.array(weights[288:288 + 32]).reshape(32,)
    weight_layer_three = np.array(weights[320:320 + 3*3*32*64]).reshape(3,3,32,64)
    #320 + 3*3*32*64 = 18752
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

with open('parameters_3.json') as json_data:
    params = json.load(json_data)

MyProb = mnist_net(fit_func=eval_model,dim=609354,lb=-1.,rb=1.)
gb = genboost(problem=MyProb)

results = []
times = []
TotalTime_0 = time.time()

for i, param in enumerate(params):
    t0 = time.time()
    print("Star of test #{}.".format(i+1))
    pop = gb.run(param)
    results.append(pop.champion_f)
    print('Parameters: {}'.format(param))
    print('Fitness: {}'.format(pop.champion_f))
    t1 = time.time() - t0
    times.append(t1)
    print("Time for test:{}".format(t1 / 60))
    print('-'*20)
print("Total time for all tests:{}".format(time.time() - TotalTime_0))

def_stdout = sys.stdout
sys.stdout = open('log.txt','w')
for i, (param, fitness, TestTime) in enumerate(zip(params,results,times)):
    print("Test #{}".format(i+1))
    print('Parameters: {}'.format(param))
    print('Fitness: {}'.format(fitness))
    print("Time for test:{}".format(TestTime / 60.))
    print('-'*20)
sys.stdout = def_stdout

template = 'Test #{}\nParameters: {}\nFitness: {}\nTime: {}\n'
for i, (param, fitness, TestTime) in enumerate(zip(params,results,times)):
    sendNotification(template.format(i+1,param, fitness, TestTime/60.))
