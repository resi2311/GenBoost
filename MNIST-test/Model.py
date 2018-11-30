import tensorflow as tf
import numpy as np
import json
from mnistprob import mnist_net
from genboost import genboost
import time
from ElephantSender import sendNotification

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_test = x_test[:1000]
y_test = y_test[:1000]

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

with open('parameters_2.json') as json_data:
    params = json.load(json_data)

MyProb = mnist_net(fit_func=eval_model,dim=407050,lb=-1.,rb=1.)
gb = genboost(problem=MyProb)

results = []
times = []
TotalTime_0 = time.time()

for param in params:
    t0 = time.time()
    print("Star of test.")
    pop = gb.run(param)
    results.append(pop.champion_f)
    print('Parameters: {}'.format(param))
    print('Fitness: {}'.format(pop.champion_f))
    t1 = time.time() - t0
    times.append(t1)
    print("Time for test:{}".format(t1))
    print('-'*20)
print("Total time for all tests:{}".format(time.time() - TotalTime_0))

for param, fitness,TestTime in zip(params,results, times):
    print('Parameters: {}'.format(param))
    print('Fitness: {}'.format(fitness))
    print("Time for test:{}".format(TestTime))
    print('-'*20)

for param, fitness, TestTime in zip(params,results,times):
    sendNotification('Parameters: {}'.format(param))
    sendNotification('Fitness: {}'.format(fitness))
    sendNotification("Time for test:{}".format(TestTime))
    sendNotification('-'*20)