import numpy as np
import json
import copy
import pickle

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, SpatialDropout1D
from keras.datasets import imdb

from scripts.problem import Problem
from scripts.genboost import GenBoost

GB_PARAMS = 'pso_best.json'
WEIGHTS_SAVE = 'best_weights_rnn.bin'
RESULT_SAVE = 'result_cnn.json'

# Устанавливаем seed для повторяемости результатов
np.random.seed(42)
# Максимальное количество слов (по частоте использования)
max_features = 5000
# Максимальная длина рецензии в словах
maxlen = 80

# Загружаем данные
datalen = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Заполняем или обрезаем рецензии
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

del X_train
del y_train

X_test = X_test[:datalen]
y_test = y_test[:datalen]

# Создаем сеть
model = Sequential()
# Слой для векторного представления слов
model.add(Embedding(max_features, 32))
model.add(SpatialDropout1D(0.2))
# Слой долго-краткосрочной памяти
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
# Полносвязный слой
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
# Функция приспособленности
def eval_model(weights):
    weight_layer_one = np.array(weights[:5000*32]).reshape(5000,32) 
    weight_layer_two = np.array(weights[5000*32:(5000*32 +32*400)]).reshape(32,400)
    weight_layer_three = np.array(weights[172800:172800+100*400]).reshape(100,400)
    weight_layer_four = np.array(weights[172800+100*400:172800+101*400]).reshape(400,)
    weight_layer_five = np.array(weights[213200:213200 + 100 * 1]).reshape(100,1)
    weight_layer_six = np.array(weights[-1]).reshape(1,)
    
    model.set_weights([
        weight_layer_one,
        weight_layer_two,
        weight_layer_three,
        weight_layer_four,
        weight_layer_five,
        weight_layer_six
    ])
    
    return np.array([-1.*model.evaluate(X_test, y_test, verbose=0 )[1]])

# Загрузка параметров для GenBoost
with open(GB_PARAMS) as json_data:
    params = json.load(json_data)

# Инициализация проблемы оптимизации
my_prob = Problem(fit_func=eval_model, dim=213200 + 100 * 1 + 1, lb=-1., rb=1.)
# Инициализация класса GenBoost
gb = GenBoost(problem=my_prob)

pop = gb.run(params)
result = copy.copy(params)
result['champion_f'] = pop.champion_f[0]
with open(RESULT_SAVE,'w', encoding="utf-8", newline='\r\n') as json_data:
    json.dump(result, json_data, indent = 4)
pickle.dump(pop.champion_x, open(WEIGHTS_SAVE,'wb'))

