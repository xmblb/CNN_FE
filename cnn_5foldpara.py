#!/usr/bin/python
# # -*- coding=utf-8 -*-
import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Activation,Dropout,Dense,LSTM,Conv1D,MaxPool1D,Flatten
import loss_history,evaluate_method,read_data
from keras import optimizers
from sklearn.model_selection import KFold
from tensorflow import set_random_seed
set_random_seed(6)
#read train data
np.random.seed(6)

from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

train_x, train_y_1D = read_data.read_data('train_data_yongxin.csv')
test_x, test_y_1D = read_data.read_data('test_data_yongxin.csv')
train_y = np_utils.to_categorical(train_y_1D, 2)
test_y = np_utils.to_categorical(test_y_1D, 2)

train_x = np.expand_dims(train_x,axis=2)
test_x = np.expand_dims(test_x,axis=2)
kfold = KFold(n_splits=5, shuffle=True, random_state=6)
result_auc = []
layer_number = [1]
for num in layer_number:
    cvscores = []
    train_loss = []
    validation_loss = []
    for train, test in kfold.split(train_x, train_y_1D):
        # create model
        model = Sequential()
        # model.add(BatchNormalization(batch_input_shape=(None, 16, 1)))
        model.add(Conv1D(15, 3, activation='tanh', input_shape=(16, 1)))
        # model.add(BatchNormalization())
        model.add(MaxPool1D(2))
        model.add(Flatten())
        model.add(Dense(15, activation='tanh'))
        # model.add(Dropout(rate=0.3))
        model.add(Dense(2, activation='softmax'))
        optimizer = optimizers.Adagrad(0.006)
        model.compile(loss = root_mean_squared_error, optimizer=optimizer, metrics=['accuracy'])
        # Fit the model
        history = loss_history.LossHistory()
        model.fit(train_x[train], np_utils.to_categorical(train_y_1D[train], 2),
                  validation_data=(train_x[test], np_utils.to_categorical(train_y_1D[test])),shuffle=True, callbacks=[history], epochs=150, verbose=2)
        # evaluate the model
        y_prob_test = model.predict(train_x[test])     #output predict probability
        probability = [prob[1] for prob in y_prob_test]
        auc = evaluate_method.get_auc(train_y_1D[test],probability)    # ACC value
        print("AUC: ", auc)
        cvscores.append(auc)
        train_loss.append(history.losses['epoch'])
        validation_loss.append(history.val_loss['epoch'])
    print((np.mean(cvscores), np.std(cvscores)))
    result_auc.append(np.mean(cvscores))
print(result_auc)

# with open('train_err.txt', 'w') as fp:
#     for loss in range(len(train_loss[0])):
#         fp.write(str(loss+1)  + ',' + str(np.mean(train_loss, axis=0)[loss]) +  '\n')
# with open('valida_err.txt', 'w') as fp:
#     for loss in range(len(validation_loss[0])):
#         fp.write(str(loss+1)  + ',' + str(np.mean(validation_loss, axis=0)[loss]) +  '\n')
#
# leagth = range(len(train_loss[0]))
# plt.plot(leagth, np.mean(train_loss, axis=0), 'r', label='train loss')
# plt.plot(leagth, np.mean(validation_loss, axis=0), 'g', label='validation loss')
# plt.show()



