import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential,Model
import keras
from keras import optimizers
from keras.layers import Dense,Conv1D,MaxPool1D,Flatten,BatchNormalization,Dropout
from sklearn import metrics
from keras.callbacks import EarlyStopping
from numpy import random
from tensorflow import set_random_seed
set_random_seed(6)
np.random.seed(6)
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def compute_accuracy(real_y,predict_y):
    length = len(real_y)
    correct = []
    for i in range(length):
        if real_y[i] == predict_y[i]:
            correct.append(1)
        else:
            correct.append(0)
    accuracy = np.mean(correct)
    return accuracy

def readData(filePath):
    
    data = pd.read_csv(filePath)
    data = data.values
    data_x = data[:,:-1]
    data_x = np.expand_dims(data_x, axis=2)
    data_y_1D = data[:,-1]
    data_y = np_utils.to_categorical(data_y_1D, 2)
    return data_x, data_y, data_y_1D


train_x, train_y, train_y_1D = readData('train_data_yongxin.csv')
test_x, test_y, test_y_1D = readData('test_data_yongxin.csv')



model = Sequential()

model.add(Conv1D(15, 3, activation='tanh', input_shape=(16, 1)))
# model.add(BatchNormalization())
model.add(MaxPool1D(2))
# model.add(BatchNormalization())
model.add(Flatten())
# model.add(BatchNormalization())
# model.add(Dense(30, activation='tanh'))
# model.add(Dropout(rate=0.05))
model.add(Dense(15, activation='tanh'))
model.add(Dense(2, activation='softmax'))
optimizer = optimizers.Adagrad(lr=0.006)
model.compile(loss='categorical_crossentropy',
             optimizer=optimizer,metrics=['accuracy'])
print(model.summary())
history = LossHistory()
model.fit(train_x,train_y,validation_data= (test_x,test_y),callbacks=[history],
          verbose=2, epochs=150)
# model.fit(train_x,train_y,validation_split=0.15,callbacks=[history],shuffle=True,verbose=2,epochs=500)
#
# with open('test_err.txt', 'w') as fp:
#     for loss in range(len(history.val_loss['epoch'])):
#         fp.write(str(loss+1)  + ',' + str(history.val_loss['epoch'][loss]) +  '\n')
# with open('train_err.txt', 'w') as fp:
#     for loss in range(len(history.losses['epoch'])):
#         fp.write(str(loss+1)  + ',' + str(history.losses['epoch'][loss]) +  '\n')

dense1_layer_model = Model(inputs = model.input, outputs = model.layers[-2].output)

y_pred_test = model.predict(test_x)
probability = [prob[1] for prob in y_pred_test]
pred_class = []
for i in probability:
    if i > 0.5:
        pred_class.append(1)
    else:
        pred_class.append(0)

accuracy = compute_accuracy(test_y_1D,pred_class)
print(accuracy)
test_auc = metrics.roc_auc_score(test_y_1D,probability)
print(test_auc)
model.save('my_model_yongxin2.h5')
dense1_layer_model.save('my_model_FE2.h5')
history.loss_plot('epoch')
