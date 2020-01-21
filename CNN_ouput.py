import numpy as np
from keras.models import Sequential, load_model
import pandas as pd
from keras.utils import np_utils
import evaluate_method

def eval_model(model, test_data_x, test_y_1D):
    y_pred_test = model.predict(test_data_x)
    probability = [prob[1] for prob in y_pred_test]
    print(len(probability))
    evaluate_method.get_ROC(test_y_1D, probability, save_path='roc_rnn_test.txt')
    acc = evaluate_method.get_acc(test_y_1D, probability)  # AUC value
    auc = evaluate_method.get_auc(test_y_1D, probability)  # ACC value
    kappa = evaluate_method.get_kappa(test_y_1D, probability)
    IOA = evaluate_method.get_IOA(test_y_1D, probability)
    mcc = evaluate_method.get_mcc(test_y_1D,probability)
    recall = evaluate_method.get_recall(test_y_1D, probability)
    precision = evaluate_method.get_precision(test_y_1D, probability)
    f1 = evaluate_method.get_f1(test_y_1D, probability)
    print("ACC = " + str(acc)+" AUC = " + str(auc)+ ' kappa = '+ str(kappa) +
          ' IOA = ' + str(IOA) + ' MCC = ' + str(mcc))
    print("precision = " + str(precision)+" recall = " + str(recall)+ ' f1 = '+ str(f1))
    # print("AUC = " + str(auc))
    # print(kappa)

def readData(filePath):
    #训练数据的读入
    data = pd.read_csv(filePath)
    data = data.values
    data_x = data[:,:-1]
    data_x = np.expand_dims(data_x, axis=2)
    data_y_1D = data[:,-1]
    data_y = np_utils.to_categorical(data_y_1D, 2)
    return data_x, data_y, data_y_1D
train_x, train_y, train_y_1D = readData('train_data_yongxin.csv')
test_x, test_y, test_y_1D = readData('test_data_yongxin.csv')
#
# test_x = train_x
# test_y_1D = train_y_1D

model_cnn = load_model('my_model_yongxin1.h5')
# model_rnn = load_model('my_model_RNN1.h5')
# model_rnn = load_model('my_model_yanshan_rnn.h5')
# model_rnn_aug = load_model('my_model_yanshan_rnn_aug.h5')
# print(str(model_cnn))


# 评估模型
# eval_model(model_cnn, test_data_x, test_y_1D)
# eval_model(model_cnn_aug, test_data_x, test_y_1D)
# eval_model(model_rnn, test_data_x_rnn, test_y_1D)
eval_model(model_cnn, test_x, test_y_1D)