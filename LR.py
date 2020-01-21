#coding=utf-8
#__future__
import numpy as np
import pandas as pd
from sklearn import metrics
from common_func import evaluate_method
from sklearn.metrics import confusion_matrix
from sklearn import svm
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

MIN_NUM = 1
MAX_NUM = [6,9,5,6,12,5,3,3,5,4,4,7,7,5,5,5]

def readData(train_file_path,test_file_path):
    train_data = pd.read_csv(train_file_path)
    train_data = train_data.values
    test_data = pd.read_csv(test_file_path)
    test_data = test_data.values

    train_data_x = train_data[:,:-1]
    train_data_y = train_data[:,-1]
    test_data_x = test_data[:,:-1]
    test_data_y = test_data[:,-1]
    return train_data_x,train_data_y,test_data_x,test_data_y


def readTotalData(total_data_file,num_factors):
    def transform_x(data):
        data_x = []
        for row in data:
            data_i = []
            for i in row:
                data_i.append(int(i))
            data_x.append(data_i)
        return data_x

    data = csv.reader(open(total_data_file,'r'))
    totaldata_x = []
    for row in data:
        totaldata_x.append(row[:num_factors])
    totaldata_x = totaldata_x[1:]
    totaldata_x = np.array(transform_x(totaldata_x))
    return totaldata_x

# def mean_data(data):
#     data = data.tolist()
#     for i in range(len(data)):
#         for j in range(len(data[i])):
#             data[i][j] = (data[i][j]-MIN_NUM)/(MAX_NUM[j]-MIN_NUM)
#     return data
def get_ROC(data_input_y,y_probability,save_path):
    fpr, tpr, thresholds = metrics.roc_curve(data_input_y, y_probability)
    fpr, tpr = fpr.tolist(), tpr.tolist()
    # print(fpr,type(fpr))
    with open(save_path, 'w') as fp:
        for num in range(len(fpr)):
            fp.write(str(fpr[num]) + ',' + str(tpr[num]) + '\n')


# x_train,x_test,y_train,y_test = train_test_split(data_x, data_y, train_size=0.8)


# model = svm.SVC (probability=True)
# model = RandomForestClassifier(oob_score=True, random_state=1)
model = LogisticRegression()
# model = AdaBoostClassifier(n_estimators=50)


train_data_x,train_data_y,test_data_x,test_data_y = readData(train_file_path='train_data_yongxin.csv',
                                                             test_file_path='test_data_yongxin.csv'
                                                             )
# test_data_x,test_data_y = train_data_x,train_data_y
# train_data_x = mean_data(train_data_x)
# train_data_x = np.array(train_data_x)


model.fit(train_data_x,train_data_y)

# test_data_x =train_data_x
# test_data_y = train_data_y
# y_pred = model.predict(data_input_x)                            
accuracy = model.score(test_data_x,test_data_y)     
y_probability = model.predict_proba(test_data_x)               
y_probability_first = [x[1] for x in y_probability]
print(y_probability_first)
test_auc = metrics.roc_auc_score(test_data_y,y_probability_first)   
kappa = evaluate_method.get_kappa(test_data_y, y_probability_first)
mcc = evaluate_method.get_mcc(test_data_y, y_probability_first)
# get_ROC(test_data_y, y_probability_first, 'SVM_roc.txt')

print ('accuracy = %f' %accuracy)
print ('AUC = %f'%test_auc)
print(kappa)
print(mcc)

# print (confusion_matrix(data_input_y,y_pred))


# total_data_x = readTotalData(total_data_file='total_data_yushan0.csv',num_factors=16)
# # total_data_x = mean_data(total_data_x)
# # total_data_x = np.array(total_data_x)
# y_probability_total = model.predict_proba(total_data_x)              
# y_probability_total_first = [x[1] for x in y_probability_total]
# with open('result_svm0.txt','w') as file:
#     for i in y_probability_total_first:
#         file.write(str(i)+'\n')

print('Bingo')

