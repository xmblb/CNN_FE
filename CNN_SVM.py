import numpy as np
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from keras.models import load_model


def readData(filePath):
    #训练数据的读入
    data = pd.read_csv(filePath)
    data = data.values
    data_x = data[:,:-1]
    data_x = np.expand_dims(data_x, axis=2)
    data_y_1D = data[:,-1]
    data_y = np_utils.to_categorical(data_y_1D, 2)
    return data_x, data_y, data_y_1D

def AIC(y_test, y_pred, k, n):
    '''赤池信息准则
    :param y_test:
    :param y_pred:
    :param k: number of features
    :param n: number of sample
    :return:
    '''
    resid = y_test - y_pred
    SSR = sum(resid ** 2)
    # AICValue = 2*k+n*np.log(float(SSR)/n)
    AICValue = k*np.log(n) + n*np.log(float(SSR)/n)
    return AICValue

gamma_set = []
gamma_min = 2**-15
for i in range(21):
    gamma_set.append(gamma_min)
    gamma_min *= 2

c_set = []
c_min = 2**-5
for c in range(21):
    c_set.append(c_min)
    c_min *= 2

def get_ROC(data_input_y,y_probability,save_path):
    fpr, tpr, thresholds = metrics.roc_curve(data_input_y, y_probability)
    fpr, tpr = fpr.tolist(), tpr.tolist()
    # print(fpr,type(fpr))
    with open(save_path, 'w') as fp:
        for num in range(len(fpr)):
            fp.write(str(fpr[num]) + ',' + str(tpr[num]) + '\n')

train_x, train_y, train_y_1D = readData('train_data_yongxin.csv')
test_x, test_y, test_y_1D = readData('test_data_yongxin.csv')


dense1_layer_model = load_model(filepath='my_model_FE1.h5')
train_features = dense1_layer_model.predict(train_x)
test_features = dense1_layer_model.predict(test_x)
print(train_features.shape)
# np.savetxt('svm_train_data.txt',train_features)
classifier = SVC(probability=True)
# classifier = RandomForestClassifier(oob_score=True, random_state=10)
# classifier = LogisticRegression()
# classifier = GaussianNB()
# classifier = BernoulliNB()
# classifier = AdaBoostClassifier(n_estimators=50)

# para_grid = {"C": c_set, "gamma": gamma_set}
# grid_search = GridSearchCV(classifier, para_grid,cv=5, n_jobs=1)
# grid_search.fit(train_features, train_y_1D)
# best_parameters = grid_search.best_estimator_.get_params()
# print(best_parameters['C'], best_parameters['gamma'])
# classifier = SVC(probability=True, C=best_parameters['C'], gamma=best_parameters['gamma'])
# # classifier.fit(train_features, train_y_1D)


classifier.fit(train_features, train_y_1D)


y_probability = classifier.predict_proba(test_features)               #得到分类概率值
y_probability_first = [x[1] for x in y_probability]
y_true = classifier.predict(test_features)
# print(y_probability)
test_auc = metrics.roc_auc_score(test_y_1D,y_probability_first)
acc = classifier.score(test_features,test_y_1D)
k_value = metrics.cohen_kappa_score(test_y_1D,y_true )
# aic_value = AIC(test_y_1D, y_true, k = 50, n = 218)
mcc = metrics.matthews_corrcoef(test_y_1D, y_true)
# get_ROC(test_y_1D, y_probability_first, 'SVMcnn_roc.txt')

print(acc)
print(test_auc)
print(k_value)
print(mcc)
# print(aic_value)
