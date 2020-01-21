import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.utils import np_utils
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from keras.models import load_model

np.random.seed(1)
def readData(filePath):
  
    data = pd.read_csv(filePath)
    data = data.values
    data_x = data[:,:-1]
    data_y_1D = data[:,-1]
    return data_x,data_y_1D

train_x, train_y_1D = readData('train_data_yongxin.csv')
test_x,  test_y_1D = readData('test_data_yongxin.csv')



def get_ROC(data_input_y,y_probability,save_path):
    fpr, tpr, thresholds = metrics.roc_curve(data_input_y, y_probability)
    fpr, tpr = fpr.tolist(), tpr.tolist()
    # print(fpr,type(fpr))
    with open(save_path, 'w') as fp:
        for num in range(len(fpr)):
            fp.write(str(fpr[num]) + ',' + str(tpr[num]) + '\n')


def AIC(y_test, y_pred, k, n):
    '''
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



# classifier = SVC(probability=True, C = 2**10, gamma = 2**-11)
classifier = RandomForestClassifier(oob_score=True, random_state=1)
# classifier = LogisticRegression()
# classifier = GaussianNB()
# classifier = DecisionTreeClassifier(random_state=0)
# classifier = BernoulliNB()
# classifier = AdaBoostClassifier(n_estimators=50)
classifier.fit(train_x, train_y_1D)


y_probability = classifier.predict_proba(test_x)
y_true = classifier.predict(test_x)
y_probability_first = [x[1] for x in y_probability]
print(y_probability_first)
test_auc = metrics.roc_auc_score(test_y_1D,y_probability_first)
acc = classifier.score(test_x,test_y_1D)
k_value = metrics.cohen_kappa_score(test_y_1D,y_true )
mcc = metrics.matthews_corrcoef(test_y_1D, y_true)
# aic_value = AIC(test_y_1D, y_true, k = 16, n = 218)
# get_ROC(test_y_1D, y_probability_first, 'RF_roc.txt')

print(acc)
print(test_auc)
print(k_value)
print(mcc)
# print(aic_value)