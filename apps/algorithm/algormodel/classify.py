#-*-coding:utf-8-*-

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import pymysql
from sklearn.externals import joblib

svm = joblib.load(BASE_DIR + '/classify_SVM.m')

def classify_train():
    pass

def classify_test(test_vec):
    test_pred = self.svm.predict(test_vec)
    return test_pred

