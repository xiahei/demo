#-*-coding:utf-8-*-

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import pymysql
from sklearn.feature_extraction.text import TfidfVectorizer

'''
fout = open('algormodel/test_cut','w')
with open("algormodel/train_cut","r") as fr:
    for line in fr:
        line = line.strip().split('\t')
        line[1] = line[1].strip().split()
        line[2] = line[2].strip().split()
        fout.write("|".join(line[1])+'\t'+"|".join(line[2])+'\n')
fout.close()

# insert preprocess data into table
conn = pymysql.connect(user='root',password='123456',database='EnrollmentQA')
cur = conn.cursor()

with open("algormodel/test_cut","r") as fr:
    for line in fr:
        line = line.strip().split('\t')
        cur.execute("insert into algorithm_preprocessdata (question,answer) values (%s,%s)",(line[0],line[1]))
        conn.commit()

cur.close()
conn.close()
'''

def cluseter():
    with open(BASE_DIR + '/algormodel/tfidf_features_20181112.m','r') as fr:
        pass
    # turn sentence to vector
    tfidf = joblib.load(BASE_DIR + '/algormodel/tfidf_features_20181112.m')
    # k-means cluster
    pass


