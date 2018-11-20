#-*-coding:utf-8-*-

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import pymysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.cluster import KMeans


questions = []
answers = []

def cluster():

    conn = pymysql.connect(user='root',password='123456',database='EnrollmentQA')
    cur = conn.cursor()

    cur.execute("select question from algorithm_preprocessdata;")
    q = list(cur.fetchall())
    cur.execute("select answer from algorithm_preprocessdata;")
    a = list(cur.fetchall())

    conn.commit()

    for i in range(len(q)):
        questions.append(q[i][0])
        answers.append(a[i][0])


    # turn sentence to vector
    tfidf = joblib.load(BASE_DIR + '/tfidf_features_20181112.m')
    vectors = tfidf.transform(answers)

    # train k-means cluster
    kmeans = KMeans(n_clusters=5)
    s = kmeans.fit(vectors)
    K = kmeans.labels_
    #print (K)

    # turn origin data into 5 clusters
    clusters = {}
    for i in range(len(K)):
        if K[i] in clusters:
            clusters[K[i]].append(i)
        else:
            clusters[K[i]] = []
            clusters[K[i]].append(i)
    clusters = sorted(clusters.items(), key=lambda x: x[0])

    for k, v in clusters:
        for j in v:
            sql = "insert into algorithm_clusterres (label,question,answer) values (%s,%s,%s)"
            cur.execute(sql,(str(k),questions[j],answers[j]))
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    cluster()




