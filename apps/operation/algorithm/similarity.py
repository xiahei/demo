#-*-coding:utf-8-*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.externals import joblib
import jieba
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Sim:
    def __init__(self):
        self.label_dict = {0:'报考点',1:'招生人数',2:'学制',3:'考试内容',4:'报名',5:'费用',6:'特殊'}
        self.stop_words = self.load_stopwords()
        self.feature = joblib.load('F:/qademo/demo/apps/operation/algorithm/tfidf_features.m')
        self.classifier = joblib.load('F:/qademo/demo/apps/operation/algorithm/classify_baseline.m')    

    def load_stopwords(self,stopwords_file='F:/qademo/demo/apps/operation/algorithm/stopwords'):
        stopwords = set()
        with open(stopwords_file) as fr:
            for line in fr:
                line = line.strip().decode('utf-8') #turn string to utf-8
                stopwords.add(line)
        return stopwords

    def classify(self,sentence):
        temp = []
        #sentence = sentence.decode('utf-8')
        sentence = jieba.cut(sentence)
        for word in sentence:
            if word.decode('utf-8') in self.stop_words:
                continue
            temp.append(word)
        temp = [' '.join(temp)]
        vector = self.feature.transform(temp)
        pred_label = self.classifier.predict(vector)
        vector = vector.todense()
        return vector, pred_label[0]
    
    def search_range(self,pred_label):
        #print ("the predict label of input sentence is:",pred_label)
        filename = 'F:/qademo/demo/apps/operation/algorithm/data/'+str(pred_label)+'.npy'
        train_matrix = np.load(filename)
        #mapping index to question in train corpus
        train_index = {}
        count = 0
        #inputname = 'data/'+self.label_dict[int(pred_label)].decode('utf-8')
        inputname = 'F:/qademo/demo/apps/operation/algorithm/data/'+self.label_dict[int(pred_label)].decode('utf-8')
        with open(inputname,'r') as fr:
            for line in fr:
                line = line.strip().split('\t')
                train_index[count] = [line[1],line[2]]
                count += 1
        return train_matrix,train_index

    def cal_sim(self,train_matrix,test_matrix,train_index):
        '''
        cosine similarity = <a,b>/(||a||*||b||)
        '''
        #print ('the shape of train matrix:',train_matrix.shape)
        #print ('the shape of test matrix:',test_matrix.shape)
        num = np.dot(test_matrix,train_matrix.T)
        denom = np.linalg.norm(test_matrix) * np.linalg.norm(train_matrix,axis=1)
        #<a,b>==0
        if(np.sum(num)==0.0):
             sim_q = 'null'
             sim_a = 'null'
             return sim_q,sim_a
        cosine = num/denom
        sim = 0.5 + 0.5 * cosine
        #print ('the shape of sim:',sim.shape)
        #||a||*||b||==0
        where_are_nan = np.isnan(sim)
        where_are_inf = np.isinf(sim)
        sim[where_are_nan] = 0.0
        sim[where_are_inf] = 0.0
        #print ("sim:",sim)
        max_value = np.argmax(sim)
        #print ("max_value:",max_value)
        sim_q = train_index[max_value][0]
        sim_a = train_index[max_value][1]
        return sim_q,sim_a
        
'''
if __name__ == "__main__":
    #sentence = '报该校的MBA需要提前联系导师吗'
    while(1):
        print ("Input question sentence:")
        sentence = sys.stdin.readline()
        if not sentence.strip():
            print ("There is no input question,system will exit.")
            break
        res = Sim()
        vector,pred_label = res.classify(sentence)
        train_matrix,train_index = res.search_range(pred_label)
        q,a = res.cal_sim(train_matrix,vector,train_index)
        if q=='null' and a=='null':
            print ('Can not find answer to the question,we will offer manual service later.')
            continue
        print ("The most similar question is:")
        print (q)
        print ("the answer is:")
        print (a)
'''