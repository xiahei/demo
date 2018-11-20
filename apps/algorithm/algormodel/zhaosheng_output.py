#coding=utf-8
import pickle
import random
import numpy as np
import jieba
import chardet
from scipy.stats import rankdata
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.autograd as autograd #torch中自动计算梯度的模块
import torch.utils.data as data
import torch.nn as nn  #神经网络模块
import torch.nn.functional as F #神经网络模块中的常用功能
import torch.optim as optim  #模型优化器模块
import pickle
import pymysql
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.feature_extraction.text import TfidfVectorizer
vocabfile = BASE_DIR + '/test_cut'##################
trainfile = BASE_DIR + '/test_cut'
vectorsfile = BASE_DIR + '/100d.bin'
s_l_path= BASE_DIR + "/zhaosheng_real_100d_1112"

#训练时1pos/50neg，测试时也是读取train，训练和测试时都把针对这个问题的所有pos都加50个neg
conf = {
	'question_len':50,#20,
	'answer_len':30,#150,
	'batch_size':20,#256,
	'epochs':70,#10000,
	'embedding_dim':100,#512,
	'hidden_dim':100,#512,
	'learning_rate':0.001,
	'margin':0.02,
	'mode':'test',###############test
	'resume':1,#1
	'CNN_feature':100,
	'neg_sample':20,
	'select_sample':20###################785
}

candidate_list = {}
conn = pymysql.connect(user='root',password='123456',database='EnrollmentQA')
cur = conn.cursor()
cur.execute("select * from algorithm_clusterres;")
data = list(cur.fetchall())
for item in data:
	id,label,question,answer = item
	if label in candidate_list:
		candidate_list[label].append([question, answer])
	else:
		candidate_list[label] = []
conn.commit()
conn.close()
cur.close()

with open(BASE_DIR + "/tfidf_features_20181112.m","rb") as fr:
	tfidf = pickle.load(fr,encoding='bytes')
with open(BASE_DIR + "/classify_SVM.m","rb") as fr:
	svm = pickle.load(fr,encoding='bytes')

'''
def build_vocab():
	global vocabfile
	code, vocab = int(0), {}
	vocab['UNKNOWN'] = code   
	code += 1
	for line in open(vocabfile,'r', encoding='UTF-8'):
		items = line.strip().split('	')
		for i in range(0, 1): 
			for word in items[i].split(' '):
				if len(word) <= 0:
					continue
				if not word in vocab:   
					vocab[word] = code   
					code += 1
	return vocab
'''

### TODO: need to load vocab data from database
def build_vocab():
	global vocabfile
	code, vocab = 0, {}
	vocab['UNKNOWN'] = code
	code += 1
	with open(vocabfile, 'r', encoding='UTF-8') as fr:
		for line in fr:
			item1, item2 = line.strip().split('\t')
			item1 = item1.strip().split('|')
			item2 = item2.strip().split('|')
			for word in item1:
				if len(word.strip()) <= 0:
					continue
				if word not in vocab:
					vocab[word] = code
					code += 1
			for word in item2:
				if len(word.strip()) <= 0:
					continue
				if word not in vocab:
					vocab[word] = code
					code += 1
	return vocab


def load_vectors():
	global vectorsfile
	vectors = {}
	with open(vectorsfile,'r', encoding='UTF-8') as fr:
		for line in fr:
			items = line.strip().split()
			#print("length:",items[-1])
			if len(items[0]) <= 0:
				continue
			vec = []
			for i in range(1, conf['embedding_dim']+1):
				vec.append(float(items[i]))
			vectors[items[0]] = vec
	return vectors   
	
def load_word_embeddings(vocab, dim):   
	vectors = load_vectors()   
	embeddings = []   
	for i in range(0, len(vocab)):	
		vec = []
		for j in range(0, dim):
			vec.append(0.01)	 
		embeddings.append(vec)
	for word, code in vocab.items():
		if word in vectors:
			embeddings[code] = vectors[word]
	return np.array(embeddings, dtype='float32')

def encode_sent(vocab, string, size): #vocab:{'unknown':0,'word1':1,'word2':2...},sentence,100
	x = []
	words = string.split('|')
	for i in range(len(words)):
		try:
			if words[i] in vocab:
				x.append(vocab[words[i]])  
			else:	
				x.append(vocab['UNKNOWN'])
		except:
			print(string)
			print(i)
			print(words[i])
			print(len(vocab))
			print(type(vocab))
			print(vocab[words[i]])
	if(len(words)>=size):
		x=x[:size]
	else:
		for i in range(len(words), size):
			x.append(vocab['UNKNOWN'])
	return x#return sentence with word index

'''
def load_train_list():
	global trainfile
	trainList = []
	for line in open(trainfile,'r', encoding='UTF-8'):
		trainList.append(line.strip().split('	'))#question<Tab>pos_answer
	return trainList#[ ['question str','pos answer str'],[],[]....]
'''

def load_train_list():
	global trainfile
	trainList = []
	with open(trainfile,'r', encoding='UTF-8') as fr:
		for line in fr:
			question,answer = line.strip().split('\t')
			trainList.append([question,answer])#question<Tab>pos_answer
	return trainList#[ ['question str','pos answer str'],[],[]....]


def load_train_data(trainList, vocab):
	train_1, train_2, train_3 = [], [], []
	for i in range(len(trainList)):
		train_3_batch = []
		pos = trainList[i]  
		if(i<4):
			neg = trainList[4:]   
		elif(i>781):
			neg = trainList[:781]   
		else:
			neg = trainList[:i-2]+trainList[i+2:]   
		train_1.append(encode_sent(vocab, pos[0], conf['question_len']))	 
		train_2.append(encode_sent(vocab, pos[1], conf['answer_len']))	 
		neg = random.sample(neg,conf['neg_sample'])
		for j in range(len(neg)):
			train_3_batch.append(encode_sent(vocab, neg[j][1], conf['answer_len']))	
		train_3.append(train_3_batch)
	return np.array(train_1, dtype='float32'), np.array(train_2, dtype='float32'), np.array(train_3, dtype='float32')

'''
def load_eval_data_real(evalList, vocab):#[ ['question str','pos answer str'],[],[]....]
	labels, questions, answers = [], [], []
	label_batch, question_batch, answer_batch = [], [], []
	last_question = evalList[0][0]
	new_question = ''
	for i in range(len(evalList)):
		new_question = evalList[i][0]
		if(last_question!=new_question):
			last_question = new_question
			labels.append(label_batch)
			questions.append(question_batch)
			answers.append(answer_batch)
			label_batch, question_batch, answer_batch = [], [], []
		label_batch.append(1)
		question_batch.append(encode_sent(vocab, evalList[i][0], conf['question_len']))
		answer_batch.append(encode_sent(vocab, evalList[i][1], conf['answer_len']))
		if(i<4):
			neg = evalList[4:]
		elif(i>781):
			neg = evalList[:781]
		else:
			neg = evalList[:i-2]+evalList[i+2:]
		neg = random.sample(neg,conf['select_sample'])
		for j in range(len(neg)):
			answer_batch.append(encode_sent(vocab, neg[j][1], conf['answer_len']))
	return labels, questions, answers
'''

def load_eval_data_real(evalList, vocab):
	labels, questions, answers = [], [], []
	for i in range(len(evalList)):
		label_batch, question_batch, answer_batch = [], [], []
		label_batch.append(1)
		question_batch.append(encode_sent(vocab, evalList[i][0], conf['question_len']))
		answer_batch.append(encode_sent(vocab, evalList[i][1], conf['answer_len']))
		if (i < 4):
			neg = evalList[4:]
		elif (i > 781):
			neg = evalList[:781]
		else:
			neg = evalList[:i - 2] + evalList[i + 2:]
		neg = random.sample(neg, conf['select_sample'])
		for j in range(len(neg)):
			answer_batch.append(encode_sent(vocab, neg[j][1], conf['answer_len']))
		labels.append(label_batch)
		questions.append(question_batch)
		answers.append(answer_batch)
	return labels, questions, answers


def classify_SVM(question):
	vector = tfidf.transform(question)
	weight = vector.toarray()
	test_pred = svm.predict(weight)
	return test_pred

def get_candidate_list(class_num):
	print("class_num:", class_num)
	neg = candidate_list[class_num]
	#neg = []
	#for i in range(5):
	#	neg.extend(candidate_list[str(i)])
	return neg

class AnswerSelection(nn.Module):#需要继承nn.Module这个类
	def __init__(self, conf, word_embeddings):
		super(AnswerSelection, self).__init__()
		self.vocab_size = conf['vocab_size']
		self.embedding_dim = conf['embedding_dim']
		self.hidden_dim = conf['hidden_dim']
		self.question_len = conf['question_len']
		self.answer_len = conf['answer_len']

		self.word_embeddings = word_embeddings #second function called
		#lstm输入维度为self.embedding_dim，隐层维度为self.hidden_dim // 2
		#num_layers:隐层的层数，batch_first：True则输入输出的数据格式为 (batch, seq, feature)
		#lstm模块输入必须是三维张量，sequence序列结构、mini-batch结构、输入元素
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
		self.cnns = nn.ModuleList([nn.Conv1d(self.hidden_dim, conf['CNN_feature'], filter_size, stride=1, padding=filter_size-(i+1)) for i, filter_size in enumerate([1,3,5])])

		self.question_maxpool = nn.MaxPool1d(self.question_len, stride=1)
		self.answer_maxpool = nn.MaxPool1d(self.answer_len, stride=1)
		self.dropout = nn.Dropout(p=0.2)

	def forward(self, question, answer):
		question_embedding = list()
		answer_embedding = list()
		for i in question:
			question_embedding.append(self.word_embeddings[i])
		for i in answer:
			answer_embedding.append(self.word_embeddings[i])
		#question_embedding = torch.FloatTensor(question_embedding).cuda()
		#answer_embedding = torch.FloatTensor(answer_embedding).cuda()
		question_embedding = torch.FloatTensor(question_embedding)
		answer_embedding = torch.FloatTensor(answer_embedding)
		q_lstm, self.hiddenq = self.lstm(question_embedding)
		#print ("q_lstm:", q_lstm.size())
		a_lstm, self.hiddena = self.lstm(answer_embedding)
		#print ("a_lstm:", a_lstm.size())
		#contiguous()实现深拷贝,防止两个向量指向同一块空间
		q_lstm = q_lstm.contiguous()
		a_lstm = a_lstm.contiguous()
		q_lstm = q_lstm.view(-1,self.hidden_dim, self.question_len)
		#print ("q_lstm:", q_lstm.size())
		a_lstm = a_lstm.view(-1,self.hidden_dim, self.answer_len)
		#print ("a_lstm:", a_lstm.size())
		L=q_lstm.view(q_lstm.size()[0],self.hidden_dim,
		self.question_len,-1)*a_lstm.view(a_lstm.size()[0],
		self.hidden_dim,-1,self.answer_len)
		Aq=F.softmax(L,dim=3)
		Aa=F.softmax(L,dim=2)
		Cq=Aq*a_lstm.view(a_lstm.size()[0],self.hidden_dim,-1,self.answer_len)
		Cq=torch.sum(Cq,dim=3)
		Ca=Aa*q_lstm.view(q_lstm.size()[0],self.hidden_dim,self.question_len,-1)
		Ca=torch.sum(Ca,dim=2)
		
		question_pool = []
		answer_pool = []
		for cnn in self.cnns:
			question_conv = cnn(Cq)
			answer_conv = cnn(Ca)
			question_max_pool = self.question_maxpool(question_conv)
			answer_max_pool = self.answer_maxpool(answer_conv)
			question_activation = F.tanh(question_max_pool)
			answer_activation = F.tanh(answer_max_pool)
			question_pool.append(question_activation)
			answer_pool.append(answer_activation)
	
		question_output = torch.reshape(torch.cat(question_pool, dim=1),(len(question_pool[0]),-1))
		answer_output = torch.reshape(torch.cat(answer_pool, dim=1),(len(answer_pool[0]),-1))
		
		question_output = self.dropout(question_output)
		answer_output = self.dropout(answer_output)

		similarity = F.cosine_similarity(question_output, answer_output, dim=1)
		return similarity

	def fit(self, questions, good_answers, bad_answers):
		good_similarity = self.forward(questions, good_answers)#batch*ques_len,batch*ans_len
		bad_similarity = self.forward(questions, bad_answers[0])
		for i in range(1,bad_answers.size()[0]):
			bad_similarity = torch.max(bad_similarity,self.forward(questions, bad_answers[i]))
		#zeros = autograd.Variable(torch.zeros(good_similarity.size()[0]), requires_grad=False).cuda()
		zeros = autograd.Variable(torch.zeros(good_similarity.size()[0]), requires_grad=False)
		loss = torch.squeeze(torch.max(zeros, conf['margin'] - good_similarity + bad_similarity), dim=0)
		
		accuracy = torch.eq(loss,zeros).type(torch.DoubleTensor).mean()
		return loss.sum(), accuracy.item() 

class Evaluate():

	def __init__(self):
		self.conf = conf
		self.embedding_dim = conf['embedding_dim']
		self.vocab = build_vocab() 
		print('dict vocab',len(self.vocab))
		self.conf['vocab_size'] = len(self.vocab) + 1
		self.word_embeddings = load_word_embeddings(self.vocab, self.embedding_dim)
		self.model = AnswerSelection(self.conf, self.word_embeddings)

		'''
		if conf['mode'] == 'train':
			print("Training...")
			self.model = AnswerSelection(self.conf, self.word_embeddings)
			if conf['resume']:
				#load和load_state_dict的差别
				#load是保存模型和参数
				#load_state_dict仅保存参数
				self.model.load_state_dict(torch.load(s_l_path, map_location='cpu'))
			#self.model.cuda()
			self.train()
		if conf['mode'] == 'test':
			print("Testing")
			self.model = AnswerSelection(self.conf, self.word_embeddings)
			self.validate_real(validation=True, for_test=True)
		'''

	def train(self):
		batch_size = self.conf['batch_size']
		epochs = self.conf['epochs']
		trainList = load_train_list()
		train_x1, train_x2, train_x3 = load_train_data(trainList, self.vocab)
		questions = torch.LongTensor(train_x1)
		good_answers = torch.LongTensor(train_x2)
		bad_answers = torch.LongTensor(train_x3)	

		print("questions.size():",questions.size())
		print("good_answers.size():",good_answers.size())
		print("bad_answers.size():",bad_answers.size())
		bad_answers = bad_answers.view(bad_answers.size()[0],-1)
		print("bad_answers.size():",bad_answers.size())

		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf['learning_rate'])

		best_test = 0
		for i in range(epochs):
			train_loader = data.DataLoader(dataset=torch.cat([questions,good_answers,bad_answers],dim=1), 
											batch_size=batch_size, shuffle=True, num_workers=0)
			avg_loss = []
			avg_acc = []
			self.model.train()

			for every_step, train_batch in enumerate(train_loader):
				batch_question = autograd.Variable(train_batch[:,:self.conf['question_len']])
				batch_good_answer = autograd.Variable(train_batch[:,self.conf['question_len']:self.conf['question_len']+self.conf['answer_len']])
				batch_bad_answer = autograd.Variable(train_batch[:,self.conf['question_len']+self.conf['answer_len']:])
				batch_bad_answer = batch_bad_answer.contiguous()
				batch_bad_answer = batch_bad_answer.view(-1,batch_bad_answer.size()[0],self.conf['answer_len'])#bad_ans_num*batch*ans_len

				optimizer.zero_grad() 
				loss, acc = self.model.fit(batch_question, batch_good_answer, batch_bad_answer)
				avg_loss.append(loss.item()) 
				avg_acc.append(acc)
				loss.backward()
				optimizer.step()

			print("Epoch: {0} Epoch Average loss: {1} Accuracy {2}".format(str(i), str(np.mean(avg_loss)), str(np.mean(avg_acc))))
			#if(np.mean(avg_acc)>best_test):
			best_test=np.mean(avg_acc)
			if(conf['mode']=='train'):
				self.validate(validation=True, for_test=False)#训练用
			elif(conf['mode']=='test'):
				self.validate_real(validation=True, for_test=False)#实际测试用
			torch.save(self.model.state_dict(), s_l_path)

	def validate(self, validation=False, for_test=False):
		if(for_test):
			self.model.load_state_dict(torch.load(s_l_path))#when train() called, don't need
		#self.model.cuda()

		evalList=load_train_list()
		print("len evalList:",len(evalList))
		eval_labels,eval_questions,eval_answers = load_eval_data_real(evalList,self.vocab)
		print("len eval_labels:",len(eval_labels))
		print("len eval_questions:",len(eval_questions))
		print("len eval_answers:",len(eval_answers))
		c_1, c_2 = 0, 0
		for i in range(len(eval_labels)):#[labels:[] questions:[] answers:[]]
			good_nums = eval_labels[i].count(1)
			### 统计label中1出现的次数
			### load_eval_data_real在处理数据时，将每个样本的label设为1
			self.model.eval()
			question = autograd.Variable(torch.LongTensor(eval_questions[i]))
			answers = autograd.Variable(torch.LongTensor(eval_answers[i]))
			similarity = self.model.forward(question,answers)
			similarity = similarity.cpu().data.numpy()
			max_r = np.argmax(similarity)
			max_n = np.argmax(similarity[:good_nums])
			r = rankdata(similarity, method='max')
			c_1 += 1 if max_r == max_n else 0
			c_2 += 1 / float(r[max_r] - r[max_n] + 1)
		top1 = c_1 / float(len(eval_labels))
		mrr = c_2 / float(len(eval_labels))
		print('Top-1 Precision: %f' % top1)
		print('MRR: %f' % mrr)

	def validate_real(self, input_question, validation=True, for_test=False):
		if(for_test):
			self.model.load_state_dict(torch.load(s_l_path, map_location=lambda storage, loc:storage))#when train() called, don't need
		#self.model.cuda()
		newquestion = ''
		#while(newquestion != "exit"):
		newquestion = input_question
		newwords = jieba.cut(newquestion) #fenci
		a = "|".join(newwords)
		print(a)
		print("type(a): ",type(a))
		print("len(a): ",len(a))
		new_question = a
		evalList=load_train_list()#[ ['question str','pos answer str'],[],[]....]

		eval_questions, eval_answers = [], []

		eval_questions.append(encode_sent(self.vocab, new_question, conf['question_len']))

		###TODO:replace random select with similar answers
		### S1、判断问句所在的类别
		### S2、返回相应类别中所有的问答对
		### S3、将答案作为neg列表供模型筛选
		#neg = random.sample(evalList,conf['neg_sample'])
		x = [new_question]
		class_num = classify_SVM(x)
		neg = get_candidate_list(class_num[0])

		#print ("neg:",neg)

		for j in range(len(neg)):
			eval_answers.append(encode_sent(self.vocab, neg[j][1], conf['answer_len']))

		print("len eval_questions:",len(eval_questions))
		print("len eval_answers:",len(eval_answers))

		self.model.eval()

		question = autograd.Variable(torch.LongTensor(eval_questions))
		answers = autograd.Variable(torch.LongTensor(eval_answers))

		similarity = self.model.forward(question,answers)

		similarity = similarity.cpu().data.numpy()#change to cpu() is for np.argmax

		max_r = np.argmax(similarity)

		print("max_r:",max_r)
		#print("most similar question:",neg[max_r][0])
		#print("most similar answer:",neg[max_r][1])
		return neg[max_r][0],neg[max_r][1]

#ev = Evaluate()
