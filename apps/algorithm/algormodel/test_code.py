# from cluster.py

'''
# insert data into algorithm_preprocessdata
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


'''
# read data from test_cut
with open(BASE_DIR + '/test_cut','r', encoding="UTF-8") as fr:
    for line in fr:
        line = line.strip().split('\t')
        answer = line[1].strip().split('|')
        answers.append(' '.join(answer))
        data_pair.append([line[0],line[1]])
'''
import pymysql

conn = pymysql.connect(user='root',password='123456',database='EnrollmentQA')
cur = conn.cursor()

cur.execute("select * from algorithm_clusterres;")
data = list(cur.fetchall())
for item in data:
    id,label,question,answer = item
    print (id)

conn.commit()

conn.close()
cur.close()