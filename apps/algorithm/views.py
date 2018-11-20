# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.shortcuts import render_to_response
from algorithm.algormodel import zhaosheng_output, cluster
from algorithm.algormodel.algorithm import similarity
from django.http import HttpResponse, HttpResponseRedirect
import pymysql
from datetime import datetime

# Create your views here.

res = similarity.Sim()
#ev = zhaosheng_output.Evaluate()
conn = pymysql.connect(user='root', password='123456', database='EnrollmentQA')
cur = conn.cursor()

# Create your views here.

def search(request):
    input_question = str(request.GET['question'])
    if input_question:
        vector, pred_label = res.classify(input_question.strip())
        train_matrix, train_index = res.search_range(pred_label)
        q, a = res.cal_sim(train_matrix, vector, train_index)
        if q == 'null':
            return render_to_response('resultnotopic.html')
        else:
            sql = "insert into operation_userask (email,question,answer) values (%s,%s,%s)"
            if request.user.is_authenticated():
                name = str(request.user)
                cur.execute(sql, (name, input_question, a))
            else:
                cur.execute(sql, ('default', input_question, a))
            conn.commit()
            answer = {}
            answer['input_q'] = input_question
            answer['pred_q'] = q
            answer['pred_a'] = a
            return render_to_response('result.html', answer)
    return render_to_response('resultnull.html')
'''

def search(request):
    input_question = str(request.GET['question'])
    if input_question:
        q,a = ev.validate_real(input_question.strip())
        #sql = "insert into operation_userask (email,question,answer) values (%s,%s,%s)"
        #if request.user.is_authenticated():
        #    cur.execute(sql,(request.user,input_question,a))
        #else:
        #    cur.execute(sql,('None',input_question,a))
        #conn.commit()
        #cur.close()
        #conn.close()
        q = "".join(q.strip().split('|'))
        a = "".join(a.strip().split('|'))
        answer = {}
        answer['input_q'] = input_question
        answer['pred_q'] = q
        answer['pred_a'] = a
        return render_to_response('result.html', answer)
    else:
        return render_to_response('resultnull.html')
'''

def favourable(request):
    get_last_item = "select * from operation_userask order by id DESC limit 1;"
    cur.execute(get_last_item)
    item = cur.fetchone()
    #print ("item:",item)
    sql = "insert into operation_userfavourable (question,answer) values (%s,%s)"
    cur.execute(sql,(item[2],item[3]))
    conn.commit()
    #print (request)
    return render_to_response('search_form.html')


def feedback(request):
    #con = str(request.GET['contact'])
    #bac = str(request.GET['message'])
    #connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', db='qa', charset='utf8mb4',
    #                             cursorclass=pymysql.cursors.DictCursor)
    #cursor = connection.cursor()
    #sql = "INSERT INTO fankui(contact,feedback) values('%s','%s');" % (con, bac)
    #cursor.execute(sql)
    #connection.commit()
    return render_to_response('feedback.html')

def feedback2(request):
    email = str(request.GET['contact'])
    feedback = str(request.GET['message'])
    get_last_item = "select * from operation_userask order by id DESC limit 1;"
    cur.execute(get_last_item)
    item = cur.fetchone()
    sql = "insert into operation_userfeedback (email,fb,is_modify,question,answer) values (%s,%s,False,%s,%s)"
    cur.execute(sql, (email, feedback, item[2], item[3]))
    #sql_feedback = "insert into operation_userfeedback () values (%s,%s,False)"
    #cur.execute(sql_feedback, (email, feedback))
    conn.commit()
    return render_to_response('finish_feedback.html')

def update(request):
    print ("Yes")
    #cluster.cluster()
    return render_to_response('/xadmin/views/updatecluster.html')
