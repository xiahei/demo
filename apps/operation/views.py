from django.shortcuts import render_to_response
from operation.algorithm import similarity

# Create your views here.
def search(request):
    input_question = str(request.GET['question'])
    if input_question:
        res = similarity.Sim()
        vector,pred_label = res.classify(input_question.strip())
        train_matrix,train_index = res.search_range(pred_label)
        q,a = res.cal_sim(train_matrix,vector,train_index)
        answer = {}
        answer['input_q'] = input_question
        answer['pred_q'] = q
        answer['pred_a'] = a 
        return render_to_response('result.html',answer)
    else:
    	return render_to_response('resultnull.html')

def test(request):
    return render_to_response('test.html')
    

def feedback(request):
    con=str(request.GET['contact'])
    bac=str(request.GET['message'])
    connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', db='qa', charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    cursor = connection.cursor()
    sql="INSERT INTO fankui(contact,feedback) values('%s','%s');" %(con,bac)
    cursor.execute(sql)
    connection.commit()
    return render_to_response('feedback.html')