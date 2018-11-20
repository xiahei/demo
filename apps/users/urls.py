#-*-coding:utf-8-*-
from django.conf.urls import url, include
from .views import UserinfoView, UploadImageView, UpdatePwdView, SendEmailCodeView, MyQuestionView, MyFeedbackView

app_name = 'users'

urlpatterns = [
    #用户信息
    url(r'^info/', UserinfoView.as_view(), name='user_info'),
    url(r'^image/upload/', UploadImageView.as_view(), name='image_upload'),
    url(r'^update/pwd/', UpdatePwdView.as_view(), name='update_pwd'),
    url(r'sendemail_code/', SendEmailCodeView.as_view(), name='sendemail_code'),
    url(r'^myquestion/', MyQuestionView.as_view(), name='my_question'),
    url(r'^myfeedback/', MyFeedbackView.as_view(), name='my_feedback'),

]

