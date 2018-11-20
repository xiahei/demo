#-*-coding:utf-8-*-
from __future__ import unicode_literals

from django.db import models
from datetime import datetime

from users.models import UserProfile

# Create your models here.
class UserAsk(models.Model):
	#user = models.ForeignKey(UserProfile, verbose_name='用户', on_delete=models.CASCADE)
	email = models.CharField('邮箱', max_length=50)
	question = models.CharField('问题', max_length=300)
	answer = models.CharField('答案', max_length=150)
	#add_time = models.DateTimeField('添加时间', auto_now=True)

	class Meta:
		verbose_name = '查询问题'
		verbose_name_plural = verbose_name


class UserFavourable(models.Model):
	#email = models.CharField('邮箱', max_length=50)
	question = models.CharField('问题', max_length=300)
	answer = models.CharField('答案', max_length=150)
	#add_time = models.DateTimeField('添加时间', default=datetime.now)

	class Meta:
		verbose_name = '用户好评'
		verbose_name_plural = verbose_name



class UserFeedback(models.Model):
	#user = models.ForeignKey(UserProfile, verbose_name='用户', on_delete=models.CASCADE)
	email = models.CharField('邮箱', max_length=50, null=True)
	question = models.CharField('问题', max_length=300)
	answer = models.CharField('系统返回答案', max_length=150)
	fb = models.CharField('反馈意见', max_length=300,null=True)
	true_answer = models.CharField('修正后的答案', max_length=300,null=True)
	is_modify = models.BooleanField('是否已修改答案')
	#add_time = models.DateTimeField('添加时间', default=datetime.now)

	class Meta:
		verbose_name = '用户差评'
		verbose_name_plural = verbose_name

'''
class UserComments(models.Model):
	user = models.ForeignKey(UserProfile, verbose_name='用户', on_delete=models.CASCADE)
	comments = models.CharField('评论', max_length=200)
	add_time = models.DateTimeField('添加时间', default=datetime.now)

	class Meta:
		verbose_name = '用户评论'
		verbose_name_plural = verbose_name
'''