#-*-coding:utf-8-*-
from __future__ import unicode_literals

from django.db import models
from datetime import datetime

from users.models import UserProfile

# Create your models here.
class qalist(models.Model):
	question=models.CharField(max_length=100)
	answer=models.CharField(max_length=100)


class UserAsk(models.Model):
	name = models.CharField('姓名', max_length=20)
	email = models.CharField('邮箱', max_length=50)
	add_time = models.DateTimeField('添加时间', default=datetime.now)

	class Meta:
		verbose_name = '用户咨询'
		verbose_name_plural = verbose_name

	def __str__(self):
		return self.name


class UserMessage(models.Model):
	user = models.IntegerField('接受用户', default=0)
	message = models.CharField('消息内容', max_length=500)
	has_read = models.BooleanField('是否已读', default=False)
	add_time = models.DateTimeField('添加时间', default=datetime.now)

	class Meta:
		verbose_name = '用户消息'
		verbose_name_plural = verbose_name

class UserComments(models.Model):
	user = models.ForeignKey(UserProfile, verbose_name='用户', on_delete=models.CASCADE)
	comments = models.CharField('评论', max_length=200)
	add_time = models.DateTimeField('添加时间', default=datetime.now)

	class Meta:
		verbose_name = '用户评论'
		verbose_name_plural = verbose_name