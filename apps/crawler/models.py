#-*-coding:utf-8-*-

from __future__ import unicode_literals

from datetime import datetime

# Create your models here.
from django.db import models

class QADisplay(models.Model):
    question = models.CharField('问题',max_length=300)
    answer = models.CharField('答案',max_length=150)
    update_time = models.DateTimeField('更新时间',default=datetime.now)

    class Meta:
        verbose_name = '问答数据'
        verbose_name_plural = verbose_name