#-*-coding:utf-8-*-

from __future__ import unicode_literals

from datetime import datetime

# Create your models here.
from django.db import models

#问答数据展示
class QADisplay(models.Model):
    question = models.CharField('问题',max_length=300)
    answer = models.CharField('答案',max_length=150)
    update_time = models.DateTimeField('更新时间',default=datetime.now)

    class Meta:
        verbose_name = '问答数据'
        verbose_name_plural = verbose_name

#已爬取网页链接
class URLDisplay(models.Model):
    url = models.CharField('链接',max_length=300)
    is_crawled = models.BooleanField('是否爬取')

    class Meta:
        verbose_name = '爬取链接'
        verbose_name_plural = verbose_name

#历年考研统计数据
class StatisticsData(models.Model):
    year = models.CharField('年份',max_length=10)
    score = models.CharField('分数线',max_length=20)

    class Meta:
        verbose_name = '统计数据'
        verbose_name_plural = verbose_name