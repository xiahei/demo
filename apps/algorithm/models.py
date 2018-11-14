# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
from django.db import models

# 预处理之后的数据
class preprocessData(models.Model):
    question = models.CharField('问题', max_length=500)
    answer = models.CharField('答案', max_length=150)

    def __unicode__(self):
        return '%s-%s' %(self.question, self.answer)

    class Meta:
        verbose_name = '预处理数据'
        verbose_name_plural = verbose_name

# 聚类后的算法结果
class clusterRes(models.Model):
    label = models.CharField('类别', max_length=10)
    question = models.CharField('问题', max_length=500)
    answer = models.CharField('答案', max_length=150)

    class Meta:
        verbose_name = '聚类结果'
        verbose_name_plural = verbose_name

class updateModel(models.Model):

    class Meta:
        verbose_name = '模型更新'
        verbose_name_plural = verbose_name

        def __unicode__(self):
            return self.Meta.verbose_name

