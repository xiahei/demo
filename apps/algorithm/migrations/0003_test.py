# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-11-13 15:50
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('algorithm', '0002_auto_20181113_1754'),
    ]

    operations = [
        migrations.CreateModel(
            name='test',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
            options={
                'verbose_name': '\u6a21\u578b\u66f4\u65b0',
                'verbose_name_plural': '\u6a21\u578b\u66f4\u65b0',
            },
        ),
    ]
