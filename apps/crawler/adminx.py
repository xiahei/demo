#-*-coding:utf-8-*-
#operation/adminx.py

import xadmin
from .models import QADisplay, URLDisplay

#数据展示
class QADisplayAdmin(object):
    list_display = ['question', 'answer', 'update_time']
    list_filter = ['question','answer','update_time']

xadmin.site.register(QADisplay, QADisplayAdmin)

class URLDisplayAdmin(object):
    list_display = ['url', 'is_crawled']
    list_filter = ['url', 'is_crawled']

xadmin.site.register(URLDisplay, URLDisplayAdmin)