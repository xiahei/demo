#-*-coding:utf-8-*-
#operation/adminx.py

import xadmin
from .models import QADisplay, URLDisplay, StatisticsData
from import_export import resources

# 原始数据展示
class QADisplayAdmin(object):
    list_display = ['question', 'answer', 'update_time']
    list_filter = ['question','answer','update_time']

xadmin.site.register(QADisplay, QADisplayAdmin)

# 爬取链接展示
class URLDisplayAdmin(object):
    list_display = ['url', 'is_crawled']
    list_filter = ['url', 'is_crawled']

xadmin.site.register(URLDisplay, URLDisplayAdmin)

# 上传数据展示
class DataResource(resources.ModelResource):
    class Meta:
        model = StatisticsData
        fields = ('id','year','score')

@xadmin.sites.register(StatisticsData)
class DataAdmin(object):
    import_export_args = {'import_resource_class':DataResource}