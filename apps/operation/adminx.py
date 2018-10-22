#-*-coding:utf-8-*-
#operation/adminx.py

import xadmin
from .models import UserAsk, UserMessage, UserComments

#用户咨询
class UserAskAdmin(object):
	list_display = ['name','email','add_time']
	search_fields = ['name','email']
	list_filter = ['name','email','add_time']

#用户消息
class UserMessageAdmin(object):
	list_display = ['user','message','has_read','add_time']
	search_fields = ['user','message','has_read']
	list_filter = ['user','message','has_read']

#用户评论
class UserCommentsAdmin(object):
	list_display = ['user','comments','add_time']
	search_fields = ['user','comments']
	list_filter = ['user','comments','add_time']


xadmin.site.register(UserAsk, UserAskAdmin)
xadmin.site.register(UserMessage, UserMessageAdmin)
xadmin.site.register(UserComments, UserCommentsAdmin)