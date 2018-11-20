#-*-coding:utf-8-*-
#operation/adminx.py

import xadmin
from .models import UserAsk, UserFavourable, UserFeedback

#用户好评
class UserAskAdmin(object):
	list_display = ['email','question','answer']
	search_fields = ['email']
	list_filter = ['email']

#用户好评
class UserFavourableAdmin(object):
	list_display = ['question','answer']
	#search_fields = ['email']
	#list_filter = ['email']

#用户差评
class UserFeedbackAdmin(object):
	list_display = ['email','question','answer','fb','true_answer','is_modify']
	search_fields = ['email','is_modify']
	list_filter = ['email','is_modify']

'''
#用户评论
class UserCommentsAdmin(object):
	list_display = ['user','comments','add_time']
	search_fields = ['user','comments']
	list_filter = ['user','comments','add_time']
'''

xadmin.site.register(UserAsk, UserAskAdmin)
xadmin.site.register(UserFavourable, UserFavourableAdmin)
xadmin.site.register(UserFeedback, UserFeedbackAdmin)
