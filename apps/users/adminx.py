#-*-coding:utf-8-*-
# users/adminx.py

import xadmin

from .models import EmailVerifyRecord, Banner

#xadmin中这里是继承object，不再是继承admin
class EmailVerifyRecordAdmin(object):
    # 显示的列
    list_display = ['code', 'email', 'send_type', 'send_time']
    # 搜索的字段，不要添加时间搜索
    search_fields = ['code', 'email', 'send_type']
    # 过滤
    list_filter = ['code', 'email', 'send_type', 'send_time']

xadmin.site.register(EmailVerifyRecord,EmailVerifyRecordAdmin)


class BannerAdmin(object):
    list_display = ['title', 'image', 'url','index', 'add_time']
    search_fields = ['title', 'image', 'url','index']
    list_filter = ['title', 'image', 'url','index', 'add_time']


xadmin.site.register(Banner,BannerAdmin)

from xadmin import views

# 创建xadmin的最基本管理器配置，并与view绑定
class BaseSetting(object):
    # 开启主题功能
    enable_themes = True
    use_bootswatch = True

# 将基本配置管理与view绑定
xadmin.site.register(views.BaseAdminView,BaseSetting)


class GlobalSettings(object):
	site_title = "招生问答后台管理界面"
	site_footer = '研招组'
	menu_style = 'accordion'

# 将title和footer信息进行注册
xadmin.site.register(views.CommAdminView,GlobalSettings)