"""demo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Import the include() function: from django.conf.urls import url, include
    3. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
#-*-coding:utf-8-*-

from django.conf.urls import url,include
from django.contrib import admin
import xadmin
from users.views import IndexView, LoginView, LogoutView, RegisterView, ForgetPwdView, ActiveUserView, ResetView, ModifyPwdView, UploadImageView
from operation import views

urlpatterns = [
    url(r'^xadmin/', xadmin.site.urls, name = "xadmin"),
    url(r'^$',IndexView.as_view(), name="index"),
    url(r'^login/$', LoginView.as_view(), name="login"),
    url(r'^logout/$', LogoutView.as_view(), name="logout"),
    url(r'^register/$', RegisterView.as_view(), name="register"),
    url(r'^search/$', views.search),
    url(r'^test/$', views.test),
    url(r'^feedback/$', views.feedback),
    url(r'^captcha/', include("captcha.urls")),
    url(r'^active/(?P<active_code>.*)/', ActiveUserView.as_view(), name="user_active"),
    url(r'^forget/', ForgetPwdView.as_view(), name="forget_pwd"),
    url(r'^reset/(?P<active_code>.*)/', ResetView.as_view(), name="reset_pwd"),
    url(r'^modify_pwd/', ModifyPwdView.as_view(), name="modify_pwd"),
    url(r'^users/', include("users.urls", namespace="users")),

]
