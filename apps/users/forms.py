#/usr/bin/python
#coding:utf-8

from django import forms
from captcha.fields import CaptchaField
from .models import UserProfile

#CharField()必须传入一个max_length字段
#但form.CharField()好像不一样

class LoginForm(forms.Form):
    username = forms.CharField(required=True)
    password = forms.CharField(required=True,min_length=5)
    # 此处说明：在前端html中，表单的<input name="username">,<input name="password">
    # 其中,LoginForm中的字段名字要与之对应

class RegisterForm(forms.Form):
    email = forms.EmailField(required=True)
    #EmailField():邮箱类型，可使用django内置EmailValidator进行邮箱地址合法性验证，默认max_length=254
    password = forms.CharField(required=True, min_length=5)
    captcha = CaptchaField(error_messages={"invalid": u"验证码错误"})


#忘记密码
class ForgetPwdForm(forms.Form):
    email = forms.EmailField(required=True)
    captcha = CaptchaField(error_messages={"invalid": u"验证码错误"})

#修改密码
class ModifyPwdForm(forms.Form):
    password1 = forms.CharField(required=True, min_length=5)
    password2 = forms.CharField(required=True, min_length=5)


class UploadImageForm(forms.ModelForm):
    '''
    用户更改图像
    '''
    class Meta:
        model = UserProfile
        fields = ["image"]


class UserInfoForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ["nick_name", "gender", "birthday", "address", "mobile"]
