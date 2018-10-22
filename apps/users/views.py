#-*-coding:utf-8-*-
from .models import UserProfile, EmailVerifyRecord
from .forms import LoginForm, RegisterForm

from django.shortcuts import render
from django.views.generic.base import View
from django.contrib.auth import authenticate, login
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse

# Create your views here.
class IndexView(View):
    """
    首页
    """
    def get(self, request):
        #all_banners = Banner.objects.all().order_by("index")
        #courses = Course.objects.filter(is_banner=False)[:5]
        # banner_courses = Course.objects.filter(is_banner=True)[:3]
        # course_orgs = CourseOrg.objects.all()[:3]

        return render(request, "search_form.html", {
            #"all_banners": all_banners,
            #"courses": courses,
            # "banner_courses": banner_courses,
            # "course_orgs": course_orgs,
        })

class RegisterView(View):
    def get(self,request):
        register_form = RegisterForm()
        return render(request,"users/register.html", {'register_form': register_form})

    def post(self,request):
        register_form = RegisterForm(request.POST)
        if register_form.is_valid():
            user_name = request.POST.get("email", "")
            pass_word = request.POST.get("password", "")
            if UserProfile.objects.filter(email=user_name):
                return render(request, "users/register.html", {"register_form": register_form, "msg": "用户已存在"})
            user_profile = UserProfile()
            user_profile.username = user_name
            user_profile.email = user_name
            user_profile.is_active = False
            # 对明文密码进行加密
            user_profile.password = make_password(pass_word)
            user_profile.save()

            # 写入欢迎注册消息
            user_message = UserMessage()
            user_message.user = user_profile.id
            user_message.message = "欢迎注册"
            user_message.save()

            send_register_email(user_name,"register")
            return render(request,"users/login.html")
        else:
            return render(request,"users/register.html", {"register_form": register_form})

class LoginView(View):
    def get(self,request):
        return render(request, "users/login.html", {})

    def post(self,request):
        login_form = LoginForm(request.POST)
        # LoginForm()在传进来的时候有一个参数，这个参数为字典，request.POST就是一个字典
        # 所以此处一般传入request.POST内容
        if login_form.is_valid():
            user_name = request.POST.get("username", None)
            pass_word = request.POST.get("password", None)
            user = authenticate(username=user_name, password=pass_word)
            if user is not None:
                if user.is_active:
                    login(request, user)
                    return HttpResponseRedirect(reverse("index"))
                else:
                    return render(request, "users/login.html", {"msg": "用户未激活"})
            else:
                return render(request, "users/login.html", {"msg": "用户名或密码错误",'login_form':login_form   })
        else:
            return render(request, "users/login.html", {"login_form": login_form})


class LogoutView(View):
    def get(self, request):
        logout(request)
        from django.core.urlresolvers import reverse
        return HttpResponseRedirect(reverse("index"))