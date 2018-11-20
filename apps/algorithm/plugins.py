from json import JSONEncoder

from django.template import loader
from django.template.response import TemplateResponse

from utils.dataUtils import replace_html
from xadmin.views import BaseAdminPlugin, ListAdminView
from xadmin.sites import site
from xadmin.plugins.actions import BaseActionView
from .models import *
import json, re


# 显示插件
class UpdateCrawlerPlugin(BaseAdminPlugin):
    pass

class UpdateClusterPlugin(BaseAdminPlugin):
    is_update = False

    def init_request(self, *args, **kwargs):
        return bool(self.is_update)

    def block_update_button(self.context,nodes):
        context.update({})
        nodes.append(loader.render_to_string('xadmin/views/update.html',context_instance=context))

class UpdateClassifyPlugin(BaseAdminPlugin):
    pass

class UpdateAnswerSelectionPlugin(BaseAdminPlugin):
    pass

site.register_plugin(UpdateCrawlerPlugin, ListAdminView)
site.register_plugin(UpdateClusterPlugin, ListAdminView)
site.register_plugin(UpdateClassifyPlugin, ListAdminView)
site.register_plugin(UpdateAnswerSelectionPlugin, ListAdminView)


