#-*-coding:utf-8-*-
#algorithm/adminx.py

import xadmin
from .models import preprocessData, clusterRes, updateModel
from import_export import resources
from django.template import loader
from xadmin.views import BaseAdminPlugin, ListAdminView

'''
# 上传
class DataResources(resources.ModelResource):
    class Meta:
        model = preprocessData
        fields = ('id','question','answer')

@xadmin.sites.register(preprocessData)
class preprocessDataAdmin(object):
    import_export_args = {'import_resource_class':DataResources}

'''

class preprocessDataAdmin(object):
    list_display = ['question', 'answer']

xadmin.site.register(preprocessData, preprocessDataAdmin)

class clusterResAdmin(object):
    list_display = ['label','question','answer']
    list_filter = ['label']

xadmin.site.register(clusterRes, clusterResAdmin)


class updateModelAdmin(object):
    list_display = []
    object_list_template = "xadmin/views/update.html"
xadmin.site.register(updateModel, updateModelAdmin)


'''
class UpdateClusterPlugin(BaseAdminPlugin):
    is_update = False

    def init_request(self, *args, **kwargs):
        return bool(self.is_update)

    def block_update_button(self,context,nodes):
        context.update({'更新完成'})
        nodes.append(loader.render_to_string('xadmin/views/update.html',context_instance=context))

xadmin.site.register_plugin(UpdateClusterPlugin, ListAdminView)
'''