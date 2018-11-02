# coding:utf-8

from crawler import UrlManager,HtmlDownloader,HtmlParser,HtmlOutputer

class SpiderMain(object):  # 在构造函数中初始化各个对象

    def __init__(self):
        self.urls = UrlManager()
        self.downloader = HtmlDownloader()
        self.parser = HtmlParser()
        self.outputer = HtmlOutputer()

    def craw(self, root_url):  # 调度程序
        count = 1  # 计数
        self.urls.add_new_url(root_url)
        while self.urls.has_new_url():  # 当管理器中存在新的URL，循环
            try:
                new_url = self.urls.get_new_url()  # 获取新的URL
                print 'crawl %d : %s' % (count, new_url)
                html_cont = self.downloader.download(new_url)  # 下载页面
                new_urls, new_data = self.parser.parse(new_url, html_cont)  # 解析器解析页面数据
                self.urls.add_new_urls(new_urls)  # 将解析出的URL添加进URL管理器
                self.outputer.collect_data(new_data)  # 收集数据

                if count == 10:  # 爬取1000个页面
                    break

                count = count + 1

            except:
                print 'crawl failed'

            self.outputer.output()  # 输出处理好的数据


if __name__ == "__main__":
    root_url = 'http://yz.chsi.com.cn/zxdy/forum--method-listDefault,year-2014,forumid-442050,start-0.dhtml'  # 入口地址
    #main_url = 'http://yz.chsi.com.cn'
    obj_spider = SpiderMain()
    obj_spider.craw(root_url)  # 启动爬虫


