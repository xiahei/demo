#-*-coding:utf-8-*-

import urllib.request
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import pymysql
import datetime
import pytz
import time

class HtmlDownloader(object):

    def download(self, url):
        if url is None:
            return None

        response = urllib.request.urlopen(url)
        # print (response.getcode()) 返回200

        if response.getcode() != 200:
            return None

        return response.read()


class HtmlParser(object):

    # UTCS时间转换为时间戳 2018-07-13T16:00:00Z
    def cst_to_str(self, cst_time):
        temp_time = time.strptime(cst_time,'%a %b %d %H:%M:%S CST %Y')
        res_time = time.strftime('%Y-%m-%d %H:%M:%S',temp_time)
        return res_time

    def _get_new_urls(self,page_url,soup):
        new_urls = set()
        links = soup.find_all('a',href = re.compile(r'/zxdy/forum--method-listDefault,year-2014,forumid-442050,*'))        
        for link in links:
            new_url = link['href']
            new_full_url = urlparse.urljoin(page_url,new_url)	#urljoin将两段Url进行拼接
            new_urls.add(new_full_url)
        return new_urls

    def _get_new_data(self,page_url,soup):
        res_data = {}  # 字典

        #content
        summary = soup.find_all('a',class_="question_t_txt")
        questions = soup.find_all('div',class_="question")
        answers = soup.find_all('div',class_="question_a")
        department = soup.find_all("div",class_="zx-yxmc")
        time = soup.find_all("td",class_="question_t ch-table-center")
        for i in range(len(summary)):
            temp = {}
            q = questions[i].get_text().strip().encode('utf-8')
            a = answers[i].get_text().strip().split()[-1].encode('utf-8')
            s = summary[i].get_text().strip().encode('utf-8')
            d = department[i].get_text().strip().encode('utf-8')
            t = time[i].get_text().strip()
            temp['question'] = q
            temp['answer'] = a
            temp['summary'] = s
            temp['depart'] = d
            temp['time'] = self.cst_to_str(t)
            res_data[i] = temp
        return res_data

    def parse(self,page_url,html_cont):
        if page_url is None or html_cont is None:
            return

        #print 'html_cont:%s' %html_cont 打印成功
        soup = BeautifulSoup(html_cont,'html.parser',from_encoding = 'utf-8')
        new_urls = self._get_new_urls(page_url,soup)		#两个本地方法处理链接和数据
        new_data = self._get_new_data(page_url,soup)
        return new_urls, new_data


class UrlManager(object):

    def __init__(self):
        self.old_urls = set()
        self.flag = False

    def get_new_url(self,cursor,connect):
        try :
            cursor.execute("select url from crawler_urldisplay where is_crawled=False LIMIT 1")
            new_url = cursor.fetchone()[0]
            connect.commit()
        except:
            new_url = None
        return new_url

    def add_old_url(self,url):
        url = hash(url)
        if url in self.old_urls:
            self.flag = True
        else:
            self.flag = False
            self.old_urls.add(hash(url))
        return self.flag

class HtmlOutputer(object):

    def output(self,data,cursor,connect):
        for k,v in data.items():
            cursor.execute("insert ignore into crawler_qadisplay (question,answer,update_time) values('%s','%s','%s')" % (v['question'], v['answer'], v['time']))
            connect.commit()


class SpiderMain(object):  # 在构造函数中初始化各个对象

    def __init__(self):
        self.urls = UrlManager()
        self.downloader = HtmlDownloader()
        self.parser = HtmlParser()
        self.outputer = HtmlOutputer()

    def craw(self, root_url):  # 调度程序
        #连接mysql数据库
        conn = pymysql.connect(user='root',password='123456',database='EnrollmentQA')
        #创建游标
        cur = conn.cursor()
        count = 1  # 计数
        cur.execute("insert into crawler_urldisplay (url,is_crawled) values('%s',False)" %root_url)
        conn.commit()
        new_url = self.urls.get_new_url(cur, conn)
        while new_url:  # 当管理器中存在新的URL，循环
            #try:
            self.urls.add_old_url(new_url)             # url存入内存
            print ('crawl %d : %s' % (count, new_url))
            cur.execute("UPDATE crawler_urldisplay SET is_crawled=True where url='%s'" % new_url) # 更新已爬取的url标签
            conn.commit()
            html_cont = self.downloader.download(new_url)  # 下载页面
            new_urls, new_data = self.parser.parse(new_url, html_cont)  # 解析器解析页面数据
            for url in new_urls:
                if self.urls.add_old_url(url):
                    continue
                else:
                    cur.execute("insert into crawler_urldisplay (url,is_crawled) values('%s',False)" % url)
                    conn.commit()
            self.outputer.output(new_data, cur, conn)
            #if count == 1:  # 爬取1000个页面
            #    break

            #count = count + 1

            new_url = self.urls.get_new_url(cur, conn)  # 获取新的URL
            #except:
            #    print ('crawl failed')

        cur.close()
        conn.close()

if __name__ == "__main__":
    root_url = 'http://yz.chsi.com.cn/zxdy/forum--method-listDefault,year-2014,forumid-442050,start-0.dhtml'  # 入口地址
    #main_url = 'http://yz.chsi.com.cn'
    obj_spider = SpiderMain()
    obj_spider.craw(root_url)  # 启动爬虫


