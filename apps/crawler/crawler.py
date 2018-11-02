#-*-coding:utf-8-*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import urllib2
from bs4 import BeautifulSoup
import re
import urlparse

class HtmlDownloader(object):

    def download(self, url):
        if url is None:
            return None

        response = urllib2.urlopen(url)
        # print response.getcode() 返回200

        if response.getcode() != 200:
            return None

        return response.read()


class HtmlParser(object):

    def _get_new_urls(self,page_url,soup):
        new_urls = set()
        #http://baike.baidu.com/view/11578081.htm
        links = soup.find_all('a',href = re.compile(r'/zxdy/forum--method-listDefault,year-2014,forumid-442050,*'))        
        for link in links:
            new_url = link['href']
            new_full_url = urlparse.urljoin(page_url,new_url)	#urljoin将两段Url进行拼接
            new_urls.add(new_full_url)
        return new_urls

    def _get_new_data(self,page_url,soup):
        res_data = {}	#字典

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
            temp['time'] = t
            res_data[i] = temp
        return res_data

    def parse(self,page_url,html_cont):
        if page_url is None or html_cont is None:
            return

        #print 'html_cont:%s' %html_cont 打印成功
        soup = BeautifulSoup(html_cont,'html.parser',from_encoding = 'utf-8')
        new_urls = self._get_new_urls(page_url,soup)		#两个本地方法处理链接和数据
        new_data = self._get_new_data(page_url,soup)
        return new_urls,new_data

class UrlManager(object):
    def __init__(self):
        self.new_urls = set()
        self.old_urls = set()

    def add_new_url(self, url):
        if url is None:
            return
        if url not in self.new_urls and url not in self.old_urls:  # 新URL添加至待爬取列表中
            self.new_urls.add(url)

    def add_new_urls(self, urls):
        if urls is None or len(urls) == 0:
            return
        for url in urls:
            self.add_new_url(url)

    def has_new_url(self):
        return len(self.new_urls) != 0

    def get_new_url(self):
        new_url = self.new_urls.pop()  # 相当于弹出URL
        self.old_urls.add(new_url)
        return new_url


class HtmlOutputer(object):

    def __init__(self):
        self.datas = []  # 构造列表用于存储数据

    def collect_data(self, data):
        if data is None:
            return
        self.datas.append(data)

    def output(self):
        fout = open('res', 'a')
        
        #fout.write("question\tanswer\tsummary\tdepart\ttime\n")
        # python默认：ascii
        for data in self.datas:
            for k,v in data.iteritems():
                fout.write(v['question']+'\t'+v['answer']+'\t'+v['summary']+'\t'+v['depart']+'\t'+v['time']+'\n')
        fout.close()
