"""
뉴스 크롤링
"""
import os
import re
import time

import requests
from bs4 import BeautifulSoup


line = '-' * 50
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
}

# range of section
SECTION_MIN = 100
SECTION_MAX = 105


# specified year for clearity
def crawl_popular_page(year, date_list):  
    """crawl pages of date_list & save to sub_dir=f'./data/{year}'"""
    
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


    for date in date_list:
        
        # task: add contents to text and write every date
        text = ""

        # make url with section
        for section in range(SECTION_MIN, SECTION_MAX+1):
            main_url = f"https://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day&sectionId={section}&date={date}"  # with date

            # crawl main_url
            # in order to avoid network error
            while True:
                try:
                    html = requests.get(main_url, headers=headers)
                except:
                    continue
                break

            soup = BeautifulSoup(html.text, 'lxml')

            # find number of articles
            # in order to avoid IndexError,
            try:
                ARTICLE_NUM = len(soup.find('ol', class_='ranking_list').find_all('li'))
            # in order to avoid AttributeError, set ARTICLE_NUM to 0
            except AttributeError:
                ARTICLE_NUM = 0

            # find title, views, conURL
            title_list = soup.find_all('div', class_='ranking_headline')
            conURL_list = [f"https://news.naver.com{title.a.get('href')}" for title in title_list]


            for index in range(0, ARTICLE_NUM):
                conURL = conURL_list[index]  # 본문 URL

                # crawl conURL
                # in order to avoid network error
                while True:
                    try:
                        con_html = requests.get(conURL, headers=headers)
                    except:
                        continue
                    break

                # in order to avoid 404 error(blank page)
                try:
                    con_soup = BeautifulSoup(con_html.text, 'lxml')
                    con_soup = BeautifulSoup(str(con_soup.find('div', id='articleBodyContents')).replace('<br>', '\n').replace('<br/>', '\n'), 'lxml')  # replace '<br>'('<br/>') with '\n'
                    contents = con_soup.find('div', id='articleBodyContents').text.strip()

                    text += f'{contents}\n'
                except:
                    pass
        
        # task: save to './data/2010/' & merge files to './data/2010.txt'
        sub_dir = os.path.join(data_dir, year)  # './data/2010/'
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        
        filename = os.path.join(sub_dir, f'{date}.txt')  # './data/2010/20100101.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"{date} complete")


def gen_date_list(start_year, end_year):

    current_datetime = time.strftime('%Y%m%d')

    day_common_year = ['31', '28', '31', '30', '31', '30', '31', '31', '30', '31', '30', '31']
    day_leap_year = ['31', '29', '31', '30', '31', '30', '31', '31', '30', '31', '30', '31']

    date_list = []

    for y in range(start_year, int(end_year)+1):
        for m in range(1, 12+1):
            if is_leap_year(y):
                for d in range(1, int(day_leap_year[m-1])+1):
                    
                    date = f'{y}{m:0>2}{d:0>2}'
                    date_list.append(date)
                    
                    # break when date matches current datetime
                    if date == current_datetime:
                        return date_list
            else:
                for d in range(1, int(day_common_year[m-1])+1):
                    
                    date = f'{y}{m:0>2}{d:0>2}'
                    date_list.append(date)

                    # break when date matches current datetime
                    if date == current_datetime:
                        return date_list
    
    return date_list


def is_leap_year(year: str):
    if (int(year)%4 == 0 and int(year)%100 != 0) or int(year)%400 == 0:
        return 1
    else:
        return 0


def merge_files(year, merged_file_name):  # year: 2010, merged_file_name: '2010.txt'

    merged_file = f'./data/{merged_file_name}'
    with open(merged_file, 'w', encoding='utf-8') as merged_f:
        
        sub_dir = f'./data/{year}'  # read files from sub_dir='./data/2010'
        file_list: list = os.listdir(sub_dir)  # list of files
        file_list = list(map(lambda x: f'{sub_dir}/{x}', file_list))  # map file names


        for file in file_list:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                text = preprocess(text)
                merged_f.write(text)


def preprocess(text):
    """Basic preprocessing."""
    # remove parenthesis
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'\<[^)]*\>', '', text)
    text = re.sub(r'\【[^】]*\】', '', text)

    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'\S*.com\S*\s?', '', text)
    text = re.sub(r'\S*.co.kr\S*\s?', '', text)

    # remove certain characters
    text = re.sub(r'▶[^\n]*\n', '', text)
    text = re.sub(r'☞[^\n]*\n', '', text)
    text = re.sub(r'●[^\n]*\n', '', text)
    text = re.sub(r'△[^\n]*\n', '', text)
    text = re.sub(r'▲[^\n]*\n', '', text)
    text = re.sub(r'◆[^\n]*\n', '', text)
    text = re.sub(r'■[^\n]*\n', '', text)
    text = re.sub(r'Copyrights ⓒ\S*\s', '', text)

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)

    return text
