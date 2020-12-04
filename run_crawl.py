"""
뉴스 크롤링
"""
from korean_polisher.crawl import crawl_popular_page, gen_date_list, merge_files

if __name__ == '__main__':

    start_year = 2010
    end_year = 2010

    # sample: 2010
    # run loop to get from multiple years
    date_list = gen_date_list(start_year, end_year)
    crawl_popular_page('2010', date_list)  # year: 2010, date_list: ['20100101', '20100102', ...]
    merge_files(date_list, '2010.txt')
