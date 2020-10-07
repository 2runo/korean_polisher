from app import crawl_popular_page, gen_date_list, merge_files

if __name__ == '__main__':

    start_year = 2013
    end_year = 2013

    # in once: run together
    date_list = gen_date_list(start_year, end_year)
    crawl_popular_page('2013', date_list)  # year: 2013, date_list: ['20130101', '20130102', ...]
    merge_files(date_list, '2013.txt')

    ''' start_year = 2010
    end_year = 2010

    date_list = gen_date_list(start_year, end_year)
    # crawl_popular_page(date_list)
    merge_files(date_list, '2010.txt') '''
