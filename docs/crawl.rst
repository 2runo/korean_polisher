korean_polisher.crawl
=====================

.. contents::

korean_polisher.crawl.crawl_popular_page
----------------------------------------

크롤링해 디렉터리 안에 일별로 저장

.. code-block:: Python

    crawl_popular_page(
        year, date_list
    )

Parameters
~~~~~~~~~~

- year: str - 저장할 파일에 들어간다. './data/{year}' 디렉터리에 저장
- date_list: List[str] - 크롤링할 일자

korean_polisher.crawl.gen_date_list
-----------------------------------

일별 날짜 리스트 생성

.. code-block:: Python

    gen_date_list(
        start_year, end_year
    )

Parameters
~~~~~~~~~~

- start_year: str - 시작 년도
- end_year: str - 종료 년도

Returns
~~~~~~~

- 일자 리스트

Return Type
~~~~~~~~~~~

- List[str]

korean_polisher.crawl.merge_files
---------------------------------

파일 합치기

.. code-block:: Python

    merge_files(
        year, merged_file_name
    )

Parameters
~~~~~~~~~~

- year: str - 합칠 파일의 년도. './data/{year}' 디렉터리에서 파일을 가져온다.
- merged_file_name: str - 합쳐진 파일 이름.
