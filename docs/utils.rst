korean_polisher.utils
=====================

.. contents::

korean_polisher.utils.get_env
-----------------------------

환경변수 가져오기

.. code-block:: Python

    korean_polisher.utils.get_env(
        key: str, fallback: Optional[str]=None
    )

Parameters
~~~~~~~~~~

- key: str - 환경변수 키값
- fallback: str - 환경변수가 없을 시 사용할 값

Returns
~~~~~~~

환경변수 값

Return Type
~~~~~~~~~~~

str

korean_polisher.utils.difference
--------------------------------

문장 차이 구하기

.. code-block:: Python

    difference(
        text1, text2
    )

Parameters
~~~~~~~~~~

- text1: str - 원본 문장
- text2: str - 대상 문장

Returns
~~~~~~~

두 문장의 차이를 포함하는 리스트

Return Type
~~~~~~~~~~~

List[Union[str, List[str]]]
