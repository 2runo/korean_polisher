korean_polisher.dataset
=======================

.. contents::

참고
----

dataset_batch_init 호출 -> iter하고 get_batch & awkfy_batch & tokenize_batch 호출

korean_polisher.dataset.dataset_batch_init
------------------------------------------

dataset을 batch 단위로 나누기

.. code-block:: Python

    dataset_batch_init(
        directory='./data/raw', batch_directory='./data/batch', batch_size=32
    )

Parameters
~~~~~~~~~~

- directory: str - batch 단위로 나뉜 데이터가 저장될 곳
- batch_directory: str - 나뉜 데이터를 담을 디렉터리
- batch_size: int - 배치 크기

korean_polisher.dataset.get_batch
---------------------------------

읽어온 텍스트 반환

.. code-block:: Python

    get_batch(
        index, batch_directory='./data/batch', batch_size=32
    )

Parameters
~~~~~~~~~~

- index: int - 읽어올 배치의 인덱스
- batch_directory: str - 나뉜 데이터가 담긴 디렉터리
- batch_size: int - 배치 크기

korean_polisher.dataset.awkfy_batch
-----------------------------------

배치를 awkfy해 반환

.. code-block:: Python
    awkfy_batch(
        batch: np.ndarray
    )
