"""
데이터를 로드하고 전처리하는 generator
"""
import os
import time

import numpy as np
import tensorflow as tf

from ..awkfy import (
    attach_josa, replace_josa,
    reverse_plural, 
    shuffle_word,
    insert_word, insert_pronoun,
    replace_word
)

"""
Note: run dataset_batch_init -> iter and run get_batch & awkfy_batch & tokenize_batch
"""


def dataset_batch_init(directory='./data/raw', batch_directory='./data/batch', batch_size=32):
    """
    directory: './data/raw'
    batch_directory = './data/batch'
    """
    
    if not os.path.exists(batch_directory):
        os.mkdir(batch_directory)

    # collect data from directory
    file_list = os.listdir(directory)
    data = []
    for file in file_list:
        filename = os.path.join(directory, file)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data.append(line)


    # 480000 -> 15000(total_step) * 32(batch_size) * 1(sentence)
    # 480010 -> 15000 * 32 * 1 + 10
    data_num = len(data)
    total_step = data_num // batch_size  # 15000
    data = data[:batch_size * total_step]  # drop last in order to avoid error when reshaping
    data = np.array(data).reshape(data_num // batch_size, batch_size, -1)


    # save to batch_dir
    for i, batch in enumerate(data):
        filename = os.path.join(batch_directory, f'batch{i}.txt')
        print(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            text = ""
            for s in batch:
                text += s.item()

            f.write(text[:-1])


def get_batch(index, batch_directory='./data/batch', batch_size=32):
    """
    Returns text read from 'batch{index}.txt'.
    """
    
    filename = f'{batch_directory}/batch{index}.txt'

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()


    lines = [s[:-1] if s[-1] == '\n' else s for s in lines]  # remove '\n' at end of sentence
    batch = np.array(lines).reshape(batch_size, 1)

    return batch


def awkfy_batch(batch: np.ndarray):
    """
    Returns a awkfied array of the batch.
    """

    # batch.shape: (batch_size, 1)
    shape = batch.shape
    batch = batch.reshape(-1)
    batch_size = len(batch)

    commands = [
        replace_josa,
        replace_josa,
        replace_josa,
        #shuffle_letter,
        attach_josa,
        reverse_plural,
        reverse_plural,
        shuffle_word,
        shuffle_word,
        insert_word,
        insert_pronoun,
        replace_word
    ]

    output = []

    for i in range(batch_size):
        try:
            output.append(commands[np.random.randint(0, len(commands))](batch[i]))
        except Exception as err:
            try:
                # 오류 발생 -> 한번만 더 시도
                output.append(commands[np.random.randint(0, len(commands))](batch[i]))
            except:
                # 오류 두 번 연속 발생 -> 원본 사용
                output.append(batch[i])
    return np.array(output).reshape(shape)


def tokenize_batch(batch: np.ndarray, tokenizer, max_len=100, padding=True):
    """
    Returns a tokenized array of the batch.
    """
    ids = np.array([tokenizer.encode('[CLS]' + s[0] + '[SEP]').ids for s in batch])
    if padding:
        ids = padding_(ids, max_len=max_len)
    return ids


def padding_(x, max_len=100):
    """패딩을 수행한다."""
    return tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len, padding="post")


def awkfy_dataset(tk, batch_directory='./data/batch', save_directory='./data/epoch0_batch'):
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    t = time.time()


    for i in range(1011, 879215):
        batch = get_batch(i, batch_directory=batch_directory)
        output = awkfy_batch(batch)
        r = []
        for j in range(len(output)):
            if output[j] == batch[j]:
                r.append(awkfy_batch(np.array([output[j]]))[0])
            else:
                r.append(output[j])
        output = np.array(r.copy())

        r = '\n'.join(['[SEP]'.join(j) for j in zip(batch.reshape(-1).tolist(), output.reshape(-1).tolist())])
        with open(f'{save_directory}/batch{i}.txt', 'w', encoding='utf-8') as f:
            f.write(r)

        if i % 100 == 0:
            print(i, time.time() - t)
