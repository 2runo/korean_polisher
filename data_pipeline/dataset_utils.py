import os
from pprint import pprint
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from awkfy import *


"""
Note: run dataset_batch_init -> iter and run get_batch & awkfy_batch & tokenize_batch
"""



def dataset_batch_init(directory='./data/raw', batch_directory='./data/batch', batch_size=32):
    """
    directory: './data/raw'
    batch_directory = './data/batch'
    """
    root = os.getcwd()
    data_dir = os.path.join(root, directory)
    batch_dir = os.path.join(root, batch_directory)
    if not os.path.exists(batch_dir):
        os.mkdir(batch_dir)


    # collect data from directory
    file_list = os.listdir(os.path.join(data_dir))
    data = []
    for file in file_list:
        filename = os.path.join(data_dir, file)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data.append(line)
    
    # 480010 -> 15000 * 32 * 1 + 10
    data_num = len(data)
    total_step = data_num//batch_size  # 15000
    data = data[:batch_size*total_step]  # drop last in order to avoid error when reshaping
    data = np.array(data).reshape(data_num//batch_size, batch_size, -1)  # 480000 -> 15000(total_step) * 32(batch_size) * 1(sentence)


    # save to batch_dir
    for i, batch in enumerate(data):
        filename = os.path.join(batch_dir, f'batch{i}.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            text = ""
            for s in batch:
                text += s.item()

            f.write(text[:-1])


def get_batch(index, batch_directory='./data/batch', batch_size=32):
    """
    Returns text read from 'batch{index}.txt'.
    """
    root = os.getcwd()
    batch_dir = os.path.join(root, batch_directory)
    filename = os.path.join(batch_dir, f'batch{index}.txt')

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lines = [s[:-1] for s in lines]  # remove '\n' at end of sentence
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

    commands = {
            0: lambda x: replace_josa(x),
            1: lambda x: shuffle_letter(x),
            2: lambda x: attach_josa(x),
            3: lambda x: reverse_plural(x),
            4: lambda x: shuffle_word(x),
            5: lambda x: insert_word(x),
            6: lambda x: insert_pronoun(x),
            7: lambda x: replace_word(x),
    }

    """
    output = []
    for i in range(batch_size):
        num = np.random.randint(0, 8)
        func = commands[num]
        output.append(func(batch[i]))
    
    output = np.array(output).reshape(shape)
    """

    output = np.array([commands[np.random.randint(0, 8)](batch[i]) for i in range(batch_size)]).reshape(shape)
    return output



def tokenize_batch(batch: np.ndarray, tokenizer, max_len=100):
    """
    Returns a tokenized array of the batch.
    """
    ids = np.array([tokenizer.encode('[CLS]' + s[0] + '[SEP]').ids for s in batch])
    ids = padding(ids, max_len=max_len)
    return ids


def padding(x, max_len=100):
    # 패딩을 수행한다.
    return tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len, padding="post")


if __name__ == '__main__':
    
    # get data
    dataset_batch_init('./data/raw', './data/batch')

    # awkfy batch
    batch = get_batch(0, batch_directory='./data/batch')
    output = awkfy_batch(batch)
    pprint(output)
    print("--------------------------------------------------")

    # tokenize batch
    tk = joblib.load('./tokenizer.joblib')
    tokenized = tokenize_batch(batch, tk)
    pprint(tokenized)
