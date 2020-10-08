import os
from pprint import pprint
import numpy as np
import pandas as pd
import joblib

from awkfy import *


def get_dataset(directory, batch_size=32, max_length=None, shuffle=True, seed=None, validation_split=None):
    """
    # Returns a list with all the lines of texts in './data/raw'.
    Returns a reshaped np.ndarray object in shape (total_steps, batch_size, 1).
    Example: 480000 data, batch_size=32 -> (15000, 32, 1)
    """

    root = os.getcwd()
    data_dir = os.path.join(root, directory)

    file_list = os.listdir(os.path.join(data_dir))
    data = []
    for file in file_list:
        filename = os.path.join(data_dir, file)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data.append(line)
    
    data_num = len(data)
    
    data = np.array(data).reshape(data_num//batch_size, batch_size, -1)  # 480000 -> 15000(num_samples) * 32(batch_size) * 1(sentence)    
    return data  # (total_steps, batch_size, 1)


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


def tokenize_batch(batch: np.ndarray, tokenizer):
    """
    Returns a tokenized array of the batch.
    """
    ids = np.array([tokenizer.encode(s[0]).ids for s in batch])
    return ids


if __name__ == '__main__':
    
    # get data
    data = get_dataset('./data/raw')
    print(data)
    print("--------------------------------------------------")

    # awkfy batch
    batch = np.array(
        ["선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다"]*32
    ).reshape(32, 1)  # (32, 1)
    output = awkfy_batch(batch)
    pprint(output)
    print("--------------------------------------------------")

    # tokenize batch
    tk = joblib.load('./tokenizer.joblib')
    output = tokenize_batch(batch, tk)
    pprint(output)
