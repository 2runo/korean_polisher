"""
데이터를 로드하고 전처리하는 generator
"""
import logging

import numpy as np
import gluonnlp as nlp
from torch.utils.data import Dataset

from kogpt2.datamanger import DataManager


class CharDataset(Dataset):
    def __init__(self, tok_path, vocab,
                 max_len=32, data_path='./data/batch',
                 u_tkn='<usr>', s_tkn='<sys>',
                 bos='<s>', eos='</s>',
                 mask='<unused0>', sent='<unused1>'):
        self._tok_path = tok_path
        self.tokenizer = None
        self.first = True

        self.q_token, self.a_token = u_tkn, s_tkn
        self.sent_token = sent
        self.bos, self.eos = bos, eos
        self.maskt = mask
        self.vocab = vocab
        self.max_len = max_len
        self.padder = nlp.data.PadSequence(
            max_len, pad_val=self.vocab[self.vocab.padding_token]
        )

        # pass max_len to DataManager
        if data_path:
            self.dm = DataManager(self.max_len, data_path=data_path)
        else:
            self.dm = None

    def _activate_sp(self):
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0)

    def __len__(self):
        return 5000 * 10

    def __getitem__(self, idx):
        if self.tokenizer is None:
            self._activate_sp()

        q, a = self.dm.get()
        sentiment = '0'
        q_toked = [
            self.q_token,
        ] + self.tokenizer(q) + [
            self.eos,
        ] + [self.sent_token] + self.tokenizer(sentiment) + [
            self.eos,
        ]
        q_len = len(q_toked)
        a_toked = [
            self.a_token,
        ] + self.tokenizer(a) + [
            self.eos,
        ]
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.maskt,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        return (self.padder(self.vocab[q_toked + a_toked]), np.array(mask),
                self.padder(self.vocab[labels]))
