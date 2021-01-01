import logging

import numpy as np
import gluonnlp as nlp
from torch.utils.data import Dataset

from kogpt2.datamanger import DataManager


class CharDataset(Dataset):
    def __init__(self, tok_path, vocab, DATAMANAGER_MAX_LEN,
                 max_len=32,
                 U_TKN='<usr>', S_TKN='<sys>',
                 BOS='<s>', EOS='</s>',
                 MASK='<unused0>', SENT='<unused1>'):
        self._tok_path = tok_path
        self.tokenizer = None
        self.first = True

        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.maskt = MASK
        self.vocab = vocab
        self.max_len = max_len
        self.padder = nlp.data.PadSequence(
            max_len, pad_val=self.vocab[self.vocab.padding_token])

        # pass max_len to DataManager
        self.dm = DataManager(DATAMANAGER_MAX_LEN)

    def _activate_sp(self):
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0)

    def __len__(self):
        #return len(self._data)
        return 5000 * 50

    def __getitem__(self, idx):
        if self.tokenizer is None:
            self._activate_sp()
        #turn = self._data.iloc[idx]
        #q = turn['Q']
        #a = turn['A']
        #sentiment = str(turn['label'])
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
