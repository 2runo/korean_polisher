"""
모델 클래스 정의
"""
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule

from gluonnlp.data import SentencepieceTokenizer

from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from kogpt2.dataset import CharDataset
from kogpt2.utils import get_tokenizer


class KoreanPolisherGPTpredict():
    def __init__(self, hparams,
                 u_tkn='<usr>', s_tkn='<sys>', eos='</s>', sent='<unused1>'):
        super(KoreanPolisherGPT, self).__init__()

        self.tok_path = get_tokenizer()
        self.tok = SentencepieceTokenizer(self.tok_path, num_best=0, alpha=0)  # tokenizer
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()  # gpt 모델 로드
        # token
        self.U_TKN, self.S_TKN = u_tkn, s_tkn
        self.SENT, self.EOS = sent, eos

    def predict(self, text):
        # text에 대해 예측 문장 생성
        q_tok = tok(text)
        a = ''
        a_tok = []
        cnt = 0
        while True:
            input_ids = torch.LongTensor([
                                             self.vocab[self.U_TKN]] + self.vocab[q_tok] +
                                         self.vocab[self.EOS, self.SENT] + self.vocab[sent_tokens] +
                                         self.vocab[self.EOS, self.S_TKN] +
                                         self.vocab[a_tok]).unsqueeze(dim=0)
            pred = self(input_ids)
            gen = self.vocab.to_tokens(
                torch.argmax(
                    pred, dim=-1
                ).squeeze().numpy().tolist()
            )[-1]
            if gen == self.EOS:
                break
            a += gen.replace('▁', ' ')
            a_tok = tok(a)

            cnt += 1
            if cnt >= 25:
                # 출력이 너무 길면
                a = q
                break

    def test(self):
        sent = '0'
        sent_tokens = self.tok(sent)
        with torch.no_grad():
            while True:
                q = input('어색한 문장 > ').strip()
                q_tok = self.tok(q)
                a = ''
                a_tok = []
                cnt = 0
                while True:
                    input_ids = torch.LongTensor([
                        self.vocab[self.U_TKN]] + self.vocab[q_tok] +
                        self.vocab[self.EOS, self.SENT] + self.vocab[sent_tokens] +
                        self.vocab[self.EOS, self.S_TKN] +
                        self.vocab[a_tok]).unsqueeze(dim=0)
                    pred = self.kogpt2(input_ids, return_dict=False)[0]
                    gen = self.vocab.to_tokens(
                        torch.argmax(
                            pred, dim=-1
                        ).squeeze().numpy().tolist()
                    )[-1]
                    if gen == self.EOS:
                        break
                    a += gen.replace('▁', ' ')
                    a_tok = self.tok(a)

                    cnt += 1
                    if cnt >= 25:
                        # 출력이 너무 길면
                        a = q
                        break
                print("수정된 문장 > {}".format(a.strip()))
