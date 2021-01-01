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


class KoreanPolisherGPT(LightningModule):
    def __init__(self, hparams,
                 u_tkn='<usr>', s_tkn='<sys>', eos='</s>', sent='<unused1>'):
        super(KoreanPolisherGPT, self).__init__()
        self.hparams = hparams  # args
        self.tok_path = get_tokenizer()
        self.tok = SentencepieceTokenizer(self.tok_path, num_best=0, alpha=0)  # tokenizer
        self.neg = -1e18  # mask에 곱할 값 (attention 연산에서 제외하기 위해)
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()  # gpt 모델 로드
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')  # loss
        # token
        self.U_TKN, self.S_TKN = u_tkn, s_tkn
        self.SENT, self.EOS = sent, eos

    def update_args(self, args) -> None:
        # max_len and data_path to pass to DataManager
        self.DATAMANAGER_MAX_LEN = args.max_len
        self.data_path = args.data_path

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output, _ = self.kogpt2(inputs, return_dict=False)
        return output

    def training_step(self, batch, batch_idx) -> dict:
        # training step
        token_ids, mask, label = batch
        out = self(token_ids)  # 연산
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))

        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        tensorboard_logs = {'train_loss': loss_avg}
        return {'loss': loss_avg, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    @staticmethod
    def _collate_fn(batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self) -> DataLoader:
        self.train_set = CharDataset(self.tok_path, self.vocab,
                                     max_len=self.hparams.max_len, data_path=self.data_path)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=0,
            shuffle=True, collate_fn=self._collate_fn
        )
        return train_dataloader

    def predict(self, q: str) -> str:
        # q에 대해 예측 문장 생성
        sent_tokens = self.tok('0')
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
            pred = self(input_ids)
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
        return a.strip()

    def test(self) -> None:
        # 테스트 수행
        with torch.no_grad():
            while True:
                q = input('어색한 문장 : ').strip()
                a = self.predict(q)
                print("수정된 문장 > {}".format(a))
