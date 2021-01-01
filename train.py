"""
학습 및 테스트 진행
"""
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from kogpt2.kogpt_model import KoreanPolisherGPT
from kogpt2.parse import add_args


parser = argparse.ArgumentParser(description='Korean Polisher')

parser = add_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()


if __name__ == "__main__":
    if args.test:  # test
        model = KoreanPolisherGPT.load_from_checkpoint(args.model_path)
        model.test()
    else:  # train
        checkpoint_callback = ModelCheckpoint(
            dirpath="model_chp",
            save_last=True,
            prefix='model',
        )

        try:
            model = KoreanPolisherGPT.load_from_checkpoint(args.model_path)
            print("restored checkpoint")
        except FileNotFoundError:
            model = KoreanPolisherGPT(args)
        model.update_args(args)
        model.train()

        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
        trainer.fit(model)
