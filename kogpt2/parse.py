import argparse


def add_args(parent_parser):
    # add model specific args
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--train',
                        action='store_true',
                        default=True,
                        help='for training')
    parser.add_argument('--test',
                        action='store_true',
                        default=False,
                        help='for testing')

    parser.add_argument('--model_path',
                        type=str,
                        default='model_chp/model-last.ckpt',
                        help='model weight path for testing')
    parser.add_argument('--data_path',
                        type=str,
                        default='./data/batch',
                        help='dataset path')

    parser.add_argument('--max-len',
                        type=int,
                        default=60,
                        help='max sentence length on input (default: 60)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=20,
                        help='batch size for training (default: 20)')
    parser.add_argument('--lr',
                        type=float,
                        default=5e-5,
                        help='The initial learning rate')
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.1,
                        help='warmup ratio')
    return parser
