from .transformer import Transformer
from . import options as opt


def get_model():
    transformer = Transformer(
        opt.num_layers, opt.d_model, opt.num_heads, opt.dff, opt.input_vocab_size, opt.target_vocab_size,
        pe_input=opt.input_vocab_size,
        pe_target=opt.target_vocab_size,
        rate=opt.dropout_rate
    )
    return transformer
