"""
Hyperparameters
"""
import numpy as np
import tensorflow as tf

int_dtype = tf.int16
int_dtype_str = 'int16'
float_dtype = tf.float32
float_dtype_np = np.float32
float_dtype_str = 'float32'

vocab_size = 5000
num_layers = 5
d_model = 256
dff = 512
num_heads = 16

input_vocab_size = vocab_size
target_vocab_size = vocab_size
dropout_rate = 0.1

EPOCHS = 20
checkpoint_path = "./korean_polisher/checkpoints"
MAX_LENGTH = 100
BATCH_SIZE = 64
