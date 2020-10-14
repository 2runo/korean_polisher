try:
    from .elmo import get_elmo_model
except:
    from elmo import get_elmo_model
from copy import copy
import random
import tensorflow as tf



def get_elmo_embedding_model(checkpoint_filename, vocab_size=5000, max_len=100, hidden_size=128):
    # get model
    elmo: tf.keras.Model = get_elmo_model(vocab_size=5000, max_len=100, hidden_size=128)
    elmo.load_weights(checkpoint_filename)
    layers = elmo.layers


    # get layers
    embedding = layers[1]
    lstm_forward_1, lstm_forward_2 = layers[2], layers[4]
    lstm_backward_1, lstm_backward_2 = layers[3], layers[5]
    s1, s2, s3 =\
        tf.Variable(random.random(), trainable=True, name='s1'),\
        tf.Variable(random.random(), trainable=True, name='s2'),\
        tf.Variable(random.random(), trainable=True, name='s3')
    gamma = tf.Variable(random.random(), trainable=True, name='gamma')


    # forward pass
    inputs = tf.keras.layers.Input(shape=(max_len,))
    hidden_forward_embedding, hidden_backward_embedding = embedding(inputs), embedding(inputs)
    hidden_forward_1, hidden_backward_1 = lstm_forward_1(hidden_forward_embedding), lstm_backward_1(hidden_backward_embedding)
    hidden_forward_2, hidden_backward_2 = lstm_forward_2(hidden_forward_1), lstm_backward_2(hidden_backward_1)

    hidden_1 = tf.keras.layers.Concatenate()([hidden_forward_embedding, hidden_backward_embedding]) * s1
    hidden_2 = tf.keras.layers.Concatenate()([hidden_forward_1, hidden_backward_1]) * s2
    hidden_3 = tf.keras.layers.Concatenate()([hidden_forward_2, hidden_backward_2]) * s3
    hidden = tf.keras.layers.Add()([hidden_1, hidden_2, hidden_3])

    outputs = hidden * gamma


    # model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model



if __name__ == '__main__':
    elmo_embedding = get_elmo_embedding_model('./elmo/test3.h5')
    elmo_embedding.summary()
