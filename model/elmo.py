import os
import random
from pprint import pprint

import numpy as np
import joblib
import hgtk

import tensorflow as tf
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, TimeDistributed, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy


"""
def get_elmo_model(vocab_size, max_len, hidden_size):
    inputs = Input(shape=(max_len,))  # (100,)

    embedding_forward = Embedding(vocab_size, hidden_size)  # (100, hidden_size)
    lstm_forward_1 = LSTM(hidden_size, return_sequences=True, dropout=0.3, go_backwards=False)  # (100, hidden_size)
    lstm_forward_2 = LSTM(hidden_size, return_sequences=True, dropout=0.3, go_backwards=False)  # (100, hidden_size)

    embedding_backward = Embedding(hidden_size, hidden_size)  # (100, hidden_size)
    lstm_backward_1 = LSTM(hidden_size, return_sequences=True, dropout=0.3, go_backwards=True)  # (100, hidden_size)
    lstm_backward_2 = LSTM(hidden_size, return_sequences=True, dropout=0.3, go_backwards=True)  # (100, hidden_size)

    hidden_forward_1 = embedding_forward(inputs)
    hidden_forward_2 = lstm_forward_1(hidden_forward_1)
    hidden_forward_3 = lstm_forward_2(hidden_forward_2)

    hidden_backward_1 = embedding_backward(inputs)
    hidden_backward_2 = lstm_backward_1(hidden_backward_1)
    hidden_backward_3 = lstm_backward_2(hidden_backward_2)

    hidden_state = Concatenate()([hidden_forward_3, hidden_backward_3])  # (100, hidden_size*2)
    hidden_state = TimeDistributed(Dense(hidden_size))(hidden_state)  # (100, hidden_size)
    hidden_state = Flatten()(hidden_state)  # (100*hidden_size,)

    fc = Dense(vocab_size)
    outputs = fc(hidden_state)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
"""

def get_elmo_model(vocab_size, max_len, hidden_size):
    inputs = Input(shape=(max_len,))  # (100,)

    embedding_forward = Embedding(vocab_size, hidden_size)  # (100, hidden_size)
    lstm_forward_1 = LSTM(hidden_size, return_sequences=True, dropout=0.3, go_backwards=False)  # (100, hidden_size)
    lstm_forward_2 = LSTM(hidden_size, return_sequences=True, dropout=0.3, go_backwards=False)  # (100, hidden_size)

    #embedding_backward = Embedding(hidden_size, hidden_size)  # (100, hidden_size)
    lstm_backward_1 = LSTM(hidden_size, return_sequences=True, dropout=0.3, go_backwards=True)  # (100, hidden_size)
    lstm_backward_2 = LSTM(hidden_size, return_sequences=True, dropout=0.3, go_backwards=True)  # (100, hidden_size)

    embedding = embedding_forward(inputs)

    hidden_forward_2 = lstm_forward_1(embedding)
    hidden_forward_3 = lstm_forward_2(hidden_forward_2)

    hidden_backward_2 = lstm_backward_1(embedding)
    hidden_backward_3 = lstm_backward_2(hidden_backward_2)

    hidden_state = Concatenate()([hidden_forward_3, hidden_backward_3])  # (100, hidden_size*2)
    hidden_state = TimeDistributed(Dense(hidden_size))(hidden_state)  # (100, hidden_size)
    hidden_state = Flatten()(hidden_state)  # (100*hidden_size,)

    fc = Dense(vocab_size)
    outputs = fc(hidden_state)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


#  def get_dataset(filename, tokenizer, max_len=100, buffer_size=1000, batch_size=32):
def generate_dataset(filename, sequence_filename, tokenizer, max_len=100, buffer_size=1000, batch_size=32):
    """
    Writes sequenced text taken from filename to sequence_filename.
    Call before calling get_dataset.
    """

    root = os.getcwd()
    filename = os.path.join(root, filename)
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = f.readlines()
    

    # make input sequences
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.encode('[CLS]' + line + '[SEP]').ids

        for i in range(2, len(token_list)):  # exclude ['[CLS]', 'word'] because prediction is imposiible
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    
    input_sequences = [sequence + [0] * (max_len-len(sequence)) for sequence in input_sequences]
    # train_dataset = tf.data.Dataset.from_tensor_slices(input_sequences).shuffle(1000).batch(batch_size)

    
    # write to sequence_filename
    joblib.dump(input_sequences, sequence_filename)


def get_batch(index):
    filename = f"./data/result/{index}.joblib"
    batch: list = joblib.load(filename)
    batch = tf.convert_to_tensor(batch)

    return batch


""" @tf.function
def train_step(inputs, targets):
    '''Make sure elmo, loss_object, optimizer is global.'''
    with tf.GradientTape() as tape:
        predicted = elmo(inputs)
        loss = loss_object(targets, predicted)
        correct = (predicted == targets).numpy().astype(np.int32).sum()
    
    gradients = tape.gradient(loss, elmo.trainable_variables)
    optimizer.apply_gradients(zip(gradients, elmo.trainable_variables))

    return loss, correct """


""" def train(num_epochs=10):
    '''Make sure train_dataset, elmo, loss_object, optimizer is global.'''
    for epoch in range(num_epochs):
        # total_steps = len(train_dataset)
        total_steps = 174
        loss_sum = correct_sum = 0

        # for i, batch in train_dataset:
        for i in range(0, 173+1):
            batch = get_batch(i)

            inputs = batch[:, :-1]
            targets = batch[:, -1]
            loss, correct = train_step(inputs, targets)

            loss_sum += loss
            correct_sum += correct
            
            if (i+1) % 100 == 0:
                avg_loss = loss_sum/(i+1)
                avg_accuracy = correct_sum/((i+1)*batch)
                print(f"Epoch {epoch+1}, Step {i+1}/{total_steps}, Loss: {avg_loss:.2f}, Accuracy: {avg_accuracy*100}%") """


def get_last_epoch():
    last_epoch_dir = './elmo_checkpoints'
    last_epoch = os.path.join(last_epoch_dir, 'last_epoch.txt')


    if not os.path.exists(last_epoch_dir):
        os.mkdir(last_epoch_dir)
    if not os.path.exists(last_epoch):
        with open(last_epoch, 'w') as f:
            epoch = str(0)
            f.write(epoch)
        return 0
    else:
        with open(last_epoch, 'r') as f:
            epoch = int(f.read())
        return epoch
        


def get_last_file():
    last_file_dir = './elmo_checkpoints'
    last_file = os.path.join(last_file_dir, 'last_file.txt')


    if not os.path.exists(last_file_dir):
        os.mkdir(last_file_dir)
    if not os.path.exists(last_file):
        with open(last_file, 'w') as f:
            file = str(0)
            f.write(file)
        return 0
    else:
        with open(last_file, 'r') as f:
            epoch = int(f.read())
        return epoch
    

def restore_checkpoint():
    checkpoint_dir = './elmo_checkpoints'

    try:
        model.load_weights(checkpoint_dir)
    except:
        pass


if __name__ == '__main__':
    """ tokenizer = joblib.load('./tokenizer/tokenizer.joblib')

    # generate dataset
    if not os.path.exists('./data/dataset/sequence_dataset.joblib'):
        gen_dataset('./data/dataset/dataset.txt', './data/dataset/sequence_dataset.joblib', tokenizer, max_len=100)
    

    # get dataset
    train_dataset = get_dataset('./data/dataset/dataset.txt', batch_size=32)
    for sample in train_dataset.take(1):
        print(sample) """

    num_epochs = 5
    checkpoint_dir = './data/elmo_checkpoints'

    model: Model = get_elmo_model(vocab_size=5000, max_len=100, hidden_size=128)
    model.summary()

    optimizer = RMSprop(learning_rate=1e-3)
    loss_object = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])

    # checkpoint
    last_epoch = get_last_epoch()
    last_file = get_last_file()

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        save_weights_only=True,
        save_best_only=True
    )
    restore_checkpoint()

    # train(num_epochs=5)
    # iter over 175 data pieces and use model.fit
    for epoch in range(last_epoch, num_epochs):
        for index in range(last_file, 174+1):
            data = joblib.load(f'./result/{index}.joblib')
            x_train = data[:, :-1]
            y_train = data[:, -1]

            model.fit(x_train, y_train, batch_size=32, epochs=1, callbacks=[checkpoint_callback])
