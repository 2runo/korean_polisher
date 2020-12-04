"""
모델 학습
"""
import os

from .transformer import (
    Transformer,
    loss_function, create_masks,
    train_loss, train_accuracy, train_step_signature
)
from .predict import (
    tk, optimizer, ckpt_manager
)
from .options import *
from ..dataset import (
    tokenize_batch
)


def ckpt_save(epoch, batch_iter):
    """체크포인트 저장 (epoch, batch_iter도 저장)"""

    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    with open(f"{checkpoint_path}/latest_epoch.txt", 'w') as f:
        f.write(str(epoch))
    with open(f"{checkpoint_path}/latest_batch_iter.txt", 'w') as f:
        f.write(str(batch_iter))
    
    return ckpt_manager.save()


def evaluate(model: Transformer, inp, tar):
    """test loss, acc 계산"""

    def split_batch(iterable, n=1):
        # data를 batch 크기로 slice
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    batch_size = BATCH_SIZE  # validation batch
    inp_batch = split_batch(inp, batch_size)
    tar_batch = split_batch(tar, batch_size)

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_loss.reset_states()
    test_accuracy.reset_states()

    for inp, tar in zip(inp_batch, tar_batch):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        predictions, _ = model(inp, tar_inp,
                                     False,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

        test_loss(loss)
        test_accuracy(tar_real, predictions)

    r = test_loss.result().numpy(), test_accuracy.result().numpy()
    del test_loss, test_accuracy
    return r


def predict(model: Transformer, inp_sentence):
    """input 텍스트 예측"""

    # 인코딩 (토크나이징)
    encoder_input = tokenize_batch([[inp_sentence]], tk)
    encoder_input = tf.cast(encoder_input, int_dtype)

    # 디코더 input
    decoder_input = [2]  # [CLS] token
    output = tf.expand_dims(decoder_input, 0)
    output = tf.cast(output, int_dtype)
    result = decoder_input.copy()


    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = \
            model(
                encoder_input,
                output,
                False,
                enc_padding_mask,
                combined_mask,
                dec_padding_mask
            )

        # 가장 마지막 단어만
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), int_dtype)
        output = tf.concat([output, predicted_id], axis=-1)
        predicted_id = predicted_id.numpy()[0][0]

        if predicted_id == 3:
            # [SEP] token이라면? -> 문장 끝
            return tk.decode(result)

        # 예측 결과 합치기
        result.append(predicted_id)

    return tk.decode(result)


def demo():
    """'demo.txt'의 텍스트를 예측하여 출력"""

    try:
        with open('./demo.txt', 'r', encoding='utf8') as f:
            d = f.read()
        for i in d.split('\n'):
            if not len(i) == 0:
                print(i)
                print(predict(i))
    except:
        print('demo error')


def history(test_loss, test_acc):
    with open('./history.txt', 'a+') as f:
        f.write('\n%s %s' % (test_loss, test_acc))


@tf.function(input_signature=train_step_signature)
def train_step(model: Transformer, inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = model(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)

        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)
