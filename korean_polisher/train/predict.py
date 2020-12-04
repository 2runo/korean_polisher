"""
학습된 모델 테스트
"""
import re
import joblib

from .transformer import (
    Transformer,
    create_masks
)
from .scheduler import CustomSchedule
from .options import *
from ..dataset import tokenize_batch

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

learning_rate = CustomSchedule(200000)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

last_epoch = 0
last_batch_iter = 0
if ckpt_manager.latest_checkpoint:
    # 체크포인트 불러오기
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('체크포인트 불러옴!')


tk = joblib.load('../assets/tokenizer/tokenizer.joblib')  # 토크나이저


def only_pure(text):
    return ''.join(re.findall(r'[ㄱ-ㅎ가-힣0-9 ]', text))

def evaluate(inp_sentence):
    start_token = [2]  # [CLS] token
    end_token = [3]  # [SEP] token

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
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # 가장 마지막 단어만
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), int_dtype)
        output = tf.concat([output, predicted_id], axis=-1)
        predicted_id = predicted_id.numpy()[0][0]

        if predicted_id == 3:
            # [SEP] token이라면? -> 문장 끝
            return result, attention_weights

        # 예측 결과 합치기
        result.append(predicted_id)

    #return tf.squeeze(output, axis=0), attention_weights
    return result, attention_weights

def predict(sentence):
    if sentence.count(' ') == 0:
        # 단어가 하나라면 -> 예측하지 않음
        return sentence
    r, attention_weight = evaluate(sentence)
    return tk.decode(r)
