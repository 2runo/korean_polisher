try:
    from .transformer import *
    from .scheduler import *
    from .options import *
    from .dataset_utils import *
except:
    from transformer import *
    from scheduler import *
    from options import *
    from dataset_utils import *
import time, os

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

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


tk = joblib.load('./tokenizer/tokenizer.joblib')  # 토크나이저


def evaluate(inp_sentence):
    start_token = [2]  # [CLS] token
    end_token = [3]  # [SEP] token

    # inp sentence is portuguese, hence adding the start and end token
    encoder_input = tokenize_batch([[inp_sentence]], tk)
    encoder_input = tf.cast(encoder_input, int_dtype)

    # as the target is english, the first word to the transformer should be the
    # english start token.
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

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), int_dtype)
        output = tf.concat([output, predicted_id], axis=-1)
        predicted_id = predicted_id.numpy()[0][0]

        # return the result if the predicted_id is equal to the end token
        if predicted_id == 3:  # [SEP] token이라면?
            #return tf.squeeze(output, axis=0), attention_weights
            return result, attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        result.append(predicted_id)

    #return tf.squeeze(output, axis=0), attention_weights
    return result, attention_weights

def predict(sentence):
    r, attention_weight = evaluate(sentence)
    return tk.decode(r)

while True:
    inp = input(':')
    print(predict(inp))
