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
last_batch_iter = -1
if ckpt_manager.latest_checkpoint:
    # 체크포인트 불러오기
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('체크포인트 불러옴!')
    with open(checkpoint_path + '/latest_epoch.txt', 'r') as f:
        last_epoch = int(f.read())
    with open(checkpoint_path + '/latest_batch_iter.txt', 'r') as f:
        last_batch_iter = int(f.read())


def ckpt_save(epoch, batch_iter):
    # 체크포인트 저장 (epoch, batch_iter도 저장)
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    with open(checkpoint_path + '/latest_epoch.txt', 'w') as f:
        f.write(str(epoch))
    with open(checkpoint_path + '/latest_batch_iter.txt', 'w') as f:
        f.write(str(batch_iter))
    return ckpt_manager.save()


def evaluate(inp, tar):
    # test loss, acc 계산
    def split_batch(iterable, n=1):
        # data를 batch 크기로 slice
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    batch_size = 64  # validation batch
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
        predictions, _ = transformer(inp, tar_inp,
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


def predict(inp_sentence):
    # input 텍스트 예측

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
            return tk.decode(result)

        # 예측 결과 합치기
        result.append(predicted_id)

    return tk.decode(result)


def demo():
    # 'demo.txt'의 텍스트를 예측하여 출력
    try:
        with open('demo.txt', 'r', encoding='utf8') as f:
            d = f.read()
        for i in d.split('\n'):
            if not len(i) == 0:
                print(i)
                print(predict(i))
    except:
        print('demo error')


def history(test_loss, test_acc):
    with open('history.txt', 'a+') as f:
        f.write('\n%s %s' % (test_loss, test_acc))


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)

        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


tk = joblib.load('./tokenizer/tokenizer.joblib')  # 토크나이저

test_inp, test_tar = joblib.load('./data/testdata.joblib')  # 테스트 데이터
test_inp = test_inp[:10000]
test_tar = test_tar[:10000]

demo()

# 학습
for epoch in range(last_epoch, EPOCHS):
    start = time.time()

    n_batch = len(os.listdir('./data/batch'))

    # inp -> portuguese, tar -> english
    for iteration in range(last_batch_iter+2, n_batch, 2):
        # 데이터 가져오기
        # 배치는 1011부터 있으므로 (0~999까지는 날아갔고 1000~1010까지는 test data임) iteration+1010을 함.

        # 배치 크기 64 (32 * 2)
        batch = get_batch(iteration+1000, batch_directory='./data/batch')
        batch = np.concatenate([batch, get_batch(iteration+1000+1, batch_directory='./data/batch')])  # 배치 크기 64이므로 배치 32짜리 파일 두 개 로드
        # 문장 어색하게 하기
        output = awkfy_batch(batch)
        # tokenizing
        inp = tokenize_batch(output, tk)  # 어색한 문장 -> inp
        tar = tokenize_batch(batch, tk)  # 자연스러운 문장 -> tar
        inp = tf.convert_to_tensor(inp, dtype=int_dtype)
        tar = tf.convert_to_tensor(tar, dtype=int_dtype)

        # 현재 step
        cur_step = n_batch * epoch + iteration

        # 학습
        train_step(inp, tar)

        if iteration % 50 == 0 or (iteration + 1) % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch, iteration, train_loss.result(), train_accuracy.result()))
            # metrics reset (초기화하지 않으면 중첩됨 -> 정확한 평가 불가능)
            train_loss.reset_states()
            train_accuracy.reset_states()
        if iteration % 1000 == 0 or (iteration + 1) % 1000 == 0:
            ckpt_save_path = ckpt_save(epoch, iteration)
            test_loss, test_acc = evaluate(test_inp, test_tar)  # test loss, acc
            print('evaluating..', end='\r')
            print('test loss, test acc:', test_loss, test_acc)
            # test loss, acc 파일에 기록
            history(test_loss, test_acc)
            # demo
            demo()

    if (epoch + 1) % 5 == 0:
        print('Saving checkpoint for epoch {} at {}'.format(epoch,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
