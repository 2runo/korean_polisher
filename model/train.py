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
    def split_batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    batch_size = 1
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
test_inp = test_inp[:500]
test_tar = test_tar[:500]

# 학습
for epoch in range(last_epoch, EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    n_batch = len(os.listdir('./data/batch'))

    # inp -> portuguese, tar -> english
    for iteration in range(last_batch_iter, n_batch):
        # 데이터 가져오기
        # 배치는 1000부터 있으므로 (0~999까지는 test data임) iteration+1000을 함.
        batch = get_batch(iteration+1000, batch_directory='./data/batch')
        # 문장 어색하게 하기
        output = awkfy_batch(batch)
        # tokenizing
        inp = tokenize_batch(output, tk)  # 어색한 문장 -> inp
        tar = tokenize_batch(batch, tk)  # 자연스러운 문장 -> tar
        inp = tf.convert_to_tensor(inp, dtype=int_dtype)
        tar = tf.convert_to_tensor(tar, dtype=int_dtype)

        # 학습
        train_step(inp, tar)

        if iteration % 50 == 0:
            ckpt_save_path = ckpt_save(epoch, iteration)
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, iteration, train_loss.result(), train_accuracy.result()))
        if iteration % 500 == 0:
            print('evaluating..', end='\r')
            print('test loss, test acc:', evaluate(test_inp, test_tar))

    if (epoch + 1) % 5 == 0:
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
