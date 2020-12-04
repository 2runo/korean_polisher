import os
import time

import joblib
import tensorflow as tf

from korean_polisher.train import (
    get_model,
    train_loss, train_accuracy
)
from korean_polisher.train.options import *
from korean_polisher.dataset import (
    awkfy_batch, tokenize_batch, get_batch
)


# model
transformer = get_model()

last_epoch = 0
last_batch_iter = -1
if transformer.ckpt_manager.latest_checkpoint:
    # 체크포인트 불러오기
    transformer.ckpt.restore(transformer.ckpt_manager.latest_checkpoint)
    print("체크포인트 불러옴!")
    with open(f"{checkpoint_path}/latest_epoch.txt", 'r') as f:
        last_epoch = int(f.read())
    with open(f"{checkpoint_path}/latest_batch_iter.txt", 'r') as f:
        last_batch_iter = int(f.read())
#print(transformer.encoder.enc_layers[0].ffn.weights)


# load tokenizer
tk = joblib.load('./korean_polisher/assets/tokenizer/tokenizer.joblib')  # 토크나이저

test_inp, test_tar = joblib.load('./data/testdata.joblib')  # 테스트 데이터
test_inp = test_inp[:10000]
test_tar = test_tar[:10000]

transformer.demo()  # 테스트 문장


# 학습
for epoch in range(last_epoch, EPOCHS):
    start = time.time()

    n_batch = len(os.listdir('./data/batch'))

    # inp -> portuguese, tar -> english
    for iteration in range(last_batch_iter+2, n_batch, BATCH_SIZE//32):
        # 데이터 가져오기
        # 배치는 1011부터 있으므로 (0~999까지는 날아갔고 1000~1010까지는 test data임) iteration+1010을 함.

        # 배치 크기 64 (32 * 2)
        batch = get_batch(iteration+1000, batch_directory='./data/batch')
        for i in range(1, BATCH_SIZE // 32):
            batch = np.concatenate([batch, get_batch(iteration+1000+i, batch_directory='./data/batch')])  # 배치 크기 32보다 크면 배치 32짜리 파일 여러 개 로드

        # 문장 어색하게 하기
        output = awkfy_batch(batch)
        if np.random.randint(0, 2):
            # 50% 확률로 -> 한 번 더
            output = awkfy_batch(output)
            print("one more awkfy!", end='\r')
        # tokenizing
        inp = tokenize_batch(output, tk)  # 어색한 문장 -> inp
        tar = tokenize_batch(batch, tk)  # 자연스러운 문장 -> tar
        inp = tf.convert_to_tensor(inp, dtype=int_dtype)
        tar = tf.convert_to_tensor(tar, dtype=int_dtype)

        # 현재 step
        cur_step = n_batch * epoch + iteration

        # 학습
        transformer.train_step(inp, tar)

        if iteration % 50 == 0:  # or (iteration + 1) % 50 == 0:
            print(f"Epoch {epoch} Batch {iteration} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
            # metrics reset (초기화하지 않으면 중첩됨 -> 정확한 평가 불가능)
            train_loss.reset_states()
            train_accuracy.reset_states()
        if iteration % 1000 == 0:  # or (iteration + 1) % 1000 == 0:
            ckpt_save_path = transformer.ckpt_save(epoch, iteration)
            test_loss, test_acc = transformer.evaluate(transformer, test_inp, test_tar)  # test loss, acc
            print("evaluating..", end='\r')
            # print("test loss, test acc:", test_loss, test_acc)
            print(f"test loss, test acc: {test_loss} {test_acc}")
            # test loss, acc 파일에 기록
            transformer.history(test_loss, test_acc)
            # demo
            transformer.demo()

    if (epoch + 1) % 5 == 0:
        print(f"Saving checkpoint for epoch {epoch} at {ckpt_save_path}")

    print(f"Epoch {epoch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")

    print(f"Time taken for 1 epoch: {time.time() - start} secs\n")
