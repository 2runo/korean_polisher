import numpy as np
import joblib
from korean_polisher.dataset import awkfy_dataset, awkfy_batch, get_batch, tokenize_batch


# 데이터 들어왔을 때 한번만 실행
# dataset_batch_init('./data/raw', './data/batch')

tk = joblib.load('./korean_polisher/assets/tokenizer/tokenizer.joblib')

# 배치 awkfy
awkfy_dataset(tk)

# 테스트 데이터 만들기
print(get_batch(0, batch_directory='./data/test_batch'))

inp = np.array([])
tar = np.array([])
for j in range(2):
    for i in range(0, 1000):
        batch = get_batch(i, batch_directory='./data/test_batch')
        output = awkfy_batch(batch)

        # tokenize batch
        tokenized = tokenize_batch(batch, tk)
        tokenized2 = tokenize_batch(output, tk)
        try:
            inp = np.concatenate([inp, tokenized])
            tar = np.concatenate([tar, tokenized2])
        except:
            inp = tokenized.copy()
            tar = tokenized2.copy()
        if i % 10 == 0:
            print(inp.shape, tar.shape)
        del tokenized, batch, output
joblib.dump([inp, tar], 'testdata.joblib')
