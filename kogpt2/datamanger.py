import os
import numpy as np

from korean_polisher.dataset import awkfy_batch, get_batch


class DataManager:
    def __init__(self, max_len):
        self.n_batch = len(os.listdir('./data/batch')) - 1000
        self.data = []
        # max_len is passed as argument
        self.max_len = max_len

    @staticmethod
    def awkfy(batch):
        batch = np.array(batch)
        # 문장 어색하게 하기
        output = awkfy_batch(batch)
        for i in range(len(output)):
            for j in range(10):
                if np.random.randint(0, max([3, len(output[i]) // 25])):
                    # 특정 확률로 -> 한 번 더
                    # 확률 : 75% -> 75/2% -> 75/4% -> ..
                    # 문장이 길수록 확률 증가
                    output[i] = awkfy_batch(np.array([output[i]]))
                else:
                    break

        return [(output[i][0].tolist(), batch[i][0].tolist()) for i in range(len(output))]

    def renew_batch(self):
        # 새로 랜덤한 파일 하나 읽어서 배치 마련
        iteration = np.random.randint(0, self.n_batch)
        batch = get_batch(iteration + 1000, batch_directory='./data/batch').tolist()
        batch = [i for i in batch if len(i[0]) < self.max_len]  # 너무 긴 텍스트는 제외
        self.data = self.awkfy(batch)  # [(x1, y1), (x2, y2), ...]

    def get(self):
        if len(self.data) == 0:
            # 데이터 없으면 -> 새로 로딩
            self.renew_batch()
        idx = np.random.randint(0, len(self.data))
        r = self.data[idx]
        del self.data[idx]
        return r
