"""
예측을 진행하는 Restful API 서버
(확장 프로그램에서 사용)
"""
import re

from sanic import Sanic
from sanic.response import json
from sanic_cors import CORS, cross_origin

from ..train import get_model, options as opt
from ..utils import difference


# initialize transformer model
transformer = get_model()

last_epoch = 0
last_batch_iter = -1
if transformer.ckpt_manager.latest_checkpoint:
    # 체크포인트 불러오기
    transformer.ckpt.restore(transformer.ckpt_manager.latest_checkpoint)
    print("체크포인트 불러옴!")
    with open(f"{opt.checkpoint_path}/latest_epoch.txt", 'r') as f:
        last_epoch = int(f.read())
    with open(f"{opt.checkpoint_path}/latest_batch_iter.txt", 'r') as f:
        last_batch_iter = int(f.read())


# Sanic app
app = Sanic("hello_example")
CORS(app)

@app.post("/polish")
@cross_origin(app)
async def test(request):
    text = request.form['text'][0]

    sentences = text.split('. ')
    preds = [transformer.predict(sen) for sen in sentences]

    pred = '. '.join(preds)
    # 텍스트 마지막에 특수문자 있으면 추가
    tmp = ''
    for i in range(len(text)):
        idx = len(text) - i - 1
        if not re.match(r'[ㄱ-ㅎ가-힣0-9 ]', text[idx]):
            tmp += text[idx]
        else:
            break
    pred += ''.join(list(reversed(tmp)))

    diff = difference(text, pred)
    print(diff)

    result = []
    org = []
    pointed = []
    cnt = 0
    for d in diff:
        if isinstance(d, list):
            org.append(d[0])
            result.append(d[1])
            pointed.append('[P{}]'.format(cnt))
            cnt += 1
        else:
            pointed.append(d)

    return json({
        'result': result,
        'org': org,
        'pointed': ' '.join(pointed)
    })
