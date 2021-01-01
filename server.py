"""
예측을 진행하는 Restful API 서버
(확장 프로그램에서 사용)
"""
import argparse

from sanic import Sanic
from sanic.response import json
from sanic_cors import CORS, cross_origin

from korean_polisher.utils import difference, get_env
from kogpt2.kogpt_model import KoreanPolisherGPT
from kogpt2.parse import add_args


parser = argparse.ArgumentParser(description='Korean Polisher')
parser = add_args(parser)
args = parser.parse_args()

model = KoreanPolisherGPT.load_from_checkpoint(args.model_path)
print(model.predict('목적입니다 테스트'))

# Sanic app
app = Sanic("Korean Polisher")
CORS(app)


@app.post("/polish")
@cross_origin(app)
async def test(request):
    text = request.form['text'][0]

    sentences = text.split('. ')
    print('input:', sentences)
    preds = [model.predict(sen) for sen in sentences]

    pred = '. '.join(preds)

    diff = difference(text, pred)
    print('output:', diff)

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


if __name__ == "__main__":
    port = get_env('PORT', 8000)
    app.run(host="0.0.0.0", port=port)
