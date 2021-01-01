"""
예측을 진행하는 Restful API 서버
(확장 프로그램에서 사용)
"""
from sanic import Sanic
from sanic.response import json
from sanic_cors import CORS, cross_origin

from korean_polisher.utils import difference, get_env
from kogpt2.kogpt2chat import KoGPT2Chat


model = KoGPT2Chat.load_from_checkpoint("model_chp/model-last_ep454.ckpt")

print(model.predict('Use the keyboard to activate the textarea.'))

# Sanic app
app = Sanic("hello_example")
CORS(app)

@app.post("/polish")
@cross_origin(app)
async def test(request):
    text = request.form['text'][0]

    sentences = text.split('. ')
    print('input:', sentences)
    preds = [model.predict(sen) for sen in sentences]

    pred = '. '.join(preds)
    # # 텍스트 마지막에 특수문자 있으면 추가
    # tmp = ''
    # for i in range(len(text)):
    #     idx = len(text) - i - 1
    #     if not re.match(r'[ㄱ-ㅎ가-힣0-9 ]', text[idx]):
    #         tmp += text[idx]
    #     else:
    #         break
    # pred += ''.join(list(reversed(tmp)))

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


if __name__ == "__main__":
    port = get_env('PORT', 8000)
    app.run(host="0.0.0.0", port=port)
