try:
    from .predict import *
    from .difference import *
except:
    from predict import *
    from difference import *
from sanic import Sanic
from sanic.response import json
from sanic.response import text
from sanic.exceptions import abort
from sanic_cors import CORS, cross_origin
import mimetypes

app = Sanic("hello_example")
CORS(app)

@app.post("/polish")
@cross_origin(app)
async def test(request):
    #print('TESTMODE!!\n'*10)
    #return json({
    #    'org': ["있는", "이것", "것은"],
    #    'pointed': '안녕하세요. 그러면서도 여전히 팔리는 [P0] 것을 보면 괜찮은 프로그램인 듯. 나는 오늘 [P1] 집에 가려고 한다는 [P2] 사실이다.',
    #    'result': ['', '', '테스트']
    #})
    text = request.form['text'][0]

    sentences = text.split('. ')
    preds = [predict(sen) for sen in sentences]

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

    '''pointed = []
    result = []
    org = []
    cnt = 0
    for i, pred in enumerate(preds):
        if only_pure(pred).replace(' ', '') == only_pure(sentences[i]).replace(' ', ''):
            pointed.append(sentences[i])
        else:
            pointed.append('[P{}]'.format(cnt))
            result.append(pred)
            org.append(sentences[i])
            cnt += 1

    print('. '.join(preds))

    return json({
        'result': result,
        'org': org,
        'pointed': '. '.join(pointed)
    })'''

# 오류 핸들러
# @app.exception(Exception)
# async def exception_handler(request, exception):
#     return text("{} Error".format(exception.status_code))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
