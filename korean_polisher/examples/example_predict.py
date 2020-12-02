from ..predict import predict


print('문장을 입력하세요.')
while True:
    inp = input(':')
    print(predict(inp))
