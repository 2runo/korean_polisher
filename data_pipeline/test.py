import joblib

tk = joblib.load('./tokenizer.joblib')


enc = tk.encode('대 등과 같이 보직에 따라 5를 지급받는 경우가 있지만 극소수이고 나머지는 실총기를 만져보기는 커녕 구경조차 못 해보고 그냥 그런게 있더라 하는 것만 알고 전역하는 병사들이 대부분이다')
print(enc.ids)
print(enc.words)
print(enc.tokens)
