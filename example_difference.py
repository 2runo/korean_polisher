from korean_polisher.utils import get_difference, comb, jun


text_1 = "안녕하세요.  안녕 그러면서도 여전히 팔리는 있는 것을 보면 괜찮은 프로그램인 듯. 나는 오늘 이것 집에 간다."
text_2 = "안녕하세요. 하이 그러면서도 여전히 팔리는 것을 하나 두울 보면 괜찮은 프로그램인 듯. 나는 집에 간다"

diff = get_difference(text_1, text_2, join=True)
print(diff)
diff = get_difference(text_1, text_2, join=False)
print(diff)
print(comb(diff))
print(jun(comb(diff)))
