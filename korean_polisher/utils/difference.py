"""
두 문장의 차이를 구하는 코드 (예측 결과 해석에 사용)
"""
import difflib


def get_difference(a: str, b: str, join=False):
    """
    Returns the difference of two sentences.
    join: set this to True if the whole combined string is needed. Default: False.
    """
    while '  ' in a:
        a = a.replace('  ', ' ')
    while '  ' in b:
        b = b.replace('  ', ' ')
    a, b = a.split(' '), b.split(' ')
    diff = difflib.ndiff(a, b)
    if join:
        result = ''.join(diff)
    else:
        result = [i for i in diff if i != '?   -\n']
        #result = list(diff)

    return result


def comb(diff):
    # [' a', ' b', ' c', '+d', '+e', '-f'] -> [' ', ' ', ' ', '+de', '-f']
    # 연속되는 같은 심볼의 글자를 합침
    symbol = None
    result = []
    tmp = ''
    for word in diff:
        if word[0] == symbol:
            tmp += word[1:]
            continue
        if tmp:
            result.append(tmp)
            tmp = ''
            symbol = None
        if word[0] == ' ':
            result.append(word)
        else:
            symbol = word[0]
            tmp = word
    if tmp:
        result.append(tmp)
    return result


def jun(diff):
    # ['  안녕하세요', '- 안녕', '+ 하이'] -> ['안녕하세요', ['안녕', '하이']]
    #                       '안녕하세요 안녕' 문장에서 '안녕'을 '하이'로 바꿔야 한다는 뜻.
    # 어떤 부분을 어떻게 수정해야 할지 반환 {사용법: jun(comb(diff))}
    result = []
    i = len(diff) -1
    while True:
        if i < 0:
            break
        word = diff[i]
        if i == 0:
            pass
        if word[0] == '+':
            prev_word = diff[i - 1]
            if prev_word[0] == '-':
                result.append([prev_word[2:], word[2:]])
            else:
                result.append([prev_word[2:], prev_word[2:] + word[1:]])
            i -= 1
        elif word[0] == '-':
            prev_word = diff[i - 1]
            if prev_word[0] == '+':
                result.append([word[2:], prev_word[2:]])
                i -= 1
            else:
                result.append([word[2:], ''])
        else:
            result.append(word[2:])
        i -= 1
    return list(reversed(result))


def difference(text1, text2):
    diff = get_difference(text1, text2, join=False)
    return jun(comb(diff))
