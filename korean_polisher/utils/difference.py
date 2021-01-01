"""
두 문장의 차이를 구하는 코드 (예측 결과 해석에 사용)
"""
import difflib


WHITESPACE = {
    '\n': '{:newline:}',
    '\r': '{:return:}',
    '\t': '{:tab:}',
    '\v': '{:vertical:}',
    '\f': '{:formfeed:}'
}
UPPER2LOWER = {
    'ᄀ': 'ㄱ', 'ᄁ': 'ㄲ', 'ᄂ': 'ㄴ', 'ᄃ': 'ㄷ', 'ᄄ': 'ㄸ', 'ᄅ': 'ㄹ', 'ᄆ': 'ㅁ', 'ᄇ': 'ㅂ',
    'ᄈ': 'ㅃ', 'ᄉ': 'ㅅ', 'ᄐ': 'ㅌ', 'ᄑ': 'ㅍ', 'ᄒ': 'ㅎ', 'ᄡ': 'ㅄ', 'ᅡ': 'ㅏ', 'ᅢ': 'ㅐ',
    'ᅣ': 'ㅑ', 'ᅤ': 'ㅒ', 'ᅥ': 'ㅓ', 'ᅦ': 'ㅔ', 'ᅧ': 'ㅕ', 'ᅨ': 'ㅖ', 'ᅩ': 'ㅗ', 'ᅰ': 'ㅞ',
    'ᅱ': 'ㅟ', 'ᅲ': 'ㅠ', 'ᅳ': 'ㅡ', 'ᅴ': 'ㅢ', 'ᅵ': 'ㅣ', 'ᅶ': 'ㅘ'
}


def replace_dict(t: str, d: dict, reverse: bool = False):
    # dict에 따라 replace 수행 (reverse=True라면 replace 순서 바뀜)
    # ex) f('hello my name', {'nam': 'NAM', 'my': 'i'}) -> 'hello i NAMe'
    for key, val in d.items():
        if reverse:
            t = t.replace(val, key)
        else:
            t = t.replace(key, val)
    return t


def normalize(t: str):
    # 연속 띄어쓰기 제거, 자모 replace
    # ex) f('hello  my name ᄀᄀ') -> 'hello my name ㄱㄱ'
    while '  ' in t:
        t = t.replace('  ', ' ')
    t = replace_dict(t, UPPER2LOWER)
    return t


def mask_whitespace(t: str):
    # whitespace replace
    # ex) f('\n hello') -> '{:newline:} hello'
    return replace_dict(t, WHITESPACE)


def unmask_whitespace(t: str):
    # whitespace replace
    # ex) f('{:newline:} hello') -> '\n hello'
    return replace_dict(t, WHITESPACE, reverse=True)


def get_difference(a: str, b: str, join=False):
    """
    Returns the difference of two sentences.
    join: set this to True if the whole combined string is needed. Default: False.
    """
    a = normalize(a)
    b = normalize(b)

    a = mask_whitespace(a)
    b = mask_whitespace(b)

    a, b = a.split(' '), b.split(' ')
    diff = difflib.ndiff(a, b)
    if join:
        result = ''.join(diff)
        result = unmask_whitespace(result)
    else:
        result = [i for i in diff if i[0] != '?']
        result = [unmask_whitespace(i) for i in result]

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
        prev_word = diff[i - 1]
        if i == 0:
            if word[0] == '+':
                if isinstance(result[-1], list):
                    result[-1][1] = word[2:] + ' ' + result[-1][1]
                else:
                    result[-1] = [result[-1], word[2:] + ' ' + result[-1]]
            elif word[0] == '-':
                result.append([word[2:], ''])
            else:
                result.append(word[2:])
            break
        if word[0] == '+':

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
