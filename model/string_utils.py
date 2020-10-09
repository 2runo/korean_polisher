import random
import hgtk
from awkfy_options import *

def insert_space_lst(pos, text):
    # ex) f([('나', 'Noun'), ('는', 'Josa'), ('축구', 'Noun'), ('가', 'Josa'), ('좋아', 'Adjective')], '나는 축구가 좋아')
    # -> [('나', 'Noun'), ('는', 'Josa'), (' ', ' '), ('축구', 'Noun'), ('가', 'Josa'), (' ', ' '), ('좋아', 'Adjective')]
    r = []
    n = 0
    while True:
        p = pos[0]
        pos = pos[1:]
        text = text[len(p[0]):]
        r.append(p)
        if not text:
            return r
        n2 = 0
        while text[0] == ' ':
            r.append((' ', ' '))
            text = text[1:]
            n2 += 1
            if n2 > 50:
                break
        if n > 100:
            break
        n += 1

def word_space(pos):
    # 품사만 보고 띄어쓰기
    r = ''
    for i, p in enumerate(pos):
        if i == 0 or p[1] in ['Josa', 'Eomi', 'PreEomi', 'Suffix'] or pos[i-1][1] in ['NounPrefix', 'VerbPrefix']:
            r += p[0]
        else:
            r += ' ' + p[0]
    return r


def only_pos(text, pos, specific_pos):
    # specific_pos 품사의 단어들 중 랜덤으로 하나만 선택하여 index 반환
    # ex) f('나는 조준희', [('나', 'Noun'), ('는', 'Josa'), ('조준희', 'Noun')], 'Noun') -> (insert_space_lst 결과, 0 or 2)
    pos = insert_space_lst(pos, text)  # 띄어쓰기 유지
    if isinstance(specific_pos, str):
        func = lambda x: x[1] == specific_pos
    elif isinstance(specific_pos, bool):
        func = lambda x: specific_pos
    else:
        func = specific_pos
    idx = random.choice([i for i, p in enumerate(pos) if func(p)])  # 랜덤으로 하나의 조사만 선택하여 index 반환
    return pos, idx


def shuffle_few_letters(word, n=2):
    # word의 특정 개수의 글자만 섞는다
    # ex) f('축구공', n=2) -> '축공구' or '구축공' 등등..
    def shuffle(x):
        return ''.join(random.sample(x, len(x)))
    if len(word) < n:
        return shuffle(word)
    i = random.randint(0, len(word) - n)
    shuffled = shuffle(word[i:i+n])
    n = 0
    while shuffled == word[i:i+n]:
        shuffled = shuffle(word[i:i + n])
        if n > 10:
            break
        n += 1
    return word[:i] + shuffled + word[i+n:]


def shuffle_few_words(words, n=2):
    # words 특정 개수의 단어만 섞는다
    # ex) f(['안녕', '나는', '축구', '좋아해'], n=2) -> ['안녕', '축구', '나는', '좋아해'] 등등..
    def shuffle(x):
        return random.sample(x, len(x))
    if len(words) < n:
        return shuffle(words)
    i = random.randint(0, len(words) - n)
    shuffled = shuffle(words[i:i+n])
    n = 0
    while shuffled == words[i:i+n]:
        shuffled = shuffle(words[i:i + n])
        if n > 10:
            break
        n += 1
    return words[:i] + shuffled + words[i+n:]


def match_by_length(text, lst):
    # ex) f('에게서', ['에게', '에', '는다']) -> '에게'
    lst = sorted(lst, key=lambda x: -len(x))
    for i in lst:
        if text.find(i) == 0:
            return i


def polish_josa(pos):
    # 조사를 수정한다. (pos 형식을 input으로 받음)
    # ex) f([('준희', 'Noun'), ('은', 'Josa')]) -> [('준희', 'Noun'), ('는', 'Josa')]
    idxs = [i for i, p in enumerate(pos) if p[1] == 'Josa']
    for i in idxs:
        if i == 0:
            continue
        pre_word = pos[i-1][0]
        josa = pos[i][0]
        if pre_word == ' ':
            continue
        try:
            decomposed = hgtk.letter.decompose(pre_word[-1])[2]
        except hgtk.exception.NotHangulException:
            # 한글이 아니면
            decomposed = False
        if decomposed:
            # 받침 있으면
            key = match_by_length(josa, JOSA_PARALLEL.keys())
            if key:
                josa = JOSA_PARALLEL[key] + josa[len(key):]
        else:
            # 받침 없으면
            key = match_by_length(josa, REVERSED_JOSA_PARALLEL.keys())
            if key:
                josa = REVERSED_JOSA_PARALLEL[key] + josa[len(key):]
        pos[i] = (josa, 'Josa')
    return pos

