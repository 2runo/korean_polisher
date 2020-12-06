"""
문장을 어색하게 만드는 코드
"""
import random

import numpy as np
from konlpy.tag import Okt

from .string_utils import only_pos, polish_josa, shuffle_few_letters, shuffle_few_words
from .awkfy_options import JOSA_LIST, POPULAR_NOUNS, PRONOUNS, REPLACES


okt = Okt()


### 조사 바꿔치기 ###
def replace_josa(text):
    # ex) 나는 축구가 좋아 -> 나는 축구를 좋아 / 시간이 부족하다 -> 시간을 부족하다
    pos = okt.pos(text)
    try:
        pos, idx = only_pos(text, pos, 'Josa')  # 하나의 조사만 무작위 선택
    except IndexError:
        # 조사가 없다면 -> 그냥 반환
        return text
    pos = [i[0] for i in pos]

    # 바꿀 조사 선택
    to_josa = random.choice(JOSA_LIST)
    n = 0
    while to_josa == pos[idx]:
        # 바꿀 조사랑 기존 조사랑 똑같으면 안 되니까
        to_josa = random.choice(JOSA_LIST)
        n += 1
        if n > 10:
            break

    # 바꾸기
    pos[idx] = to_josa
    return ''.join(pos)


### 존댓말 어색화 ###
def awkfy_honorific(text):
    # ex) 나는 축구가 좋아 -> 저는 축구가 좋아 / 저는 축구가 좋아요 -> 저는 축구가 좋아
    pass


### 단어 글자 위치 바꾸기 ###
def shuffle_letter(text):
    # ex) 선생님 안녕하세요 -> 생선님 안녕하세요
    pos = okt.pos(text)
    try:
        pos, idx = only_pos(text, pos, 'Noun')  # 하나의 명사만 무작위 선택
    except IndexError:
        # 명사가 없다면 -> 다른 품사 선택
        pos, idx = only_pos(text, pos, True)
    pos = [i[0] for i in pos]

    pos[idx] = shuffle_few_letters(pos[idx])

    return ''.join(pos)


### 뜬금없이 조사 붙이기 ###
def attach_josa(text):
    # ex)  안녕하세요 반가워요 -> 안녕하세요는 반가워요
    words = text.split(' ')
    # 어느 단어에 붙일지 선택
    if len(words) == 1:
        idx = 0
    else:
        idx = random.randint(0, len(words) - 2)
    to_josa = random.choice(JOSA_LIST)  # 붙일 조사 선택
    words[idx] += to_josa
    return ' '.join(words)


### 단수,복수 뒤바꾸기 ###
def reverse_plural(text):
    # ex) 한국인은 쌀밥을 먹는다 -> 한국인은 쌀밥들을 주로 먹는다
    pos = okt.pos(text)
    try:
        pos, idx = only_pos(text, pos, 'Josa')  # 하나의 조사 무작위 선택
    except IndexError:
        # 조사가 없다면? -> 그냥 반환
        return text

    # if 선택된 조사가
    if pos[idx-1] == ('들', 'Suffix'):
        # 복수형이라면 -> 단수형으로 바꾸기
        del pos[idx-1]
    else:
        # 단수형이라면 -> 복수형으로 바꾸기
        pos.insert(idx, ('들', 'Suffix'))
    pos = polish_josa(pos)  # 어색한 조사 수정

    return ''.join([i[0] for i in pos])


### 단어 위치 바꾸기 ###
def shuffle_word(text):
    # ex) 나는 밥을 먹어 -> 밥을 나는 먹어
    # 이웃한 두 단어를 선택해서 위치를 바꿈

    # 띄어쓰기 교정

    # 단어 위치 바꾸기
    words = shuffle_few_words(text.split(' '))
    if len(text.split(' ')) > 15:
        words = shuffle_few_words(words)
    if len(text.split(' ')) > 30:
        words = shuffle_few_words(words)
    return ' '.join(words)


### 뜬금없는 단어 삽입 ###
def insert_word(text):
    words = text.split(' ')
    to_word = random.choice(POPULAR_NOUNS)  # 삽입할 단어
    words.insert(random.randint(0, len(words) - 1), to_word)
    return ' '.join(words)


### 뜬금없는 대명사 삽입 ###
def insert_pronoun(text):
    words = text.split(' ')
    to_word = random.choice(PRONOUNS)  # 삽입할 단어
    words.insert(random.randint(0, len(words) - 1), to_word)
    return ' '.join(words)


### 미리 지정된 단어 교체 ###
def replace_word(text):
    def get_random_idx(text, key):
        # text의 key와 매치되는 index들 중 하나만 무작위로 반환
        # ex) f('안녕 이건 축구공 축구공', '축구') -> 6 or 10
        try:
            return random.choice(np.where(np.array([text[i:i + len(key)] for i in range(len(text) - len(key) + 1)]) == key)[0])
        except:
            return None

    for key, val in random.sample(list(REPLACES.items()), len(REPLACES.items())):
        idx = get_random_idx(text, key)
        if not idx:
            continue
        text = text[:idx] + val + text[idx+len(key):]
        return text  # 한 번만 replace하고 반환
    return text
