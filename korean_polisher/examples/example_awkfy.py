from ..awkfy import (
    attach_josa, replace_josa,
    reverse_plural, 
    shuffle_letter, shuffle_word,
    insert_word, insert_pronoun,
    replace_word
)


if __name__ == '__main__':
    print("==조사 바꿔치기==")
    print("원문: 선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다")
    for _ in range(0, 4):
        print(f"결과: {replace_josa('선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다')}")
    
    print("==단어 글자 위치 바꾸기==")
    print('원문: 선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다')
    for _ in range(0, 4):
        print(f"결과: {shuffle_letter('선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다')}")
    
    print("==뜬금없이 조사 붙이기==")
    print("원문: 선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다")
    for _ in range(0, 4):
        print(f"결과: {attach_josa('선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다')}")

    print("==단수, 복수 뒤바꾸기==")
    print("원문: 선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다 한국인들은 쌀밥을 좋아합니다")
    for _ in range(0, 4):
        print(f"결과: {reverse_plural('선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다 한국인들은 쌀밥을 좋아합니다')}")
    
    print("==단어 위치 바꾸기==")
    print("원문: 선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다 한국인들은 쌀밥을 좋아합니다")
    for _ in range(0, 4):
        print(f"결과: {shuffle_word('선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다 한국인들은 쌀밥을 좋아합니다')}")
    
    print("==뜬금없는 단어 삽입==")
    print("원문: 선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다")
    for _ in range(0, 4):
        print(f"결과: {insert_word('선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다')}")

    print("==뜬금없는 대명사 삽입==")
    print("원문: 선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다")
    for _ in range(0, 4):
        print(f"결과: {insert_pronoun('선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다')}")

    print("==미리 지정된 단어 교체==")
    print("원문: 선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다")
    for _ in range(0, 4):
        print("결과: {replace_word('선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다')}")

    print("==짬뽕==")
    # print('결과:', reverse_plural(attach_josa(insert_pronoun(replace_word('선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다')))))
    print(f"결과: {reverse_plural(attach_josa(insert_pronoun(replace_word('선생님 안녕하세요 이건 축구공이에요 잘부탁드립니다'))))}")
