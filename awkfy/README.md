문장을 어색하게 만드는 기능.

awkfy.py가 메인 파일이다.

함수 종류

- `replace_josa` : 조사를 무작위로 바꾸기

	ex) 이건 축구공이야 -> 이건 축구공**은**
	
- `shuffle_letter` : 한 단어를 골라서 그 단어의 글자 위치를 바꿔버림

	ex) 선생님 안녕하세요 -> **생선**님 안녕하세요
	
- `attach_josa` : 아무 단어에 조사 붙이기

	ex) 선생님 안녕하세요 -> 선생님**은** 안녕하세요
	
- `reverse_plural` : 한 단어를 골라서 그 단어의 단수, 복수를 뒤바꿈

	ex) 선생님 안녕하세요 -> 선생님**들** 안녕하세요
	
- `shuffle_word` : 두 단어의 위치를 뒤바꿈

	ex) 나는 밥을 먹는다 -> **밥을 나는** 먹는다
	
- `insert_word` : 뜬금없이 명사를 삽입한다 (삽입할 단어는 assets/popular_nouns.txt 목록 중 하나 선택)

	ex) 나는 밥을 먹는다 -> 나는 **사람** 밥을 먹는다
	
- `insert_pronoun` : 뜬금없이 대명사를 삽입한다 (삽입할 대명사는 assets/pronouns.txt 목록 중 하나 선택)

	ex) 나는 밥을 먹는다 -> **그** 나는 밥을 먹는다
	
- `replace_word` : 미리 지정된 규칙대로 replace한다 (규칙은 assets/replace.txt를 따름)

	ex) 규칙 예시 : `안녕\t하이` -> '안녕'이란 단어를 '하이'라는 단어로 변경함
	
