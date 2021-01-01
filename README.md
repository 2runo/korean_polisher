# 바른문장 (GPT2)
## 소개
<바른문장>은 어색한 한국어 문장을 자연스럽게 바꿔 주는 프로그램입니다.

**해당 브랜치에서는 GPT2를 사용한 버전을 다룹니다.** 트랜스포머 (encoder-decoder) 구조를 사용한 버전은 [master 브랜치](https://github.com/2runo/korean_polisher)를 확인해 주세요.

## 시연 영상
https://youtu.be/USg-cSiLXTA

## 구동
1. git clone을 통해 레포지토리를 클론하거나 Code > Download ZIP을 통해 프로젝트를 다운로드합니다.
2. [체크포인트](https://drive.google.com/drive/folders/1Z8vvTnqH9zTjl6xQm6FT0-xTVCMRuzrN?usp=sharing)를 다운받은 후 압축을 풀어 `/model_chp` 폴더에 체크포인트 파일(`model-last.ckpt`)을 넣습니다.
3. 밑의 의존성 문단에 있는 패키지를 설치합니다.
4. 터미널에서 `python server.py`를 입력하여 서버를 실행합니다.
5. chrome://extensions/에 들어가 개발자 모드를 켜고 ‘압축해제된 확장 프로그램을 로드합니다’를 누른 후 프로젝트의 디렉토리(`extension`)를 선택합니다.
- 위 절차를 모두 밟으셨다면 [https://2runo.github.io/demo](https://2runo.github.io/demo)에서 성능을 테스트해보세요!
- 브라우저에서 글을 입력하는 도중, 확장 프로그램이 어색한 문장을 감지해 수정해 줍니다.

## 의존성
사용하는 파이썬 환경에 poetry 또는 requirements.txt를 통해 패키지를 설치해 주세요.

또한 크로미움 계열 브라우저에서만 정상작동합니다.

## 데이터
본 모델의 학습 방식은 비지도(unsupervised learning) 방식입니다. 즉, 지도학습 데이터는 필요하지 않으며 **단순 말뭉치**만 있으면 됩니다.

학습하는 구체적인 과정은 다음과 같습니다.

1. 나무위키, 위키피디아 등에서 정형화된 말뭉치를 수집합니다.
2. 문장을 어색하게 만드는 규칙을 작성합니다. (조사 바꾸기, 단어 섞기 등) [[코드 참조](https://github.com/2runo/korean_polisher/blob/master/korean_polisher/awkfy/awkfy.py)]
3. 규칙을 적용해 말뭉치 데이터를 어색하게 만들고 이를 복원하도록 GPT2 모델을 학습(fine tuning)합니다.

## 참조
<바른문장-GPT2>는 [KoGPT2](https://github.com/SKT-AI/KoGPT2)를 사용합니다.

[haven-jeon](https://github.com/haven-jeon) 님의 [KoGPT2-chatbot](https://github.com/haven-jeon/KoGPT2-chatbot) 레포지토리를 참고하였습니다.



## 노트
|    날짜    |                   내용                    |
| :--------: | :---------------------------------------: |
| 2021-01-01 |                 최초 배포                 |
