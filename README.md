# 바른문장
## 소개
<바른문장>은 어색한 한국어 문장을 자연스럽게 바꿔 주는 프로그램입니다.

## 시연 영상
https://youtu.be/USg-cSiLXTA

## 구동
1. git clone을 통해 레포지토리를 클론하거나 Code > Download ZIP을 통해 프로젝트를 다운로드합니다.
2. [체크포인트](https://drive.google.com/drive/folders/1ce_IdeLs2fsoIjmo1tYsVpmtt-wTI3Sf?usp=sharing)를 다운받은 후 `korean_polisher` 폴더 안에 압축을 풀어 `korean_polisher/checkpoints` 폴더에 체크포인트 파일들을 넣습니다.
3. 밑의 의존성 문단에 있는 패키지를 설치합니다.
4. 터미널에서 `python run_server.py`를 입력하여 서버를 실행합니다.
5. chrome://extensions/에 들어가 개발자 모드를 켜고 ‘압축해제된 확장 프로그램을 로드합니다’를 누른 후 프로젝트의 디렉토리(`extension`)를 선택합니다.
- 위 절차를 모두 밟으셨다면 [https://2runo.github.io/demo](https://2runo.github.io/demo)에서 성능을 테스트해보세요!
- 브라우저에서 글을 입력하는 도중, 확장 프로그램이 어색한 문장을 감지해 수정해 줍니다.

## 의존성
사용하는 파이썬 환경에 poetry 또는 requirements.txt를 통해 패키지를 설치해 주세요.

또한 크로미움 계열 브라우저에서만 정상작동합니다.

## 데이터
본 모델의 학습 방식은 self-supervised learning 방식입니다. 즉, 지도학습 데이터는 필요하지 않으며, **단순 말뭉치**만 있으면 됩니다.

학습하는 구체적인 과정은 다음과 같습니다.

1. 나무위키, 위키피디아 등에서 정형화된 말뭉치를 수집합니다.
2. 문장을 어색하게 만드는 규칙을 작성합니다. (조사 바꾸기, 단어 섞기 등) [[코드 참조](https://github.com/2runo/korean_polisher/blob/master/korean_polisher/awkfy/awkfy.py)]
3. 이를 활용하여 말뭉치 데이터를 어색하게 만들고 이를 복원하도록 트랜스포머(transformer) 모델을 학습합니다.

## 노트
|날짜|내용|
|:---:|:---:|
|2020-10-18|최초 배포|
|2020-12-11|추가 학습 진행 (약 4000만 문장 추가 학습)|
|2020-12-21|추가 학습 진행 (약 8000만 문장 추가 학습)|
