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
