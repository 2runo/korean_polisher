<h1>바른문장</h1>
<h2>소개</h2>
바른문장은 어색한 한국어 문장을 자연스럽게 바꾸어 주는 프로그램입니다.<br>

<br><h2>구동</h2>
<ul>
  <li>git clone을 통해 레포지토리를 클론하거나 Code > Download ZIP을 통해 프로젝트를 다운로드합니다.</li>
  <li><a href="https://drive.google.com/drive/folders/1WGtpaz0pMwkNPR6Ek3UitPUHHSkp0KLW?usp=sharing">
    체크포인트
  </a>를 다운받은 후 model 폴더 안에 압축을 풀어 model/checkpoints 폴더에 체크포인트 파일들을 넣습니다.</li>
  <li>밑의 의존성 문단에 있는 패키지를 설치합니다.</li>
  <li>cmd 또는 터미널에서 cd model; python server.py를 입력해 server.py를 실행합니다.</li>
  <li>chrome://extensions/에 들어가 개발자 모드를 켜고 ‘압축해제된 확장 프로그램을 로드합니다’를 누른 후 프로젝트의 디렉토리를 선택합니다.</li>
  <li>브라우저에서 글을 입력하는 도중에 프로그램이 어색한 문장을 감지해 수정해 줍니다.</li>
</ul>

<br><h2>의존성</h2>
사용하는 파이썬 환경에 다음의 패키지들을 설치해 주세요. 아래에 기술되지 않은 패키지는 아마 대부분 다른 패키지와 함께 설치될 것입니다.<br>
또한 크로미움 계열 브라우저에서만 정상작동합니다.

numpy==1.19.1<br>
joblib==0.17.0<br>
beautifulsoup4==4.6.0<br>
requests==2.24.0<br>

tensorflow==2.3.1<br>
tokenizers==0.8.1<br>
konlpy==0.5.2<br>
hgtk==0.1.3<br>

sanic==20.9.0<br>
Sanic-Cors==0.10.0.post3<br>
