<h1 align="center">

&nbsp; <img src="https://github.com/OpenMind/OM1/assets/129569768/09e83c09-2847-48a8-8717-4908e5d22d35" alt="OM1" width="400" />

</h1>



<p align="center"><strong>OS for Intelligent Machines.</strong></p>



<p align="center">

&nbsp; <a href="https://docs.openmind.org">Technical Paper</a> ·

&nbsp; <a href="https://docs.openmind.org">Documentation</a> ·

&nbsp; <a href="https://twitter.com/OpenMind\_AGI">X</a> ·

&nbsp; <a href="https://discord.gg/openmind-996804118603120740">Discord</a>

</p>



---



\# 소개



\*\*OpenMind의 OM1\*\*은 개발자가 \*\*디지털 환경과 실제 로봇 모두에서 멀티모달 AI 에이전트를 생성·배포\*\*할 수 있도록 설계된 \*\*모듈형 AI 런타임\*\*입니다.  

휴머노이드, 스마트폰 앱, 웹사이트, 4족 보행 로봇, TurtleBot 4 같은 교육용 로봇까지 모두 지원합니다.



OM1 에이전트는 다음과 같은 입력을 처리할 수 있습니다:



\- 웹 데이터

\- 카메라 영상

\- 마이크 오디오

\- 소셜 미디어

\- LIDAR 등 다양한 센서 데이터



또한 다음과 같은 실제 물리 행동도 수행할 수 있습니다:



\- 이동 및 내비게이션

\- 자연스러운 대화

\- 물체 인식 후 동작

\- 얼굴 표정 동작 등



OM1의 목표는 \*\*사람을 중심으로 한 다기능 로봇을 누구나 쉽게 만들 수 있도록 하는 것\*\*이며,  

다양한 물리 포맷에 맞춰 \*\*업그레이드·재구성하기 쉽게 만드는 것\*\*입니다.



---



\# Capabilities of OM1



\### • Modular Architecture

Python 기반으로 설계되어 확장과 통합이 쉽습니다.



\### • Data Input

새로운 데이터·센서를 즉시 추가 가능.



\### • Hardware Support via Plugins

플러그인 방식으로 새 하드웨어를 연결할 수 있습니다.



지원 미들웨어:  

\- ROS2  

\- Zenoh (신규 개발 시 권장)  

\- CycloneDDS  



\### • Web-Based Debugging Display

WebSim UI(http://localhost:8000/)를 통해  

센서, 음성, 얼굴 표정, 제어 흐름 등을 실시간 시각화 가능.



\### • Pre-configured Endpoints

이미 다양한 AI 모델 엔드포인트를 설정해 제공:



\- OpenAI  

\- xAI  

\- DeepSeek  

\- Anthropic  

\- Meta  

\- Google Gemini  

\- NearAI  

\- 여러 Visual Language Models (VLMs)



---



\# Architecture Overview



(공식 README의 다이어그램 이미지를 그대로 사용하세요)



---



\# Getting Started



OM1을 처음 실행할 때는 \*\*Spot 에이전트\*\*를 사용해 보세요.  

Spot은 웹캠으로 사물을 감지해 텍스트 캡션을 생성하고,  

LLM이 이를 기반으로 \*\*movement / speech / face action\*\* 명령을 생성합니다.  

모든 동작은 WebSim UI에 표시됩니다.



---



\# Package Management and VENV



OM1은 \*\*uv 패키지 매니저\*\*를 사용합니다.



---



\# Clone the Repo



```bash

git clone https://github.com/OpenMind/OM1.git

cd OM1

git submodule update --init

uv venv

```



---



\# Install Dependencies



\## For MacOS



```bash

brew install portaudio ffmpeg

```



\## For Linux



```bash

sudo apt-get update

sudo apt-get install portaudio19-dev python-dev ffmpeg

```



---



\# Obtain an OpenMind API Key



OpenMind Portal에서 API Key를 발급받아 아래 중 하나로 설정하세요.



\- `config/spot.json5` 파일에 `openmind\_free` 값을 발급받은 키로 교체

\- 또는 `.env.example`을 `.env`로 복사 후 추가:



```bash

cp env.example .env

```



---



\# Launching OM1



Spot 에이전트 실행:



```bash

uv run src/run.py spot

```



이후 Spot 에이전트가 웹캠 입력을 받아 분석하고,  

LLM이 행동·음성·표정 명령을 생성해 WebSim에서 확인할 수 있습니다.



> 참고: `spot.json5`에서 ASR/TTS 설정이 필요합니다.



---



\# What’s Next?



\- 다양한 예제 실행  

\- 새로운 input/action 추가  

\- 직접 json5 구성 파일을 만들어 나만의 Agent 구성  

\- 시스템 프롬프트 수정해 새로운 행동 패턴 만들기  



---



\# Interfacing with New Robot Hardware



OM1은 다음 같은 고수준 명령을 받을 수 있는 로봇 SDK(HAL)를 가정합니다:



```

backflip  

run  

gently pick up the red apple  

move(0.37, 0, 0)  

smile

```



예시 (ROS2):



```python

elif output\_interface.action == "shake paw":

&nbsp;   if self.sport\_client:

&nbsp;       self.sport\_client.Hello()

```



만약 HAL이 없으면, RL + Unity/Gazebo + Depth Camera(ZED) + VLA 조합으로 직접 구현해야 합니다.



OM1이 지원하는 통신 방식:



\- USB / Serial  

\- ROS2  

\- CycloneDDS  

\- Zenoh  

\- WebSockets  



고급 HAL 예시는 Unitree의 C++ SDK를 참고하세요.



---



\# Recommended Development Platforms



OM1은 다음 환경에서 개발/테스트 되었습니다:



\- Jetson AGX Orin 64GB  

\- Mac Studio (M2 Ultra)  

\- Mac Mini (M4 Pro)  

\- 일반 Linux(Ubuntu 22.04)



Windows 및 Raspberry Pi 5 16GB에서도 실행 가능.



---



\# Full Autonomy Guidance



OM1의 \*\*완전 자율 모드\*\*는 4개의 서비스가 결합해 동작합니다:



1\. \*\*om1\*\*  

2\. \*\*unitree\_sdk\*\*  

3\. \*\*om1-avatar\*\*  

4\. \*\*om1-video-processor\*\*



각 서비스 역할:



\### ✔ om1  

AI Runtime



\### ✔ unitree\_sdk  

Unitree Go2용 SLAM/Nav2 제공



\### ✔ om1-avatar  

React 기반 UI \& Avatar Display



\### ✔ om1-video-processor  

얼굴 인식 / 오디오 캡처 / 스트리밍



---



\# Intro to BrainPack



연구에서 실제 로봇까지 모두 아우르는 새로운 \*\*BrainPack 플랫폼\*\*을 준비 중입니다.  

곧 BOM 및 DIY 문서가 공개될 예정입니다.



---



\# Clone the following repos



```text

https://github.com/OpenMind/OM1.git

https://github.com/OpenMind/unitree-sdk.git

https://github.com/OpenMind/OM1-avatar.git

https://github.com/OpenMind/OM1-video-processor.git

```



---



\# Starting the system



\## 1. OM1 실행



\### API Key 설정



\### Bash



```bash

vim ~/.bashrc

```



\### Zsh



```bash

vim ~/.zshrc

```



추가:



```bash

export OM\_API\_KEY="your\_api\_key"

```



\### docker-compose 실행



```bash

cd OM1

docker-compose up om1 -d --no-build

```



---



\## 2. unitree\_sdk 실행



```bash

cd unitree\_sdk

docker-compose up orchestrator -d --no-build

docker-compose up om1\_sensor -d --no-build

docker-compose up watchdog -d --no-build

```



---



\## 3. OM1-avatar 실행



```bash

cd OM1-avatar

docker-compose up om1\_avatar -d --no-build

```



---



\# Detailed Documentation



더 자세한 문서:  

https://docs.openmind.org



---



\# Contributing



컨트리뷰션 가이드를 꼭 읽고 PR을 제출해주세요.



---



\# License



이 프로젝트는 \*\*MIT License\*\* 하에 배포됩니다.  

자유로운 사용·수정·배포가 가능한 오픈소스 라이선스입니다.



---



<p align="center"><i>Powered by <a href="https://openmind.org">OpenMind</a></i></p>



