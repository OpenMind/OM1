![OM_Banner_X2 (1)](https://github.com/user-attachments/assets/853153b7-351a-433d-9e1a-d257b781f93c) 

<p align="center">
<a href="https://arxiv.org/abs/2412.18588">技术论文</a> |
<a href="https://docs.openmind.org/">文档</a> |
<a href="https://x.com/openmind_agi">X</a> |
<a href="https://discord.gg/openmind">Discord</a>
</p>
<p align="right">
   <strong>中文</strong> | <a href="README.md">English</a>
</p>
**OpenMind 的 OM1 是一个模块化的 AI 运行时，它使开发者能够跨数字环境和物理机器人，（包括人形机器人、手机应用、网站、四足机器人以及 TurtleBot 4 等教育机器人）创建和部署多模态 AI 代理。OM1 代理可以处理各种输入，例如网络数据、社交媒体、摄像头画面和激光雷达数据，同时还能执行包括运动、自主导航和自然对话在内的物理动作。OM1 的目标是让开发者能够轻松创建功能强大的、以人为本的机器人，并且这些机器人易于升级和（重新）配置，以适应不同的物理形态。

## OM1 功能

* **模块化架构**：采用 Python 设计，以实现简洁性和无缝集成。
* **数据输入**：轻松处理新数据和传感器。
* **通过插件支持硬件**：通过插件支持新的硬件，用于 API 端点和特定机器人硬件连接到 `ROS2`、`Zenoh` 和 `CycloneDDS`。（推荐所有新开发使用 `Zenoh`）。
* **基于 Web 的调试界面**：通过 WebSim（访问 http://localhost:8000/）实时监控系统运行状态，便于可视化调试。
* **预配置的端点**：支持文本转语音、来自 OpenAI、xAI、DeepSeek、Anthropic、Meta、Gemini、NearAI 的多种大语言模型（LLM），以及多种视觉语言模型（VLM），并为每种服务提供了预配置端点。

## 架构概览
![Artboard 1@4x 1 (1)](https://github.com/user-attachments/assets/dd91457d-010f-43d8-960e-d1165834aa58)

## 快速开始

下面通过运行 Spot 代理 来快速体验 OM1。  
Spot 使用你的摄像头捕获并标注物体，这些文本描述会被发送给 LLM，LLM 返回 `movement`、`speech` 和 `face` 等动作指令。这些指令会在 WebSim 中展示，并附带基础的时间信息和调试信息。

### 包管理与虚拟环境（VENV）

你需要安装 [`uv` 包管理器](https://docs.astral.sh/uv/getting-started/installation/)。

### 克隆仓库

```bash
git clone https://github.com/OpenMind/OM1.git
cd OM1
git submodule update --init
uv venv
```

### 安装依赖项

适用于 macOS
```bash
brew install portaudio ffmpeg
```

适用于 Linux

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python-dev ffmpeg
```

###  获取 OpenMind API 密钥

前往 [OpenMind Portal](https://portal.openmind.org/) 获取你的 API Key。  
将其复制到 `config/spot.json5` 中，替换其中的 `openmind_free` 占位符。  
或者执行 `cp env.example .env`，并将你的 API Key 添加到 `.env` 文件中。

### 发布 OM1

运行
```bash
uv run src/run.py spot
```
启动 OM1 后，Spot Agent 将与你进行交互并执行（模拟）动作。  
如需了解如何将 OM1 连接到你的机器人硬件，请参考 [入门指南](https://docs.openmind.org/developing/1_get-started)。

## 接下来做什么？

- 尝试一些 [示例](https://docs.openmind.org/examples)
- 添加新的 `inputs` 和 `actions`
- 通过创建自定义的 `json5` 配置文件，使用不同的 inputs 与 actions 组合，设计你自己的 Agent 和机器人
- 修改位于 `/config/` 目录中的系统提示词，以创建新的行为模式

## 对接新的机器人硬件

OM1 假设机器人硬件提供一个高层 SDK，能够接收基础的运动和行为指令，例如 `backflip`、`run`、`gently pick up the red apple`、`move(0.37, 0, 0)` 以及 `smile`。

示例代码位于 `src/actions/move/connector/ros2.py`：

```python
...
elif output_interface.action == "shake paw":
    if self.sport_client:
        self.sport_client.Hello()
...
```

如果您的机器人硬件尚未提供合适的硬件抽象层 (HAL)，则需要结合强化学习 (RL) 等传统机器人方法、合适的仿真环境（例如 Unity、Gazebo）、传感器（例如手持式 ZED 深度相机）以及定制的垂直阵列阵列 (VLA) 来构建 HAL。此外，我们假设您的 HAL 能够接收运动轨迹，提供电池和热管理/监控，并校准和调整惯性测量单元 (IMU)、激光雷达 (LIDAR) 和磁力计等传感器。

OM1 可以通过 USB、串口、ROS2、CycloneDDS、Zenoh 或 WebSocket 与您的 HAL 进行交互。有关高级人形 HAL 的示例，请参阅[Unitree 的 C++ SDK](https://github.com/unitreerobotics/unitree_sdk2/blob/adee312b081c656ecd0bb4e936eed96325546296/example/g1/high_level/g1_loco_client_example.cpp#L159)。通常，HAL（尤其是 ROS2 代码）会被容器化，然后可以通过 DDS 中间件或 WebSocket 与 OM1 进行交互。


## 推荐开发平台

OM1 主要在以下平台上进行开发：

- Nvidia Thor（运行 JetPack 7.0）—— 完整支持
- Jetson AGX Orin 64GB（运行 Ubuntu 22.04 与 JetPack 6.1）—— 有限支持
- Mac Studio（Apple M2 Ultra，48 GB 统一内存，运行 macOS Sequoia）
- Mac Mini（Apple M4 Pro，48 GB 统一内存，运行 macOS Sequoia）
- 通用 Linux 设备（运行 Ubuntu 22.04）

理论上，OM1 也可以运行在其他平台上（如 Windows），以及微控制器设备，例如 Raspberry Pi 5 16GB。

## 完全自主引导

我们很高兴地宣布Unitree Go2 和 G1**全面实现自主运行**。全面自主运行包含四项服务，它们协同工作，形成一个循环，无需人工干预：
- **om1**
- **unitree_sdk**  
  一个 ROS 2 软件包，使用 RPLiDAR 传感器、SLAM Toolbox 以及 Nav2 技术栈，为 Unitree Go2 机器人提供 SLAM（同步定位与建图）能力
- **om1-avatar**  
  一个基于 React 的现代化前端应用，为 OM1 机器人软件提供用户界面与 Avatar 显示系统
- **om1-video-processor**  
  OM1 视频处理器是一个基于 Docker 的解决方案，用于实现实时视频流、人脸识别以及音频采集

## BrainPack 是什么？

从科研走向真实世界自治，一套能够与你一起学习、行动并共同构建的平台。

我们即将发布 **BOM（物料清单）** 以及相关的 **DIY** 细节，敬请期待！

### 克隆以下仓库

- https://github.com/OpenMind/OM1.git
- https://github.com/OpenMind/unitree-sdk.git
- https://github.com/OpenMind/OM1-avatar.git
- https://github.com/OpenMind/OM1-video-processor.git

## 启动系统

要启动所有服务，请按以下步骤执行：

- For OM1

设置 API Key

For Bash: vim ~/.bashrc or ~/.bash_profile.

For Zsh: vim ~/.zshrc.

需添加

```bash
export OM_API_KEY="your_api_key"
```

更新 `docker-compose` 文件，将 `"unitree_go2_autonomy_advance"` 替换为你希望运行的Agent代理名称。
```bash
command: ["unitree_go2_autonomy_advance"]
```

```bash
cd OM1
docker compose up om1 -d --no-build
```

- For unitree_sdk
```bash
cd unitree_sdk
docker compose up orchestrator -d --no-build
docker compose up om1_sensor -d --no-build
docker compose up watchdog -d --no-build
docker compose up zenoh_bridge -d --no-build
```

- For OM1-avatar
```bash
cd OM1-avatar
docker compose up om1_avatar -d --no-build
```

- For OM1-video-processor
```bash
cd OM1-video-processor
docker compose up -d
```

## 详细文档

更多详细内容请访问 [docs.openmind.org](https://docs.openmind.org/).

## 贡献

在提交 pull request 之前，请务必阅读 [docs.openmind.org](https://docs.openmind.org/)

## 许可证

本项目采用 MIT 许可证，这是一种宽松的自由软件许可证，允许用户自由使用、修改和分发软件。MIT 许可证因其简洁性和灵活性而广为人知，并被广泛使用和认可。通过使用 MIT 许可证，本项目旨在鼓励软件的协作、修改和分发。