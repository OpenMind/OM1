# OM1 Configuration Examples | OM1 配置示例集

<div align="center">

**Ready-to-use configuration examples for different use cases**  
**不同使用场景的即用配置示例**

[English](#english) | [中文](#中文)

</div>

---

<a name="english"></a>

## 📚 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Configuration Guide](#configuration-guide)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This directory contains a complete set of OM1 configuration examples designed for users of all skill levels. Each example includes:

- ✅ Detailed configuration file with bilingual comments
- ✅ Complete usage documentation
- ✅ Hardware and software requirements
- ✅ Customization tips
- ✅ Troubleshooting guide

---

## 🚀 Quick Start

### 1. Prerequisites

- **OM1 installed** ([Installation Guide](https://docs.openmind.org/installation))
- **OpenMind API Key** ([Get your key](https://portal.openmind.org))
- **Hardware** (varies by example - see individual requirements)

### 2. Set API Key

**Linux/macOS**:
```bash
export OPENMIND_API_KEY="your_api_key_here"
```

**Windows PowerShell**:
```powershell
$env:OPENMIND_API_KEY="your_api_key_here"
```

**Or create `.env` file**:
```bash
echo "OPENMIND_API_KEY=your_key" > .env
```

### 3. Run an Example

```bash
# Navigate to OM1 directory
cd OM1

# Copy example to config directory
cp config/examples/beginner-hello-world.json5 config/my-agent.json5

# Run
uv run src/run.py my-agent
```

---

## 📖 Examples

### 🎓 Beginner Examples

Perfect for first-time users and learning OM1 basics.

| Example | Description | Hardware | Time | Difficulty |
|---------|-------------|----------|------|-----------|
| **[beginner-hello-world](beginner-hello-world.json5)** | Simplest webcam application<br>最简单的摄像头应用 | Webcam | 5 min | ⭐ |
| **[beginner-chatbot](beginner-chatbot.json5)** | Voice chatbot with Chinese support<br>支持中文的语音聊天机器人 | Microphone | 10 min | ⭐⭐ |

### 🚀 Intermediate Examples

More advanced features and robot control.

| Example | Description | Hardware | Time | Difficulty |
|---------|-------------|----------|------|-----------|
| **[intermediate-object-detection](intermediate-object-detection.json5)** | Real-time object detection with YOLO<br>使用YOLO的实时物体检测 | Webcam | 15 min | ⭐⭐⭐ |
| **[intermediate-turtlebot-basic](intermediate-turtlebot-basic.json5)** | Voice-controlled TurtleBot<br>语音控制的TurtleBot | TurtleBot 4 + ROS2 | 20 min | ⭐⭐⭐ |

### 💎 Advanced Examples

Complex multimodal applications for production use.

| Example | Description | Hardware | Time | Difficulty |
|---------|-------------|----------|------|-----------|
| **[advanced-multimodal](advanced-multimodal.json5)** | Full autonomous agent with SLAM<br>带SLAM的完全自主智能体 | Quadruped + LiDAR + ROS2 | 30+ min | ⭐⭐⭐⭐⭐ |

---

## 🔧 Configuration Guide

### Basic Structure

All OM1 configurations follow this structure:

```json5
{
  "inputs": [/* Input modules */],
  "actions": [/* Output actions */],
  "llm": {/* Language model settings */},
  "system_prompt": "AI behavior instructions"
}
```

### Input Types

- **`webcam`** - Camera input for vision
- **`microphone`** - Audio input for speech
- **`lidar`** - LiDAR sensor for spatial awareness
- **`ros2_topic`** - Any ROS2 topic
- **`websocket`** - WebSocket connections

### Action Types

- **`text_output`** - Display text in console
- **`speech`** - Text-to-speech output
- **`ros2_movement`** - Robot movement commands
- **`ros2_navigation`** - Autonomous navigation
- **`avatar`** - Visual avatar with expressions
- **`websim`** - Web-based debugging interface

### LLM Providers

- **OpenAI** - GPT-4o, GPT-4o-mini
- **DeepSeek** - DeepSeek models
- **Custom** - Any OpenAI-compatible API

---

## ⚙️ Customization

### Adjust Processing Frequency

```json5
"frequency": 2.0,  // Process 2 times per second
```

**Recommended values**:
- `0.5` - Every 2 seconds (low resource)
- `1.0` - Every second (recommended)
- `2.0` - Twice per second (responsive)

### Modify AI Behavior

Change the `system_prompt` to customize behavior:

```json5
"system_prompt": "You are a helpful assistant that..."
```

### Add Safety Limits

For robot control, always set safety parameters:

```json5
"safety": {
  "max_linear_speed": 0.5,  // m/s
  "collision_detection": true,
  "emergency_stop": true
}
```

---

## 🐛 Troubleshooting

### Camera Not Working

1. Check camera permissions
2. Try different camera index:
   ```json5
   "camera_index": 1,  // Try 1, 2, etc.
   ```

### API Key Error

1. Verify your API key:
   ```bash
   echo $OPENMIND_API_KEY
   ```
2. Check `.env` file exists and is correct

### Slow Performance

1. Reduce frequency:
   ```json5
   "frequency": 0.5,
   ```
2. Use smaller model:
   ```json5
   "model": "gpt-4o-mini",
   ```

### ROS2 Connection Issues

1. Check ROS_DOMAIN_ID:
   ```bash
   echo $ROS_DOMAIN_ID
   ```
2. Verify topics:
   ```bash
   ros2 topic list
   ```

---

## 💡 Best Practices

### 1. Start Simple
Begin with beginner examples, then increase complexity.

### 2. Test Safely
Always test robot movements in a safe, open area.

### 3. Monitor Resources
Use WebSim (http://localhost:8000/) to monitor performance.

### 4. Iterate
Start with default settings, then fine-tune based on results.

### 5. Document Changes
Keep notes of what works well for your specific use case.

---

## 📚 Additional Resources

- **Full Documentation**: https://docs.openmind.org
- **API Reference**: https://docs.openmind.org/api
- **Community Forum**: https://github.com/OpenMind/OM1/discussions
- **Discord**: https://discord.gg/openmind
- **Email Support**: ask@openmind.org

---

## 🤝 Contributing

Have a useful configuration? Share it!

1. Create your example configuration
2. Add detailed documentation
3. Test thoroughly
4. Submit a Pull Request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

---

## 📋 Comparison Table

For a detailed comparison of all examples, see [COMPARISON.md](COMPARISON.md).

---

<a name="中文"></a>

# OM1 配置示例集

## 📚 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [示例列表](#示例列表)
- [配置指南](#配置指南)
- [自定义](#自定义)
- [故障排除](#故障排除-1)
- [贡献](#贡献)

---

## 🎯 概述

本目录包含一套完整的 OM1 配置示例，专为各种技能水平的用户设计。每个示例包括：

- ✅ 带双语注释的详细配置文件
- ✅ 完整的使用文档
- ✅ 硬件和软件需求
- ✅ 自定义技巧
- ✅ 故障排除指南

---

## 🚀 快速开始

### 1. 前提条件

- **已安装 OM1** ([安装指南](https://docs.openmind.org/installation))
- **OpenMind API 密钥** ([获取密钥](https://portal.openmind.org))
- **硬件** (根据示例而异 - 查看各自需求)

### 2. 设置 API 密钥

**Linux/macOS**:
```bash
export OPENMIND_API_KEY="你的密钥"
```

**Windows PowerShell**:
```powershell
$env:OPENMIND_API_KEY="你的密钥"
```

**或创建 `.env` 文件**:
```bash
echo "OPENMIND_API_KEY=你的密钥" > .env
```

### 3. 运行示例

```bash
# 进入 OM1 目录
cd OM1

# 复制示例到配置目录
cp config/examples/beginner-hello-world.json5 config/my-agent.json5

# 运行
uv run src/run.py my-agent
```

---

## 📖 示例列表

### 🎓 新手示例

适合首次使用者和学习 OM1 基础。

| 示例 | 描述 | 硬件 | 时间 | 难度 |
|------|------|------|------|------|
| **[beginner-hello-world](beginner-hello-world.json5)** | 最简单的摄像头应用 | 摄像头 | 5分钟 | ⭐ |
| **[beginner-chatbot](beginner-chatbot.json5)** | 支持中文的语音聊天机器人 | 麦克风 | 10分钟 | ⭐⭐ |

### 🚀 中级示例

更高级的功能和机器人控制。

| 示例 | 描述 | 硬件 | 时间 | 难度 |
|------|------|------|------|------|
| **[intermediate-object-detection](intermediate-object-detection.json5)** | 使用YOLO的实时物体检测 | 摄像头 | 15分钟 | ⭐⭐⭐ |
| **[intermediate-turtlebot-basic](intermediate-turtlebot-basic.json5)** | 语音控制的TurtleBot | TurtleBot 4 + ROS2 | 20分钟 | ⭐⭐⭐ |

### 💎 高级示例

用于生产环境的复杂多模态应用。

| 示例 | 描述 | 硬件 | 时间 | 难度 |
|------|------|------|------|------|
| **[advanced-multimodal](advanced-multimodal.json5)** | 带SLAM的完全自主智能体 | 四足机器人 + 激光雷达 + ROS2 | 30+分钟 | ⭐⭐⭐⭐⭐ |

---

## 🔧 配置指南

### 基本结构

所有 OM1 配置都遵循此结构：

```json5
{
  "inputs": [/* 输入模块 */],
  "actions": [/* 输出动作 */],
  "llm": {/* 语言模型设置 */},
  "system_prompt": "AI 行为指令"
}
```

### 输入类型

- **`webcam`** - 摄像头输入用于视觉
- **`microphone`** - 音频输入用于语音
- **`lidar`** - 激光雷达用于空间感知
- **`ros2_topic`** - 任何 ROS2 话题
- **`websocket`** - WebSocket 连接

### 动作类型

- **`text_output`** - 在控制台显示文字
- **`speech`** - 文字转语音输出
- **`ros2_movement`** - 机器人移动命令
- **`ros2_navigation`** - 自主导航
- **`avatar`** - 带表情的虚拟形象
- **`websim`** - 基于 Web 的调试界面

### LLM 提供商

- **OpenAI** - GPT-4o, GPT-4o-mini
- **DeepSeek** - DeepSeek 模型
- **Custom** - 任何 OpenAI 兼容 API

---

## ⚙️ 自定义

### 调整处理频率

```json5
"frequency": 2.0,  // 每秒处理2次
```

**推荐值**:
- `0.5` - 每2秒一次（低资源）
- `1.0` - 每秒一次（推荐）
- `2.0` - 每秒两次（响应快）

### 修改 AI 行为

更改 `system_prompt` 以自定义行为：

```json5
"system_prompt": "你是一个有帮助的助手..."
```

### 添加安全限制

对于机器人控制，始终设置安全参数：

```json5
"safety": {
  "max_linear_speed": 0.5,  // 米/秒
  "collision_detection": true,
  "emergency_stop": true
}
```

---

## 🐛 故障排除

### 摄像头不工作

1. 检查摄像头权限
2. 尝试不同的摄像头索引：
   ```json5
   "camera_index": 1,  // 尝试 1, 2 等
   ```

### API 密钥错误

1. 验证您的 API 密钥：
   ```bash
   echo $OPENMIND_API_KEY
   ```
2. 检查 `.env` 文件是否存在且正确

### 性能慢

1. 降低频率：
   ```json5
   "frequency": 0.5,
   ```
2. 使用更小的模型：
   ```json5
   "model": "gpt-4o-mini",
   ```

### ROS2 连接问题

1. 检查 ROS_DOMAIN_ID：
   ```bash
   echo $ROS_DOMAIN_ID
   ```
2. 验证话题：
   ```bash
   ros2 topic list
   ```

---

## 💡 最佳实践

### 1. 从简单开始
从新手示例开始，然后增加复杂性。

### 2. 安全测试
始终在安全、开放的区域测试机器人移动。

### 3. 监控资源
使用 WebSim (http://localhost:8000/) 监控性能。

### 4. 迭代改进
从默认设置开始，然后根据结果微调。

### 5. 记录更改
记录哪些设置对您的特定用例效果好。

---

## 📚 额外资源

- **完整文档**: https://docs.openmind.org
- **API 参考**: https://docs.openmind.org/api
- **社区论坛**: https://github.com/OpenMind/OM1/discussions
- **Discord**: https://discord.gg/openmind
- **邮件支持**: ask@openmind.org

---

## 🤝 贡献

有有用的配置？分享它！

1. 创建您的示例配置
2. 添加详细文档
3. 充分测试
4. 提交 Pull Request

详见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

---

## 📋 对比表

有关所有示例的详细对比，请参阅 [COMPARISON.md](COMPARISON.md)。

---

<div align="center">

**Happy building with OM1!** 🚀  
**用 OM1 愉快地构建！** 🚀

[⭐ Star us on GitHub](https://github.com/OpenMind/OM1) | [🐛 Report Issues](https://github.com/OpenMind/OM1/issues) | [💬 Join Discord](https://discord.gg/openmind)

Made with ❤️ by the OpenMind Community

</div>

