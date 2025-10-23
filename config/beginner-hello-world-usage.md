# Beginner Hello World Configuration | 新手 Hello World 配置

## 📝 Overview | 概述

**English**: The simplest OM1 configuration to get you started. This example uses your webcam to capture images and an AI model to describe what it sees.

**中文**: 最简单的 OM1 配置，帮助您入门。此示例使用摄像头捕获图像，并使用 AI 模型描述它看到的内容。

---

## 🎯 What You'll Learn | 您将学到什么

- ✅ Basic OM1 configuration structure | 基本 OM1 配置结构
- ✅ How to use webcam input | 如何使用摄像头输入
- ✅ How to configure text output | 如何配置文字输出
- ✅ How to set up an LLM | 如何设置语言模型
- ✅ How to write system prompts | 如何编写系统提示词

---

## 📋 Requirements | 需求

### Hardware | 硬件
- 💻 Computer with webcam | 带摄像头的电脑
- 🌐 Internet connection | 网络连接

### Software | 软件
- ✅ Python 3.8+ | Python 3.8+
- ✅ OM1 installed | 已安装 OM1
- ✅ uv package manager | uv 包管理器

### API Key | API 密钥
- 🔑 OpenMind API Key (get from https://portal.openmind.org)
- 🔑 OpenMind API 密钥（从 https://portal.openmind.org 获取）

---

## 🚀 Quick Start | 快速开始

### Step 1: Set API Key | 步骤 1：设置 API 密钥

**Windows PowerShell**:
```powershell
$env:OPENMIND_API_KEY="your_api_key_here"
```

**Linux/macOS**:
```bash
export OPENMIND_API_KEY="your_api_key_here"
```

**Or add to `.env` file | 或添加到 `.env` 文件**:
```bash
echo "OPENMIND_API_KEY=your_api_key_here" >> .env
```

---

### Step 2: Copy Configuration | 步骤 2：复制配置

```bash
# Navigate to OM1 directory | 进入 OM1 目录
cd OM1

# Copy the example configuration | 复制示例配置
cp config/examples/beginner-hello-world.json5 config/my-first-agent.json5
```

---

### Step 3: Run | 步骤 3：运行

```bash
uv run src/run.py my-first-agent
```

---

## 💡 What to Expect | 预期效果

After running the command, you should see:

运行命令后，您应该看到：

1. **Camera Activation | 摄像头激活**
   - Your webcam light turns on
   - 您的摄像头指示灯亮起

2. **Console Output | 控制台输出**
   ```
   [INFO] Starting OM1 agent...
   [INFO] Webcam initialized
   [INFO] Capturing frame...
   [INFO] AI Response: I can see a desk with a laptop, a coffee mug, and some books. The laptop screen is displaying code.
   ```

3. **Continuous Updates | 持续更新**
   - Every second, the AI describes what it sees
   - 每秒钟，AI 描述它看到的内容

---

## 🔧 Customization | 自定义

### Adjust Camera Frequency | 调整摄像头频率

Edit the configuration file:

编辑配置文件：

```json5
"frequency": 2.0,  // Capture 2 frames per second | 每秒捕获2帧
```

**Recommended values | 推荐值**:
- `0.5` - Every 2 seconds (very slow, low resource)
  每2秒一次（非常慢，低资源）
- `1.0` - Every second (recommended for beginners)
  每秒一次（推荐新手使用）
- `2.0` - Twice per second (more responsive)
  每秒两次（响应更快）

---

### Change AI Model | 更改 AI 模型

```json5
"llm": {
  "model": "gpt-4o",  // More capable but more expensive
                      // 更强大但更昂贵
}
```

**Available models | 可用模型**:
- `gpt-4o-mini` - Cheap, fast, good for simple tasks
  便宜、快速，适合简单任务
- `gpt-4o` - More capable, higher cost
  更强大，成本更高
- `gpt-4` - Most capable, highest cost
  最强大，成本最高

---

### Modify System Prompt | 修改系统提示词

Make the AI more creative or specific:

使 AI 更有创意或更具体：

```json5
"system_prompt": "You are an enthusiastic art critic. Describe what you see as if it were a piece of art, focusing on colors, composition, and mood."
```

Or in Chinese | 或中文版本：

```json5
"system_prompt": "你是一个热情的艺术评论家。把你看到的东西当作艺术品来描述，关注色彩、构图和氛围。"
```

---

## 🐛 Troubleshooting | 故障排除

### Camera Not Found | 找不到摄像头

**Problem**: Error: "No camera detected"

**Solution**:
1. Check camera permissions
   检查摄像头权限
2. Try different camera index:
   尝试不同的摄像头索引：
   ```json5
   "camera_index": 1,  // Try 1, 2, etc. | 尝试 1, 2 等
   ```

---

### API Key Error | API 密钥错误

**Problem**: Error: "Invalid API key"

**Solution**:
1. Verify your API key is correct
   验证您的 API 密钥是否正确
2. Check environment variable:
   检查环境变量：
   ```bash
   echo $OPENMIND_API_KEY  # Linux/macOS
   echo $env:OPENMIND_API_KEY  # Windows PowerShell
   ```

---

### Slow Performance | 性能慢

**Problem**: Agent takes too long to respond

**Solution**:
1. Reduce camera frequency:
   降低摄像头频率：
   ```json5
   "frequency": 0.5,  // Slower = less resource intensive
   ```
2. Use smaller model:
   使用更小的模型：
   ```json5
   "model": "gpt-4o-mini",
   ```

---

## 📚 What's Next? | 下一步？

Congratulations! You've run your first OM1 agent!

恭喜！您已运行了第一个 OM1 智能体！

### Try These Next | 接下来尝试这些：

1. **Add Voice** | 添加语音
   - Try `beginner-chatbot.json5`
   - 尝试 `beginner-chatbot.json5`

2. **Object Detection** | 物体检测
   - Try `intermediate-object-detection.json5`
   - 尝试 `intermediate-object-detection.json5`

3. **Create Your Own** | 创建您自己的
   - Modify the system prompt
   - 修改系统提示词
   - Experiment with different settings
   - 尝试不同的设置

---

## 💬 Get Help | 获取帮助

- **Documentation**: https://docs.openmind.org
- **Discord**: https://discord.gg/openmind
- **Issues**: https://github.com/OpenMind/OM1/issues
- **Email**: ask@openmind.org

---

## 🎉 Congratulations! | 恭喜！

You've successfully set up and run your first OM1 configuration!

您已成功设置并运行了第一个 OM1 配置！

**Share your experience | 分享您的体验**:
- Tweet about it: @openmind_agi
- 发推文：@openmind_agi
- Join the community
- 加入社区

---

**Happy building with OM1!** 🚀

**用 OM1 愉快地构建！** 🚀

