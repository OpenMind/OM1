# Conversation Agent Example

这是一个基于官方文档 `OM1/docs/examples/conversation.mdx` 复原的 Conversation Agent 示例代码。

## 功能说明

该示例演示了如何使用云端的 AI 端点进行语音输入处理，实现语音转文字（ASR）功能。支持 Google ASR 和 Riva ASR 两种服务。

## 运行要求

### 依赖库

```bash
pip install pyaudio soundfile sounddevice websockets aiohttp python-dotenv numpy
pip install git+https://github.com/OpenMind/om1-modules.git@eed03c9ccaf00641c706404e32f5b161804512a8
```

### 环境配置

1. **API Key 设置**：
   ```bash
   # Windows PowerShell
   $env:OM_API_KEY="your_api_key"
   
   # Linux/Mac
   export OM_API_KEY=your_api_key
   ```

2. **音频设备**：
   - 确保系统已配置默认麦克风（音频输入）
   - 确保系统已配置默认扬声器（音频输出）

## 运行命令

```bash
python agent.py
```

## 使用说明

- 程序启动后，会通过默认麦克风采集音频
- 音频数据通过 WebSocket 发送到云端 ASR 服务
- 识别结果会实时显示在控制台
- 按 `Ctrl+C` 停止服务

## 日志

程序运行日志会同时输出到：
- 控制台（标准输出）
- `runtime.log` 文件（当前目录）

