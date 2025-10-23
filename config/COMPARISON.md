# Configuration Examples Comparison | 配置示例对比

Quick reference guide for choosing the right configuration example.  
快速参考指南，帮助选择合适的配置示例。

---

## 📊 Feature Matrix | 功能矩阵

| Example<br>示例 | Camera<br>摄像头 | Mic<br>麦克风 | LiDAR<br>激光雷达 | Robot<br>机器人 | Speech<br>语音 | Navigation<br>导航 | Detection<br>检测 | Avatar<br>形象 | Difficulty<br>难度 |
|---------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **beginner-hello-world** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⭐ |
| **beginner-chatbot** | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ⭐⭐ |
| **intermediate-object-detection** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ⭐⭐⭐ |
| **intermediate-turtlebot-basic** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ⭐⭐⭐ |
| **advanced-multimodal** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |

---

## 💻 Resource Requirements | 资源需求

### Hardware | 硬件

| Example<br>示例 | RAM | CPU | GPU | Robot Hardware<br>机器人硬件 |
|---------------|-----|-----|-----|---------------------|
| **beginner-hello-world** | 2GB | Low<br>低 | No<br>否 | None<br>无 |
| **beginner-chatbot** | 2GB | Low<br>低 | No<br>否 | None<br>无 |
| **intermediate-object-detection** | 4GB | Medium<br>中 | Optional<br>可选 | None<br>无 |
| **intermediate-turtlebot-basic** | 4GB | Medium<br>中 | Optional<br>可选 | TurtleBot 4 + ROS2 |
| **advanced-multimodal** | 8GB+ | High<br>高 | Recommended<br>推荐 | Quadruped + LiDAR + ROS2<br>四足 + 激光雷达 + ROS2 |

### API Costs | API 成本

Estimated API costs per hour of operation (approximate).  
每小时运行的估算 API 成本（近似值）。

| Example<br>示例 | Model<br>模型 | Requests/Hour<br>请求/小时 | Est. Cost/Hour<br>估算成本/小时 |
|---------------|------|--------------|----------------|
| **beginner-hello-world** | gpt-4o-mini | ~3,600 | $0.05 - $0.10 |
| **beginner-chatbot** | gpt-4o | ~varies<br>变化 | $0.10 - $0.50 |
| **intermediate-object-detection** | gpt-4o | ~7,200 | $0.15 - $0.30 |
| **intermediate-turtlebot-basic** | gpt-4o | ~varies<br>变化 | $0.20 - $0.60 |
| **advanced-multimodal** | gpt-4o | ~7,200+ | $0.30 - $1.00+ |

*Note: Costs vary based on usage patterns and prompt/response lengths.*  
*注意：成本因使用模式和提示/响应长度而异。*

---

## 🎯 Use Cases | 使用场景

### If you want to... | 如果您想要...

| Goal | Recommended Example |
|------|---------------------|
| Learn OM1 basics<br>学习 OM1 基础 | **beginner-hello-world** |
| Build a voice assistant<br>构建语音助手 | **beginner-chatbot** |
| Detect and track objects<br>检测和追踪物体 | **intermediate-object-detection** |
| Control a robot with voice<br>用语音控制机器人 | **intermediate-turtlebot-basic** |
| Build a fully autonomous agent<br>构建完全自主的智能体 | **advanced-multimodal** |
| Test computer vision<br>测试计算机视觉 | **intermediate-object-detection** |
| Prototype quickly<br>快速原型 | **beginner-hello-world** |
| Production deployment<br>生产部署 | **advanced-multimodal** |

---

## ⏱️ Time Requirements | 时间需求

### Setup Time | 设置时间

| Example<br>示例 | First Time<br>首次 | Subsequent<br>后续 |
|---------------|----------|-----------|
| **beginner-hello-world** | 5-10 min | 1 min |
| **beginner-chatbot** | 10-15 min | 2 min |
| **intermediate-object-detection** | 15-20 min | 3 min |
| **intermediate-turtlebot-basic** | 30-45 min | 5 min |
| **advanced-multimodal** | 60+ min | 10 min |

### Learning Curve | 学习曲线

| Example<br>示例 | Understanding<br>理解 | Customization<br>定制 | Debugging<br>调试 |
|---------------|-------------|--------------|----------|
| **beginner-hello-world** | 10 min | Easy<br>容易 | Easy<br>容易 |
| **beginner-chatbot** | 15 min | Easy<br>容易 | Medium<br>中等 |
| **intermediate-object-detection** | 30 min | Medium<br>中等 | Medium<br>中等 |
| **intermediate-turtlebot-basic** | 45 min | Medium<br>中等 | Hard<br>困难 |
| **advanced-multimodal** | 2+ hours | Hard<br>困难 | Hard<br>困难 |

---

## 🔧 Customization Complexity | 自定义复杂度

| Example<br>示例 | Parameters<br>参数数量 | Customization Points<br>自定义点 | Flexibility<br>灵活性 |
|---------------|-----------|------------------|-----------|
| **beginner-hello-world** | ~10 | Low<br>低 | Basic<br>基础 |
| **beginner-chatbot** | ~15 | Medium<br>中 | Good<br>良好 |
| **intermediate-object-detection** | ~25 | Medium<br>中 | Good<br>良好 |
| **intermediate-turtlebot-basic** | ~40 | High<br>高 | Very Good<br>很好 |
| **advanced-multimodal** | ~80+ | Very High<br>很高 | Excellent<br>优秀 |

---

## 🎓 Skill Level Requirements | 技能水平要求

### Prerequisites | 前提条件

| Example<br>示例 | Python | ROS2 | Computer Vision<br>计算机视觉 | Robotics<br>机器人学 |
|---------------|--------|------|--------------|---------|
| **beginner-hello-world** | Basic<br>基础 | None<br>无 | None<br>无 | None<br>无 |
| **beginner-chatbot** | Basic<br>基础 | None<br>无 | None<br>无 | None<br>无 |
| **intermediate-object-detection** | Intermediate<br>中级 | None<br>无 | Basic<br>基础 | None<br>无 |
| **intermediate-turtlebot-basic** | Intermediate<br>中级 | Basic<br>基础 | Basic<br>基础 | Basic<br>基础 |
| **advanced-multimodal** | Advanced<br>高级 | Intermediate<br>中级 | Intermediate<br>中级 | Intermediate<br>中级 |

---

## 📈 Performance Characteristics | 性能特征

### Processing Speed | 处理速度

| Example<br>示例 | Latency<br>延迟 | Throughput<br>吞吐量 | Real-time<br>实时性 |
|---------------|---------|-----------|----------|
| **beginner-hello-world** | ~1-2s | 1 fps | No<br>否 |
| **beginner-chatbot** | ~2-5s | Varies<br>变化 | Yes<br>是 |
| **intermediate-object-detection** | ~0.5-1s | 2 fps | Yes<br>是 |
| **intermediate-turtlebot-basic** | ~1-3s | 1 fps | Yes<br>是 |
| **advanced-multimodal** | ~0.3-1s | 2+ fps | Yes<br>是 |

### Resource Utilization | 资源利用率

| Example<br>示例 | CPU Usage<br>CPU使用 | Memory<br>内存 | Network<br>网络 | GPU<br>GPU |
|---------------|---------|--------|---------|-----|
| **beginner-hello-world** | 5-10% | Low<br>低 | Low<br>低 | N/A |
| **beginner-chatbot** | 5-15% | Low<br>低 | Medium<br>中 | N/A |
| **intermediate-object-detection** | 30-50% | Medium<br>中 | Low<br>低 | Optional<br>可选 |
| **intermediate-turtlebot-basic** | 20-40% | Medium<br>中 | Medium<br>中 | Optional<br>可选 |
| **advanced-multimodal** | 50-80% | High<br>高 | High<br>高 | Recommended<br>推荐 |

---

## 🛡️ Safety Features | 安全功能

| Example<br>示例 | Collision Avoidance<br>避碰 | Emergency Stop<br>紧急停止 | Speed Limits<br>速度限制 | Boundary Protection<br>边界保护 |
|---------------|-----------------|-------------|-----------|------------------|
| **beginner-hello-world** | N/A | N/A | N/A | N/A |
| **beginner-chatbot** | N/A | N/A | N/A | N/A |
| **intermediate-object-detection** | N/A | N/A | N/A | N/A |
| **intermediate-turtlebot-basic** | ✅ Yes<br>是 | ✅ Yes<br>是 | ✅ Yes<br>是 | ⚠️ Optional<br>可选 |
| **advanced-multimodal** | ✅✅ Advanced<br>高级 | ✅ Yes<br>是 | ✅ Yes<br>是 | ✅ Yes<br>是 |

---

## 🌟 Best For | 最适合

### By User Type | 按用户类型

| User Type<br>用户类型 | Best Example<br>最佳示例 | Alternative<br>备选 |
|----------|-------------|-----------|
| Complete beginners<br>完全新手 | beginner-hello-world | beginner-chatbot |
| Students<br>学生 | beginner-chatbot | intermediate-object-detection |
| Hobbyists<br>爱好者 | intermediate-object-detection | intermediate-turtlebot-basic |
| Researchers<br>研究人员 | advanced-multimodal | intermediate-turtlebot-basic |
| Professional developers<br>专业开发者 | advanced-multimodal | All<br>全部 |

### By Project Type | 按项目类型

| Project Type<br>项目类型 | Best Example<br>最佳示例 |
|-------------|-------------|
| Learning project<br>学习项目 | beginner-hello-world |
| Demo/Presentation<br>演示/展示 | intermediate-object-detection |
| Voice interface<br>语音界面 | beginner-chatbot |
| Home automation<br>家庭自动化 | intermediate-turtlebot-basic |
| Research platform<br>研究平台 | advanced-multimodal |
| Production system<br>生产系统 | advanced-multimodal |

---

## 📊 Comparison Summary | 对比总结

### Quick Decision Guide | 快速决策指南

**Choose beginner-hello-world if:**
- You're new to OM1
- You want the simplest possible setup
- You don't have specialized hardware

**选择 beginner-hello-world 如果：**
- 您是 OM1 新手
- 您想要最简单的设置
- 您没有专用硬件

---

**Choose beginner-chatbot if:**
- You want voice interaction
- You're building a conversational assistant
- You need Chinese language support

**选择 beginner-chatbot 如果：**
- 您想要语音交互
- 您正在构建对话助手
- 您需要中文支持

---

**Choose intermediate-object-detection if:**
- You need computer vision capabilities
- You want to track objects in real-time
- You're exploring AI vision applications

**选择 intermediate-object-detection 如果：**
- 您需要计算机视觉能力
- 您想要实时追踪物体
- 您正在探索 AI 视觉应用

---

**Choose intermediate-turtlebot-basic if:**
- You have a TurtleBot 4 robot
- You want voice-controlled robot movement
- You're learning robotics with ROS2

**选择 intermediate-turtlebot-basic 如果：**
- 您有 TurtleBot 4 机器人
- 您想要语音控制机器人移动
- 您正在学习 ROS2 机器人技术

---

**Choose advanced-multimodal if:**
- You need a fully autonomous agent
- You have advanced robotic hardware
- You're building a production system
- You want maximum capabilities

**选择 advanced-multimodal 如果：**
- 您需要完全自主的智能体
- 您有高级机器人硬件
- 您正在构建生产系统
- 您想要最大能力

---

## 🔄 Upgrade Path | 升级路径

Recommended progression for learning:

学习的推荐进展路径：

```
beginner-hello-world 
    ↓
beginner-chatbot
    ↓
intermediate-object-detection
    ↓
intermediate-turtlebot-basic
    ↓
advanced-multimodal
```

Or choose your path based on hardware:

或根据硬件选择您的路径：

```
Have webcam only:
beginner-hello-world → intermediate-object-detection

Have microphone:
beginner-chatbot → (add webcam) → intermediate-turtlebot-basic

Have robot:
intermediate-turtlebot-basic → advanced-multimodal
```

---

## 📞 Need Help Choosing? | 需要帮助选择？

Ask yourself these questions:

问自己这些问题：

1. **What hardware do I have?**
   - Just a computer → beginner examples
   - Robot with ROS2 → intermediate/advanced

2. **What's my goal?**
   - Learn OM1 → beginner examples
   - Build something useful → intermediate examples
   - Production system → advanced example

3. **How much time do I have?**
   - < 15 minutes → beginner-hello-world
   - 30-60 minutes → intermediate examples
   - Multiple hours → advanced-multimodal

4. **What's my skill level?**
   - New to robotics → beginner examples
   - Some experience → intermediate examples
   - Expert → advanced-multimodal

Still unsure? Start with **beginner-hello-world** - you can always upgrade!

还不确定？从 **beginner-hello-world** 开始 - 您随时可以升级！

---

<div align="center">

**Happy Choosing!** 🎯  
**祝选择愉快！** 🎯

[Back to Examples](README.md) | [Get Help](https://github.com/OpenMind/OM1/issues)

</div>

