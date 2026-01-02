"""
Conversation Agent - Voice to Text Processing with OpenAI/Google ASR

This example demonstrates how to use cloud-based AI endpoints for voice input processing.
Based on the conversation.mdx documentation example.

Usage:
    python agent.py

Requirements:
    - API key for OpenMind services (set in environment variable or .env file)
    - Microphone and speaker configured in system settings
"""

import json
import logging
import os
import sys
import time
from typing import Optional

# 配置日志系统 - 输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('runtime.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def process_asr_message(msg: str) -> None:
    """
    处理 ASR 识别结果消息
    
    Parameters
    ----------
    msg : str
        ASR 服务返回的消息（JSON 格式）
    """
    logger.info(f"Entering process_asr_message...")
    try:
        data = json.loads(msg)
        asr_reply = data.get("asr_reply", "")
        if asr_reply:
            logger.info(f"识别到的文本: {asr_reply}")
            print(f"\n[ASR 识别结果] {asr_reply}\n")
        else:
            logger.warning(f"ASR 消息中未找到 'asr_reply' 字段: {data}")
    except json.JSONDecodeError as e:
        logger.error(f"解析 ASR 消息 JSON 失败: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"处理 ASR 消息时发生错误: {e}", exc_info=True)
    finally:
        logger.info(f"Exiting process_asr_message...")


def main():
    """
    主函数：初始化并运行语音识别服务
    """
    logger.info("=" * 60)
    logger.info("Entering main...")
    logger.info("启动 Conversation Agent - 语音转文字服务")
    logger.info("=" * 60)
    
    try:
        # 获取 API Key
        api_key = os.getenv("OM_API_KEY") or os.getenv("OPENMIND_API_KEY")
        if not api_key:
            logger.error("未找到 API Key！请设置环境变量 OM_API_KEY 或 OPENMIND_API_KEY")
            logger.error("或者在项目根目录创建 .env 文件，添加: OM_API_KEY=your_api_key")
            raise ValueError("API Key 未设置")
        
        logger.info("API Key 已加载")
        
        # 导入必要的模块
        logger.info("正在导入 om1_speech 和 om1_utils 模块...")
        try:
            from om1_speech import AudioInputStream
            from om1_utils import ws
            logger.info("模块导入成功")
        except ImportError as e:
            logger.error(f"导入模块失败: {e}", exc_info=True)
            logger.error("请确保已安装 om1-modules 和 om1-utils 包")
            logger.error("安装命令: pip install om1-modules om1-utils")
            raise
        
        # 选择 ASR 服务端点
        # 可以使用 Google ASR 或 Riva ASR
        use_google_asr = os.getenv("USE_GOOGLE_ASR", "true").lower() == "true"
        
        if use_google_asr:
            # Google ASR 端点
            ws_url = f"wss://api.openmind.org/api/core/google/asr?api_key={api_key}"
            logger.info("使用 Google ASR 服务")
        else:
            # Riva ASR 端点
            ws_url = f"wss://api-asr.openmind.org?api_key={api_key}"
            logger.info("使用 Riva ASR 服务")
        
        logger.info(f"WebSocket URL: {ws_url.replace(api_key, '***')}")
        
        # 初始化 WebSocket 客户端
        logger.info("正在初始化 WebSocket 客户端...")
        ws_client = ws.Client(url=ws_url)
        logger.info("WebSocket 客户端初始化成功")
        
        # 初始化音频输入流
        logger.info("正在初始化音频输入流...")
        audio_stream_input = AudioInputStream(
            audio_data_callback=ws_client.send_message
        )
        logger.info("音频输入流初始化成功")
        
        # 注册消息回调
        logger.info("正在注册 ASR 消息回调...")
        ws_client.register_message_callback(process_asr_message)
        logger.info("消息回调注册成功")
        
        # 启动服务
        logger.info("正在启动 WebSocket 客户端...")
        ws_client.start()
        logger.info("WebSocket 客户端已启动")
        
        logger.info("正在启动音频输入流...")
        audio_stream_input.start()
        logger.info("音频输入流已启动")
        
        logger.info("=" * 60)
        logger.info("服务已启动！请对着麦克风说话...")
        logger.info("按 Ctrl+C 停止服务")
        logger.info("=" * 60)
        
        # 保持运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n收到停止信号，正在关闭服务...")
        finally:
            # 清理资源
            logger.info("正在停止音频输入流...")
            try:
                audio_stream_input.stop()
                logger.info("音频输入流已停止")
            except Exception as e:
                logger.error(f"停止音频输入流时出错: {e}", exc_info=True)
            
            logger.info("正在关闭 WebSocket 连接...")
            try:
                ws_client.stop()
                logger.info("WebSocket 连接已关闭")
            except Exception as e:
                logger.error(f"关闭 WebSocket 连接时出错: {e}", exc_info=True)
            
            logger.info("服务已完全停止")
            logger.info("Exiting main...")
    
    except ValueError as e:
        logger.error(f"配置错误: {e}", exc_info=True)
        sys.exit(1)
    except ImportError as e:
        logger.error(f"导入错误: {e}", exc_info=True)
        logger.error("请检查依赖库是否已正确安装")
        sys.exit(1)
    except Exception as e:
        logger.error(f"运行时发生未预期的错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

