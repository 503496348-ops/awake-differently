"""
别样觉醒 — 数字人引擎
融合自 LiveTalking (8.1K⭐, Apache-2.0) 的实时流式数字人架构。

将人格模拟器的文本回复，通过 LiveTalking 驱动数字人实时说话。
支持 WebRTC 浏览器预览 / RTMP 推流 / 虚拟摄像头 / MP4录制。

数据流：
  persona_simulator(生成回复) → digital_avatar(发送到LiveTalking) → TTS → 口型同步 → 视频输出

依赖：
  - LiveTalking 服务（需单独部署启动）
  - requests 库

作者：AtomCollide-智械工坊
融合来源：lipku/LiveTalking (https://github.com/lipku/LiveTalking)
"""

import os
import json
import time
import requests
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum


# ── 配置 ──────────────────────────────────────────────────────────────────────

LIVETALKING_DEFAULT_HOST = "http://localhost:8010"

# LiveTalking 支持的数字人模型
class AvatarModel(str, Enum):
    WAV2LIP = "wav2lip"           # 轻量，RTX 3060即可，60FPS
    MUSETALK = "musetalk"         # 高质量，需RTX 3080Ti+
    ULTRALIGHT = "ultralight"     # 极轻量，移动端友好
    ERNERF = "ernerf"             # NeRF 3D，最高质量

# LiveTalking 支持的 TTS 引擎
class TTSEngine(str, Enum):
    EDGE = "edge"                 # 免费，300+音色
    SOVITS = "sovits"             # GPT-SoVITS 声音克隆
    COSYVOICE = "cosyvoice"      # 阿里CosyVoice
    FISH = "fish"                 # FishAudio
    DOUBAO = "doubao"             # 豆包TTS
    AZURE = "azure"               # Azure TTS
    TENCENT = "tencent"           # 腾讯云TTS

# LiveTalking 支持的输出方式
class OutputMode(str, Enum):
    WEBRTC = "webrtc"             # 浏览器实时预览
    RTMP = "rtmp"                 # 推流到直播平台
    VIRTUAL_CAM = "virtualcam"    # 虚拟摄像头
    RECORD = "record"             # 录制MP4


@dataclass
class AvatarConfig:
    """数字人配置"""
    model: AvatarModel = AvatarModel.WAV2LIP
    avatar_id: str = "wav2lip256_avatar1"  # 预训练形象ID
    tts_engine: TTSEngine = TTSEngine.EDGE
    tts_voice: str = "zh-CN-XiaoxiaoNeural"  # TTS音色
    output_mode: OutputMode = OutputMode.WEBRTC
    host: str = LIVETALKING_DEFAULT_HOST

    # LLM配置（LiveTalking内置chat模式）
    llm_enabled: bool = False  # True=用LiveTalking内置LLM, False=用外部persona_simulator
    llm_model: str = "qwen-plus"
    llm_api_key: str = ""

    # 高级配置
    interrupt_enabled: bool = True   # 支持打断
    custom_config: str = ""          # 动作编排JSON


@dataclass
class SessionState:
    """会话状态"""
    session_id: str = ""
    connected: bool = False
    is_speaking: bool = False
    message_count: int = 0
    created_at: float = 0.0


# ── LiveTalking API 客户端 ───────────────────────────────────────────────────

class LiveTalkingClient:
    """
    LiveTalking API 客户端。

    封装 LiveTalking 的 HTTP/WebRTC 接口，提供简洁的数字人驱动能力。
    融合自 lipku/LiveTalking server/routes.py 的 API 设计。

    用法：
        client = LiveTalkingClient("http://localhost:8010")
        session = client.connect()
        client.speak(session, "你好，我是陈龙")
        client.speak(session, "今天我们聊聊AI变现", interrupt=True)
    """

    def __init__(self, host: str = LIVETALKING_DEFAULT_HOST, timeout: int = 30):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._sessions: Dict[str, SessionState] = {}

    def health_check(self) -> bool:
        """检查 LiveTalking 服务是否可用"""
        try:
            resp = requests.get(f"{self.host}/", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def connect(self, avatar_id: str = "wav2lip256_avatar1") -> SessionState:
        """
        建立 WebRTC 连接，获取 session_id。

        Args:
            avatar_id: 数字人形象ID

        Returns:
            SessionState 包含 session_id
        """
        # WebRTC offer (简化版，实际需要浏览器端SDP)
        # 这里用 HTTP API 模式获取 session
        resp = requests.post(
            f"{self.host}/offer",
            json={
                "sdp": "",  # 实际使用时由浏览器提供
                "type": "offer",
                "avatar": avatar_id,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        session = SessionState(
            session_id=data.get("sessionid", ""),
            connected=True,
            created_at=time.time(),
        )
        self._sessions[session.session_id] = session
        return session

    def speak(
        self,
        session_id: str,
        text: str,
        interrupt: bool = False,
        tts_config: Optional[Dict] = None,
    ) -> Dict:
        """
        驱动数字人说话。

        Args:
            session_id: 会话ID
            text: 要说的文本
            interrupt: 是否打断当前播报
            tts_config: TTS透传配置（voice, emotion等）

        Returns:
            API响应
        """
        payload = {
            "sessionid": session_id,
            "text": text,
            "type": "echo",  # echo=直接复读, chat=触发LLM
            "interrupt": interrupt,
        }
        if tts_config:
            payload["tts"] = tts_config

        resp = requests.post(
            f"{self.host}/human",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()

        if session_id in self._sessions:
            self._sessions[session_id].message_count += 1
            self._sessions[session_id].is_speaking = True

        return result

    def speak_chat(
        self,
        session_id: str,
        text: str,
        interrupt: bool = False,
    ) -> Dict:
        """
        用 LiveTalking 内置 LLM 对话模式驱动数字人。

        与 speak() 的区别：speak() 是直接复读文本，
        speak_chat() 会先经过 LLM 生成回复，再驱动数字人说话。
        """
        payload = {
            "sessionid": session_id,
            "text": text,
            "type": "chat",
            "interrupt": interrupt,
        }
        resp = requests.post(
            f"{self.host}/human",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def speak_audio(self, session_id: str, audio_path: str) -> Dict:
        """
        用音频文件驱动数字人说话。

        Args:
            session_id: 会话ID
            audio_path: 音频文件路径
        """
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{self.host}/humanaudio",
                files={"file": f},
                data={"sessionid": session_id},
                timeout=self.timeout,
            )
        resp.raise_for_status()
        return resp.json()

    def interrupt(self, session_id: str) -> Dict:
        """打断当前播报"""
        resp = requests.post(
            f"{self.host}/interrupt_talk",
            json={"sessionid": session_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def is_speaking(self, session_id: str) -> bool:
        """查询数字人是否正在说话"""
        resp = requests.post(
            f"{self.host}/is_speaking",
            json={"sessionid": session_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("is_speaking", False)

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """获取会话状态"""
        return self._sessions.get(session_id)


# ── 数字人管理器 ──────────────────────────────────────────────────────────────

class DigitalAvatarManager:
    """
    数字人管理器 — 桥接人格模拟器与 LiveTalking。

    将别样觉醒的人格模拟输出，转化为数字人实时说话视频。

    用法：
        manager = DigitalAvatarManager(config)
        manager.start()
        manager.say("你好，我是陈龙的数字分身")
        manager.say_from_persona("你觉得这个方案怎么样？")
    """

    def __init__(self, config: Optional[AvatarConfig] = None):
        self.config = config or AvatarConfig()
        self.client = LiveTalkingClient(host=self.config.host)
        self._session: Optional[SessionState] = None

    def start(self) -> bool:
        """启动数字人会话"""
        if not self.client.health_check():
            print(f"❌ LiveTalking 服务未运行: {self.config.host}")
            print(f"   请先启动: python app.py --transport webrtc --model {self.config.model.value}")
            return False

        self._session = self.client.connect(avatar_id=self.config.avatar_id)
        print(f"✅ 数字人已连接: session={self._session.session_id}")
        return True

    def say(self, text: str, interrupt: bool = False) -> bool:
        """让数字人说指定文本"""
        if not self._session:
            print("❌ 数字人未连接，请先调用 start()")
            return False

        try:
            self.client.speak(
                self._session.session_id,
                text,
                interrupt=interrupt and self.config.interrupt_enabled,
            )
            return True
        except Exception as e:
            print(f"❌ 说话失败: {e}")
            return False

    def say_from_persona(self, user_input: str) -> bool:
        """
        用 LiveTalking 内置 LLM（已注入人格提示词）回复并说话。

        这是与别样觉醒人格系统集成的核心方法：
        1. 用户输入问题
        2. LiveTalking 内置 LLM 用人格提示词生成回复
        3. TTS 合成语音
        4. 口型同步驱动数字人说话
        """
        if not self._session:
            print("❌ 数字人未连接")
            return False

        try:
            self.client.speak_chat(self._session.session_id, user_input)
            return True
        except Exception as e:
            print(f"❌ 对话失败: {e}")
            return False

    def stop(self):
        """停止数字人"""
        if self._session:
            try:
                self.client.interrupt(self._session.session_id)
            except Exception:
                pass
            self._session = None
            print("🛑 数字人已停止")

    @property
    def is_active(self) -> bool:
        return self._session is not None and self._session.connected


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="别样觉醒 — 数字人引擎（融合自 LiveTalking）",
        epilog="""
示例:
  # 检查 LiveTalking 服务状态
  python3 digital_avatar.py --check

  # 让数字人说话
  python3 digital_avatar.py --say "你好，我是陈龙的数字分身"

  # 用LLM对话模式
  python3 digital_avatar.py --chat "你觉得AI能变现吗？"

  # 指定LiveTalking地址
  python3 digital_avatar.py --host http://192.168.1.100:8010 --say "测试"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default=LIVETALKING_DEFAULT_HOST, help="LiveTalking 服务地址")
    parser.add_argument("--check", action="store_true", help="检查服务状态")
    parser.add_argument("--say", help="让数字人说指定文本")
    parser.add_argument("--chat", help="用LLM对话模式")
    parser.add_argument("--model", default="wav2lip", choices=["wav2lip", "musetalk", "ultralight"], help="数字人模型")
    parser.add_argument("--avatar-id", default="wav2lip256_avatar1", help="数字人形象ID")
    parser.add_argument("--tts", default="edge", choices=["edge", "sovits", "cosyvoice", "fish"], help="TTS引擎")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural", help="TTS音色")

    args = parser.parse_args()

    config = AvatarConfig(
        model=AvatarModel(args.model),
        avatar_id=args.avatar_id,
        tts_engine=TTSEngine(args.tts),
        tts_voice=args.voice,
        host=args.host,
    )
    manager = DigitalAvatarManager(config)

    if args.check:
        ok = manager.client.health_check()
        print(f"{'✅' if ok else '❌'} LiveTalking {args.host}: {'运行中' if ok else '未运行'}")
        if ok:
            plugins = requests.get(f"{args.host}/", timeout=5).text[:200]
            print(f"   服务信息: {plugins[:100]}...")
        return

    if not manager.start():
        return

    try:
        if args.say:
            manager.say(args.say)
            print(f"💬 已发送: {args.say}")
            # 等待说完
            time.sleep(2)
        elif args.chat:
            manager.say_from_persona(args.chat)
            print(f"💬 对话模式: {args.chat}")
            time.sleep(5)
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
