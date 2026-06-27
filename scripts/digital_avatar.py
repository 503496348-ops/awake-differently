"""
别样觉醒 · AwakeEngine — 数字人引擎

实时流式数字人架构，将人格模拟器的文本回复驱动数字人实时说话。
支持 WebRTC 浏览器预览 / RTMP 推流 / 虚拟摄像头 / MP4录制。

数据流：
  persona_simulator(生成回复) → AwakeEngine(驱动数字人) → TTS → 口型同步 → 视频输出

依赖：
  - AwakeEngine 数字人服务（需单独部署启动）
  - requests 库

作者：AtomCollide-智械工坊
"""

import os
import json
import time
import requests
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum


# ── 配置 ──────────────────────────────────────────────────────────────────────

# AwakeEngine 默认地址
ENGINE_DEFAULT_HOST = "http://localhost:8010"

# 数字人模型
class AvatarModel(str, Enum):
    WAV2LIP = "wav2lip"           # 轻量，RTX 3060即可，60FPS
    MUSETALK = "musetalk"         # 高质量，需RTX 3080Ti+
    ULTRALIGHT = "ultralight"     # 极轻量，移动端友好
    ERNERF = "ernerf"             # NeRF 3D，最高质量

# TTS 引擎
class TTSEngine(str, Enum):
    EDGE = "edge"                 # 免费，300+音色
    SOVITS = "sovits"             # GPT-SoVITS 声音克隆
    COSYVOICE = "cosyvoice"      # 阿里CosyVoice
    FISH = "fish"                 # FishAudio
    DOUBAO = "doubao"             # 豆包TTS
    AZURE = "azure"               # Azure TTS
    TENCENT = "tencent"           # 腾讯云TTS

# 输出方式
class OutputMode(str, Enum):
    WEBRTC = "webrtc"             # 浏览器实时预览
    RTMP = "rtmp"                 # 推流到直播平台
    VIRTUAL_CAM = "virtualcam"    # 虚拟摄像头
    RECORD = "record"             # 录制MP4


# 后端类型
class BackendType(str, Enum):
    LOCAL = "local"       # 本地GPU引擎（需要部署AwakeEngine）
    DID = "did"           # D-ID 云端API（无需GPU，注册即用）


@dataclass
class AvatarConfig:
    """数字人配置"""
    backend: BackendType = BackendType.LOCAL
    model: AvatarModel = AvatarModel.WAV2LIP
    avatar_id: str = "wav2lip256_avatar1"  # 预训练形象ID
    tts_engine: TTSEngine = TTSEngine.EDGE
    tts_voice: str = "zh-CN-XiaoxiaoNeural"  # TTS音色
    output_mode: OutputMode = OutputMode.WEBRTC
    host: str = ENGINE_DEFAULT_HOST

    # LLM配置（内置chat模式）
    llm_enabled: bool = False  # True=用内置LLM, False=用外部persona_simulator
    llm_model: str = "qwen-plus"
    llm_api_key: str = ""

    # D-ID 云端配置
    did_api_key: str = ""            # D-ID API Key (从 studio.d-id.com 获取)
    did_source_url: str = ""         # D-ID 形象照片URL

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


# ── 数字人API客户端 ───────────────────────────────────────────────────

class AwakeEngineClient:
    """
    AwakeEngine 数字人 API 客户端。

    封装数字人引擎的 HTTP/WebRTC 接口，提供简洁的数字人驱动能力。

    用法：
        client = AwakeEngineClient("http://localhost:8010")
        session = client.connect()
        client.speak(session, "你好，我是陈龙")
        client.speak(session, "今天我们聊聊AI变现", interrupt=True)
    """

    def __init__(self, host: str = ENGINE_DEFAULT_HOST, timeout: int = 30):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._sessions: Dict[str, SessionState] = {}

    def health_check(self) -> bool:
        """检查数字人引擎服务是否可用"""
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
        用内置 LLM 对话模式驱动数字人。

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


# ── D-ID 云端API客户端 ─────────────────────────────────────────────────────

class DIDClient:
    """
    D-ID 云端数字人 API 客户端。

    无需GPU，注册即用。支持：
    - 照片+文本 → 说话视频
    - 照片+音频 → 说话视频
    - 实时对话代理（Agents API）

    注册获取API Key: https://studio.d-id.com/account-settings
    """
    BASE_URL = "https://api.d-id.com"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
        }

    def create_talk(
        self,
        source_url: str,
        text: str,
        voice: str = "zh-CN-XiaoxiaoNeural",
    ) -> str:
        """
        创建说话视频任务。

        Args:
            source_url: 人物照片URL（公网可访问）
            text: 要说的文本
            voice: TTS音色

        Returns:
            talk_id，用于轮询结果
        """
        import requests
        payload = {
            "source_url": source_url,
            "script": {
                "type": "text",
                "input": text,
                "provider": "microsoft",
                "voice_id": voice,
            },
        }
        resp = requests.post(
            f"{self.BASE_URL}/talks",
            headers=self._headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("id", "")

    def get_talk(self, talk_id: str) -> dict:
        """查询视频生成状态"""
        import requests
        resp = requests.get(
            f"{self.BASE_URL}/talks/{talk_id}",
            headers=self._headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_talk(self, talk_id: str, max_wait: int = 120) -> str:
        """等待视频生成完成，返回视频URL"""
        import time, requests
        start = time.time()
        while time.time() - start < max_wait:
            result = self.get_talk(talk_id)
            status = result.get("status", "")
            if status == "done":
                return result.get("result_url", "")
            elif status == "error":
                raise RuntimeError(f"D-ID生成失败: {result.get('error', {})}")
            time.sleep(3)
        raise TimeoutError(f"D-ID生成超时({max_wait}秒)")

    def speak(self, source_url: str, text: str, voice: str = "") -> str:
        """
        一步完成：文本→视频，返回视频URL。

        Args:
            source_url: 人物照片URL
            text: 要说的文本
            voice: TTS音色（默认中文女声）

        Returns:
            视频URL
        """
        if not voice:
            voice = "zh-CN-XiaoxiaoNeural"
        talk_id = self.create_talk(source_url, text, voice)
        return self.wait_for_talk(talk_id)

    def health_check(self) -> bool:
        """检查D-ID API是否可用"""
        import requests
        try:
            resp = requests.get(
                f"{self.BASE_URL}/talks",
                headers=self._headers,
                timeout=10,
            )
            return resp.status_code in (200, 401)  # 401=key无效但服务可达
        except Exception:
            return False


# ── 数字人管理器 ──────────────────────────────────────────────────────────────

class DigitalAvatarManager:
    """
    数字人管理器 — 桥接人格模拟器与数字人引擎。

    将别样觉醒的人格模拟输出，转化为数字人实时说话视频。

    用法：
        manager = DigitalAvatarManager(config)
        manager.start()
        manager.say("你好，我是陈龙的数字分身")
        manager.say_from_persona("你觉得这个方案怎么样？")
    """

    def __init__(self, config: Optional[AvatarConfig] = None):
        self.config = config or AvatarConfig()
        self._session: Optional[SessionState] = None
        self._did_client: Optional[DIDClient] = None

        if self.config.backend == BackendType.DID:
            if not self.config.did_api_key:
                raise ValueError("D-ID后端需要配置 did_api_key")
            self._did_client = DIDClient(api_key=self.config.did_api_key)
        else:
            self.client = AwakeEngineClient(host=self.config.host)

    def start(self) -> bool:
        """启动数字人会话"""
        if self.config.backend == BackendType.DID:
            if not self._did_client.health_check():
                print("❌ D-ID API 不可用，请检查 API Key")
                return False
            print(f"✅ D-ID 云端数字人已就绪")
            return True
        else:
            if not self.client.health_check():
                print(f"❌ 数字人引擎服务未运行: {self.config.host}")
                print(f"   请先启动数字人引擎服务")
                return False
            self._session = self.client.connect(avatar_id=self.config.avatar_id)
            print(f"✅ 数字人已连接: session={self._session.session_id}")
            return True

    def say(self, text: str, interrupt: bool = False) -> bool:
        """让数字人说指定文本，返回视频URL（D-ID模式）或True/False（本地模式）"""
        if self.config.backend == BackendType.DID:
            if not self.config.did_source_url:
                print("❌ D-ID后端需要配置 did_source_url（人物照片URL）")
                return False
            try:
                video_url = self._did_client.speak(
                    self.config.did_source_url,
                    text,
                    voice=self.config.tts_voice,
                )
                print(f"🎬 视频已生成: {video_url}")
                return True
            except Exception as e:
                print(f"❌ D-ID生成失败: {e}")
                return False
        else:
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
        用内置 LLM（已注入人格提示词）回复并说话。
        D-ID模式下暂不支持chat，会提示使用say。

        这是与别样觉醒人格系统集成的核心方法：
        1. 用户输入问题
        2. 内置 LLM 用人格提示词生成回复
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
        if self.config.backend == BackendType.DID:
            self._did_client = None
            print("🛑 D-ID数字人已断开")
        elif self._session:
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
        description="别样觉醒 · AwakeEngine — 数字人引擎",
        epilog="""
示例:
  # 检查数字人引擎服务状态
  python3 digital_avatar.py --check

  # 让数字人说话
  python3 digital_avatar.py --say "你好，我是陈龙的数字分身"

  # 用LLM对话模式
  python3 digital_avatar.py --chat "你觉得AI能变现吗？"

  # 使用D-ID云端（无需GPU）
  python3 digital_avatar.py --backend did --did-key YOUR_KEY --did-photo https://example.com/face.jpg --say "你好"

  # 指定本地引擎地址
  python3 digital_avatar.py --host http://192.168.1.100:8010 --say "测试"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", default="local", choices=["local", "did"], help="后端: local(本地GPU) / did(D-ID云端)")
    parser.add_argument("--did-key", default="", help="D-ID API Key")
    parser.add_argument("--did-photo", default="", help="D-ID 人物照片URL")
    parser.add_argument("--host", default=ENGINE_DEFAULT_HOST, help="本地引擎服务地址")
    parser.add_argument("--check", action="store_true", help="检查服务状态")
    parser.add_argument("--say", help="让数字人说指定文本")
    parser.add_argument("--chat", help="用LLM对话模式")
    parser.add_argument("--model", default="wav2lip", choices=["wav2lip", "musetalk", "ultralight"], help="数字人模型")
    parser.add_argument("--avatar-id", default="wav2lip256_avatar1", help="数字人形象ID")
    parser.add_argument("--tts", default="edge", choices=["edge", "sovits", "cosyvoice", "fish"], help="TTS引擎")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural", help="TTS音色")

    args = parser.parse_args()

    config = AvatarConfig(
        backend=BackendType(args.backend),
        model=AvatarModel(args.model),
        avatar_id=args.avatar_id,
        tts_engine=TTSEngine(args.tts),
        tts_voice=args.voice,
        host=args.host,
        did_api_key=args.did_key or os.environ.get("DID_API_KEY", ""),
        did_source_url=args.did_photo,
    )
    manager = DigitalAvatarManager(config)

    if args.check:
        if config.backend == BackendType.DID:
            ok = manager._did_client and manager._did_client.health_check()
            key_valid = "未配置" if not args.did_key else ("已配置" if ok else "无效")
            print(f"{'✅' if ok else '❌'} D-ID API: {'可达' if ok else '不可达'}")
            print(f"   API Key: {key_valid}")
        else:
            ok = manager.client.health_check()
            print(f"{'✅' if ok else '❌'} AwakeEngine {args.host}: {'运行中' if ok else '未运行'}")
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
