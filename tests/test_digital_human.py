"""
别样觉醒 · 数字人引擎 — 单元测试

覆盖：
  - AwakeEngineClient API调用
  - VoiceProfile 生成与参数推导
  - AvatarBuilder 形象管理
  - persona_to_voice_params 风格映射

作者：AtomCollide-智械工坊
"""

import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

# 添加 scripts 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from digital_avatar import (
    AwakeEngineClient,
    AvatarConfig,
    AvatarModel,
    TTSEngine,
    OutputMode,
    BackendType,
    DIDClient,
    DigitalAvatarManager,
)
from voice_profile import (
    VoiceProfile,
    persona_to_voice_params,
    recommend_voice,
    create_voice_profile_from_persona,
)
from avatar_builder import AvatarBuilder, AvatarBuildResult


# ── VoiceProfile 测试 ────────────────────────────────────────────────────────

class TestVoiceProfile(unittest.TestCase):

    def test_default_profile(self):
        p = VoiceProfile()
        self.assertEqual(p.engine, "edge")
        self.assertEqual(p.speed, 1.0)
        self.assertEqual(p.language, "zh")

    def test_to_tts_config(self):
        p = VoiceProfile(voice_id="zh-CN-YunxiNeural", speed=1.2, pitch=-1)
        config = p.to_tts_config()
        self.assertEqual(config["voice"], "zh-CN-YunxiNeural")
        self.assertEqual(config["speed"], 1.2)
        self.assertEqual(config["pitch"], -1)

    def test_to_dict(self):
        p = VoiceProfile(name="测试声音", engine="sovits")
        d = p.to_dict()
        self.assertEqual(d["name"], "测试声音")
        self.assertEqual(d["engine"], "sovits")


class TestPersonaToVoiceParams(unittest.TestCase):

    def test_short_sentence_style(self):
        traits = {"speaking_style": "短句为主", "tone": "直接", "energy": "high", "pace": "快"}
        params = persona_to_voice_params(traits)
        self.assertGreater(params["speed"], 1.0)
        self.assertEqual(params["emotion"], "confident")

    def test_calm_style(self):
        traits = {"speaking_style": "沉稳缓慢", "tone": "平和", "energy": "low", "pace": "慢"}
        params = persona_to_voice_params(traits)
        self.assertLess(params["speed"], 1.0)
        self.assertEqual(params["emotion"], "calm")

    def test_sarcastic_style(self):
        traits = {"speaking_style": "调侃", "tone": "嘴毒", "energy": "medium", "pace": "medium"}
        params = persona_to_voice_params(traits)
        self.assertEqual(params["pitch"], -1)


class TestRecommendVoice(unittest.TestCase):

    def test_zh_male_default(self):
        voice = recommend_voice("zh", "male", "default")
        self.assertIn("Neural", voice)

    def test_en_female(self):
        voice = recommend_voice("en", "female", "default")
        self.assertIn("Neural", voice)

    def test_unknown_lang_fallback(self):
        voice = recommend_voice("xx", "male")
        self.assertIn("Neural", voice)


class TestCreateVoiceProfileFromPersona(unittest.TestCase):

    def test_basic_persona(self):
        persona = {
            "name": "测试用户",
            "language": "zh",
            "gender": "male",
            "speaking_style": {"description": "短句为主", "pace": "快"},
            "personality": {"tone": "直接", "energy": "high"},
        }
        profile = create_voice_profile_from_persona(persona)
        self.assertEqual(profile.name, "测试用户的声音")
        self.assertEqual(profile.language, "zh")
        self.assertGreater(profile.speed, 1.0)


# ── AwakeEngineClient 测试 ───────────────────────────────────────────────────

class TestAwakeEngineClient(unittest.TestCase):

    def test_init(self):
        client = AwakeEngineClient("http://test:8010")
        self.assertEqual(client.host, "http://test:8010")

    def test_host_trailing_slash(self):
        client = AwakeEngineClient("http://test:8010/")
        self.assertEqual(client.host, "http://test:8010")

    @patch("digital_avatar.requests.get")
    def test_health_check_ok(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        client = AwakeEngineClient()
        self.assertTrue(client.health_check())

    @patch("digital_avatar.requests.get")
    def test_health_check_fail(self, mock_get):
        import requests
        mock_get.side_effect = requests.ConnectionError()
        client = AwakeEngineClient()
        self.assertFalse(client.health_check())

    @patch("digital_avatar.requests.post")
    def test_speak(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"code": 0, "msg": "ok"},
            raise_for_status=lambda: None,
        )
        client = AwakeEngineClient()
        client._sessions["test-session"] = MagicMock(message_count=0)
        result = client.speak("test-session", "你好")
        self.assertEqual(result["code"], 0)

    @patch("digital_avatar.requests.post")
    def test_interrupt(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"code": 0},
            raise_for_status=lambda: None,
        )
        client = AwakeEngineClient()
        result = client.interrupt("test-session")
        self.assertEqual(result["code"], 0)


# ── DigitalAvatarManager 测试 ────────────────────────────────────────────────

class TestDigitalAvatarManager(unittest.TestCase):

    def test_init_default_config(self):
        config = AvatarConfig()
        self.assertEqual(config.model, AvatarModel.WAV2LIP)

    def test_init_custom_config(self):
        config = AvatarConfig(model=AvatarModel.MUSETALK, tts_engine=TTSEngine.COSYVOICE)
        mgr = DigitalAvatarManager(config)
        self.assertEqual(mgr.config.model, AvatarModel.MUSETALK)

    @patch("digital_avatar.AwakeEngineClient.health_check", return_value=False)
    def test_start_fails_when_offline(self, mock_hc):
        mgr = DigitalAvatarManager()
        result = mgr.start()
        self.assertFalse(result)


# ── AvatarBuilder 测试 ───────────────────────────────────────────────────────

class TestAvatarBuilder(unittest.TestCase):

    def test_build_from_video_missing_file(self):
        builder = AvatarBuilder()
        result = builder.build_from_video("/nonexistent/video.mp4", "test")
        self.assertEqual(result.status, "failed")
        self.assertIn("不存在", result.message)

    def test_build_from_photo_missing_file(self):
        builder = AvatarBuilder()
        result = builder.build_from_photo("/nonexistent/photo.jpg", "test")
        self.assertEqual(result.status, "failed")

    @patch("avatar_builder.requests.get")
    def test_health_check(self, mock_get):
        import requests
        mock_get.side_effect = requests.ConnectionError()
        builder = AvatarBuilder()
        self.assertFalse(builder.health_check())


# ── AvatarConfig 测试 ────────────────────────────────────────────────────────

class TestAvatarConfig(unittest.TestCase):

    def test_default_values(self):
        config = AvatarConfig()
        self.assertEqual(config.model, AvatarModel.WAV2LIP)
        self.assertEqual(config.tts_engine, TTSEngine.EDGE)
        self.assertTrue(config.interrupt_enabled)

    def test_enum_values(self):
        self.assertEqual(AvatarModel.WAV2LIP.value, "wav2lip")
        self.assertEqual(TTSEngine.EDGE.value, "edge")
        self.assertEqual(OutputMode.WEBRTC.value, "webrtc")


if __name__ == "__main__":
    unittest.main()


class TestDIDClient(unittest.TestCase):

    def test_init(self):
        client = DIDClient("test-key")
        self.assertEqual(client.api_key, "test-key")

    def test_backend_type_enum(self):
        self.assertEqual(BackendType.LOCAL.value, "local")
        self.assertEqual(BackendType.DID.value, "did")
