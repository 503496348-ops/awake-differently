"""
别样觉醒 — 声音画像模块
配置和管理数字人的声音特征，对接 GPT-SoVITS / CosyVoice 声音克隆。

从聊天记录中提取的说话风格 → 转化为 TTS 参数 → 驱动数字人用"对的声音"说话。

支持：
  - Edge TTS 音色选择（免费，基于语言/性别/风格匹配）
  - GPT-SoVITS 声音克隆（需参考音频）
  - CosyVoice 声音克隆（阿里云）
  - 声音参数调节（语速、音调、情感）

依赖：
  - 无额外依赖（Edge TTS部分复用 ideasphere 的 tts_dubbing.py）

作者：AtomCollide-智械工坊
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List


# ── 声音画像 ──────────────────────────────────────────────────────────────────

@dataclass
class VoiceProfile:
    """
    声音画像 — 从人格蒸馏数据中提取的声音特征。

    与 persona.md 的"表达风格"对应，转化为可量化的 TTS 参数。
    """
    # 基础信息
    name: str = ""                    # 声音名称（如"陈龙的声音"）
    language: str = "zh"              # 主要语言
    gender: str = "male"              # 性别（用于音色推荐）

    # TTS 引擎选择
    engine: str = "edge"              # edge / sovits / cosyvoice / fish
    voice_id: str = ""                # 具体音色ID（引擎相关）

    # 声音参数（从人格蒸馏推导）
    speed: float = 1.0                # 语速 (0.5-2.0)
    pitch: int = 0                    # 音调 (-12 to 12)
    volume: float = 1.0               # 音量 (0.0-1.0)
    emotion: str = "neutral"          # 情感风格

    # 声音克隆配置（GPT-SoVITS / CosyVoice）
    reference_audio: str = ""         # 参考音频文件路径
    reference_text: str = ""          # 参考音频对应的文本

    # 风格映射（从人格蒸馏自动推导）
    style_tags: List[str] = field(default_factory=list)  # ["短句", "直接", "嘴毒"]

    def to_tts_config(self) -> Dict:
        """转换为数字人引擎 TTS 透传配置"""
        config = {
            "voice": self.voice_id,
            "speed": self.speed,
            "pitch": self.pitch,
            "volume": self.volume,
        }
        if self.emotion != "neutral":
            config["emotion"] = self.emotion
        return config

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
            "engine": self.engine,
            "voice_id": self.voice_id,
            "speed": self.speed,
            "pitch": self.pitch,
            "volume": self.volume,
            "emotion": self.emotion,
            "reference_audio": self.reference_audio,
            "style_tags": self.style_tags,
        }


# ── 音色推荐表 ────────────────────────────────────────────────────────────────

# 音色推荐库
# 按语言+性别+风格分类，从人格蒸馏自动匹配
VOICE_RECOMMENDATIONS = {
    "zh": {
        "male": {
            "default": "zh-CN-YunxiNeural",         # 云希（自然男声）
            "casual": "zh-CN-YunjianNeural",         # 云健（轻松男声）
            "serious": "zh-CN-YunyangNeural",        # 云扬（新闻播报）
            "young": "zh-CN-YunxiNeural",            # 云希（年轻男声）
        },
        "female": {
            "default": "zh-CN-XiaoxiaoNeural",      # 晓晓（自然女声）
            "casual": "zh-CN-XiaoyiNeural",          # 晓依（活泼女声）
            "serious": "zh-CN-XiaochenNeural",       # 晓辰（沉稳女声）
            "young": "zh-CN-XiaoxiaoNeural",         # 晓晓（年轻女声）
        },
    },
    "en": {
        "male": {
            "default": "en-US-GuyNeural",
            "casual": "en-US-ChristopherNeural",
            "serious": "en-US-DavisNeural",
        },
        "female": {
            "default": "en-US-JennyNeural",
            "casual": "en-US-AriaNeural",
            "serious": "en-US-SaraNeural",
        },
    },
    "ja": {
        "male": {"default": "ja-JP-KeitaNeural"},
        "female": {"default": "ja-JP-NanamiNeural"},
    },
}


# ── 人格风格 → 声音参数映射 ──────────────────────────────────────────────────

def persona_to_voice_params(persona_traits: Dict) -> Dict:
    """
    从人格蒸馏特征自动推导声音参数。

    输入（来自 persona.md 的风格描述）：
        {
            "speaking_style": "短句为主，一两句话说完",
            "tone": "嘴毒心热",
            "energy": "高",
            "pace": "快",
        }

    输出（TTS参数）：
        {"speed": 1.2, "pitch": -2, "emotion": "confident"}
    """
    params = {
        "speed": 1.0,
        "pitch": 0,
        "volume": 1.0,
        "emotion": "neutral",
    }

    style = persona_traits.get("speaking_style", "")
    tone = persona_traits.get("tone", "")
    energy = persona_traits.get("energy", "medium")
    pace = persona_traits.get("pace", "medium")

    # 语速映射
    if "短句" in style or "快" in pace or "直接" in style:
        params["speed"] = 1.15
    elif "慢" in pace or "沉稳" in style:
        params["speed"] = 0.9

    # 音调映射
    if "嘴毒" in tone or "调侃" in style:
        params["pitch"] = -1
    elif "热情" in tone:
        params["pitch"] = 1

    # 能量映射
    if energy == "high" or "感叹号" in style:
        params["volume"] = 1.0
        params["emotion"] = "confident"
    elif energy == "low":
        params["volume"] = 0.85
        params["emotion"] = "calm"

    return params


def recommend_voice(
    language: str = "zh",
    gender: str = "male",
    style: str = "default",
) -> str:
    """
    根据语言、性别、风格推荐最佳音色。

    Returns:
        Edge TTS 音色名称
    """
    lang_voices = VOICE_RECOMMENDATIONS.get(language, VOICE_RECOMMENDATIONS.get("zh"))
    gender_voices = lang_voices.get(gender, lang_voices.get("male"))
    return gender_voices.get(style, gender_voices.get("default", "zh-CN-YunxiNeural"))


# ── 声音画像工厂 ──────────────────────────────────────────────────────────────

def create_voice_profile_from_persona(
    persona_data: Dict,
    engine: str = "edge",
    reference_audio: str = "",
) -> VoiceProfile:
    """
    从人格蒸馏数据自动创建声音画像。

    Args:
        persona_data: persona.md 解析后的数据
        engine: TTS引擎 (edge/sovits/cosyvoice)
        reference_audio: 声音克隆参考音频路径

    Returns:
        VoiceProfile 实例
    """
    # 提取基本信息
    name = persona_data.get("name", "")
    language = persona_data.get("language", "zh")
    gender = persona_data.get("gender", "male")

    # 提取风格特征
    style_traits = persona_data.get("speaking_style", {})
    personality = persona_data.get("personality", {})

    # 推导声音参数
    voice_params = persona_to_voice_params({
        "speaking_style": style_traits.get("description", ""),
        "tone": personality.get("tone", ""),
        "energy": personality.get("energy", "medium"),
        "pace": style_traits.get("pace", "medium"),
    })

    # 选择音色
    style_key = "casual" if "调侃" in str(style_traits) else "default"
    voice_id = recommend_voice(language, gender, style_key)

    # 如果有参考音频，切换到声音克隆引擎
    if reference_audio and os.path.exists(reference_audio):
        engine = "sovits"  # GPT-SoVITS

    profile = VoiceProfile(
        name=f"{name}的声音",
        language=language,
        gender=gender,
        engine=engine,
        voice_id=voice_id,
        speed=voice_params["speed"],
        pitch=voice_params["pitch"],
        volume=voice_params["volume"],
        emotion=voice_params["emotion"],
        reference_audio=reference_audio,
        style_tags=list(style_traits.get("tags", [])),
    )

    return profile


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="别样觉醒 — 声音画像工具")
    parser.add_argument("--list-voices", action="store_true", help="列出推荐音色")
    parser.add_argument("--language", "-l", default="zh", help="语言")
    parser.add_argument("--gender", "-g", default="male", choices=["male", "female"], help="性别")
    parser.add_argument("--recommend", action="store_true", help="推荐音色")
    parser.add_argument("--persona", help="人格画像JSON文件路径")
    parser.add_argument("--output", "-o", help="输出声音画像JSON路径")

    args = parser.parse_args()

    if args.list_voices:
        print("\n🎙️ 可用音色推荐:")
        lang = VOICE_RECOMMENDATIONS.get(args.language, {})
        for gender, voices in lang.items():
            print(f"\n  [{gender}]")
            for style, voice in voices.items():
                print(f"    {style:<12} → {voice}")
        return

    if args.recommend:
        voice = recommend_voice(args.language, args.gender)
        print(f"🎙️ 推荐音色: {voice}")
        return

    if args.persona:
        with open(args.persona, "r", encoding="utf-8") as f:
            persona_data = json.load(f)
        profile = create_voice_profile_from_persona(persona_data)
        print(f"🎙️ 声音画像: {profile.name}")
        print(f"   引擎: {profile.engine}")
        print(f"   音色: {profile.voice_id}")
        print(f"   语速: {profile.speed}")
        print(f"   音调: {profile.pitch}")
        print(f"   情感: {profile.emotion}")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"   保存到: {args.output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
