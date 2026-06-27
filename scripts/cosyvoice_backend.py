"""
Awake Differently — CosyVoice Backend
======================================

Integration with FunAudioLLM/CosyVoice for high-quality Chinese TTS
with voice cloning support.

Inference modes:
1. pretrained   — Use pre-trained speaker embeddings
2. clone_3s     — Zero-shot voice cloning with 3s prompt audio
3. cross_lingual — Cross-language voice cloning
4. instruct     — Natural language voice control

Usage:
    backend = CosyVoiceBackend(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")
    
    # Pre-trained voice
    audio = backend.synthesize("你好世界", speaker="中文女")
    
    # 3s voice clone
    audio = backend.synthesize_clone(
        "你好世界",
        prompt_audio="speaker.wav",
        prompt_text="这是参考音频的文本"
    )
    
    # Natural language control
    audio = backend.synthesize_instruct(
        "你好世界",
        instruct="用温柔的语气说"
    )

Author: AtomCollide-智械工坊
License: MIT
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger("awake_differently.cosyvoice")


@dataclass
class CosyVoiceConfig:
    """Configuration for CosyVoice backend."""
    model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B"
    sample_rate: int = 24000
    speed: float = 1.0
    seed: int = 42
    stream: bool = False
    device: str = "cuda"  # cuda or cpu


@dataclass
class AudioResult:
    """Result from TTS synthesis."""
    sample_rate: int
    audio_data: Any  # numpy array
    duration_ms: float
    mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_wav_bytes(self) -> bytes:
        """Convert to WAV bytes."""
        import io
        import struct

        data = self.audio_data
        if hasattr(data, 'numpy'):
            data = data.numpy()
        data = data.flatten()

        # Normalize to int16
        import numpy as np
        data = (data * 32767).astype(np.int16)

        buf = io.BytesIO()
        # WAV header
        buf.write(b'RIFF')
        buf.write(struct.pack('<I', 36 + len(data) * 2))
        buf.write(b'WAVE')
        buf.write(b'fmt ')
        buf.write(struct.pack('<I', 16))
        buf.write(struct.pack('<HHIIHH', 1, 1, self.sample_rate,
                              self.sample_rate * 2, 2, 16))
        buf.write(b'data')
        buf.write(struct.pack('<I', len(data) * 2))
        buf.write(data.tobytes())
        return buf.getvalue()

    def save_wav(self, path: str) -> str:
        """Save as WAV file."""
        with open(path, 'wb') as f:
            f.write(self.to_wav_bytes())
        return path

    def to_opus_bytes(self) -> bytes:
        """Convert to Opus bytes (for Feishu voice bubbles)."""
        import numpy as np

        data = self.audio_data
        if hasattr(data, 'numpy'):
            data = data.numpy()
        data = data.flatten()

        # Save as temp WAV first
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            self.save_wav(tmp_path)

        try:
            # Convert WAV to Opus using ffmpeg
            opus_path = tmp_path.replace('.wav', '.opus')
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', tmp_path,
                 '-c:a', 'libopus', '-b:a', '32k',
                 '-ar', '16000', '-ac', '1',
                 opus_path],
                capture_output=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg opus conversion failed: {result.stderr.decode()}")

            with open(opus_path, 'rb') as f:
                return f.read()
        finally:
            for p in [tmp_path, tmp_path.replace('.wav', '.opus')]:
                if os.path.exists(p):
                    os.unlink(p)

    def save_opus(self, path: str) -> str:
        """Save as Opus file (for Feishu voice bubbles)."""
        with open(path, 'wb') as f:
            f.write(self.to_opus_bytes())
        return path


class CosyVoiceBackend:
    """CosyVoice TTS backend with voice cloning support.

    Requires:
        - CosyVoice installed (pip install -e . from CosyVoice repo)
        - Pretrained models downloaded to model_dir
        - ffmpeg installed (for opus conversion)
    """

    def __init__(self, config: Optional[CosyVoiceConfig] = None):
        self.config = config or CosyVoiceConfig()
        self._model = None
        self._available_speakers: List[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the CosyVoice model."""
        if self._model is not None:
            return

        logger.info(f"Loading CosyVoice model from {self.config.model_dir}")

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
            from cosyvoice.utils.file_utils import load_wav

            self._model = CosyVoice(self.config.model_dir)
            self._available_speakers = self._model.list_available_speakers()
            self._load_wav = load_wav

            logger.info(f"Loaded. Available speakers: {self._available_speakers}")
        except ImportError:
            raise ImportError(
                "CosyVoice not installed. Install with:\n"
                "  git clone https://github.com/FunAudioLLM/CosyVoice.git\n"
                "  cd CosyVoice && pip install -e ."
            )

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
        self._available_speakers = []
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def list_speakers(self) -> List[str]:
        """List available pre-trained speakers."""
        if not self.is_loaded:
            self.load()
        return self._available_speakers

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        speed: Optional[float] = None,
        seed: Optional[int] = None,
        stream: Optional[bool] = None,
    ) -> AudioResult:
        """Synthesize speech using pre-trained voice.

        Args:
            text: Text to synthesize
            speaker: Speaker name (from list_speakers())
            speed: Speech speed (0.5-2.0)
            seed: Random seed
            stream: Enable streaming

        Returns:
            AudioResult with synthesized audio
        """
        if not self.is_loaded:
            self.load()

        speed = speed or self.config.speed
        seed = seed or self.config.seed
        stream = stream if stream is not None else self.config.stream

        speaker = speaker or (self._available_speakers[0] if self._available_speakers else None)
        if not speaker:
            raise ValueError("No speaker available. Provide a speaker name.")

        import torch
        from cosyvoice.utils.file_utils import set_all_random_seed
        set_all_random_seed(seed)

        logger.info(f"Synthesizing with speaker={speaker}, speed={speed}")
        results = self._model.inference_sft(text, speaker, speed=speed, stream=stream)

        return self._collect_result(results, mode="pretrained")

    def synthesize_clone(
        self,
        text: str,
        prompt_audio: str,
        prompt_text: str,
        speed: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> AudioResult:
        """Synthesize speech using 3s voice cloning.

        Args:
            text: Text to synthesize
            prompt_audio: Path to reference audio (3-30s, 16kHz+)
            prompt_text: Transcript of reference audio
            speed: Speech speed
            seed: Random seed

        Returns:
            AudioResult with synthesized audio in cloned voice
        """
        if not self.is_loaded:
            self.load()

        speed = speed or self.config.speed
        seed = seed or self.config.seed

        from cosyvoice.utils.file_utils import set_all_random_seed, load_wav
        set_all_random_seed(seed)

        # Load prompt audio at 16kHz
        prompt_speech = load_wav(prompt_audio, 16000)

        logger.info(f"Cloning voice from {prompt_audio}")
        results = self._model.inference_zero_shot(
            text, prompt_text, prompt_speech, speed=speed
        )

        return self._collect_result(results, mode="clone_3s")

    def synthesize_cross_lingual(
        self,
        text: str,
        prompt_audio: str,
        speed: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> AudioResult:
        """Synthesize speech using cross-lingual voice cloning.

        Args:
            text: Text to synthesize (can be different language than prompt)
            prompt_audio: Path to reference audio
            speed: Speech speed
            seed: Random seed

        Returns:
            AudioResult with synthesized audio
        """
        if not self.is_loaded:
            self.load()

        speed = speed or self.config.speed
        seed = seed or self.config.seed

        from cosyvoice.utils.file_utils import set_all_random_seed, load_wav
        set_all_random_seed(seed)

        prompt_speech = load_wav(prompt_audio, 16000)

        logger.info(f"Cross-lingual clone from {prompt_audio}")
        results = self._model.inference_cross_lingual(
            text, prompt_speech, speed=speed
        )

        return self._collect_result(results, mode="cross_lingual")

    def synthesize_instruct(
        self,
        text: str,
        instruct: str,
        speaker: Optional[str] = None,
        speed: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> AudioResult:
        """Synthesize speech with natural language instructions.

        Args:
            text: Text to synthesize
            instruct: Natural language instruction (e.g., "用温柔的语气说")
            speaker: Speaker name
            speed: Speech speed
            seed: Random seed

        Returns:
            AudioResult with synthesized audio
        """
        if not self.is_loaded:
            self.load()

        speed = speed or self.config.speed
        seed = seed or self.config.seed
        speaker = speaker or (self._available_speakers[0] if self._available_speakers else None)

        from cosyvoice.utils.file_utils import set_all_random_seed
        set_all_random_seed(seed)

        logger.info(f"Instruct mode: '{instruct}'")
        results = self._model.inference_instruct(
            text, instruct, speaker, speed=speed
        )

        return self._collect_result(results, mode="instruct")

    def _collect_result(self, results, mode: str) -> AudioResult:
        """Collect streaming/non-streaming results into AudioResult."""
        import numpy as np

        audio_chunks = []
        for chunk in results:
            audio_chunks.append(chunk['tts_speech'].numpy().flatten())

        audio_data = np.concatenate(audio_chunks)
        duration_ms = len(audio_data) / self.config.sample_rate * 1000

        return AudioResult(
            sample_rate=self.config.sample_rate,
            audio_data=audio_data,
            duration_ms=duration_ms,
            mode=mode,
            metadata={
                "model_dir": self.config.model_dir,
                "num_chunks": len(audio_chunks),
            }
        )


# ── Convenience Functions ──────────────────────────────────────────────────

_global_backend: Optional[CosyVoiceBackend] = None


def get_backend(model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B") -> CosyVoiceBackend:
    """Get or create global CosyVoice backend."""
    global _global_backend
    if _global_backend is None:
        _global_backend = CosyVoiceBackend(CosyVoiceConfig(model_dir=model_dir))
    return _global_backend


def tts(
    text: str,
    speaker: Optional[str] = None,
    model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B",
) -> AudioResult:
    """Quick TTS with pre-trained voice.

    Usage:
        result = tts("你好世界", speaker="中文女")
        result.save_wav("output.wav")
        result.save_opus("output.opus")  # For Feishu voice bubbles
    """
    backend = get_backend(model_dir)
    return backend.synthesize(text, speaker=speaker)


def tts_clone(
    text: str,
    prompt_audio: str,
    prompt_text: str,
    model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B",
) -> AudioResult:
    """Quick TTS with voice cloning.

    Usage:
        result = tts_clone("你好世界", "speaker.wav", "参考文本")
        result.save_opus("output.opus")  # For Feishu voice bubbles
    """
    backend = get_backend(model_dir)
    return backend.synthesize_clone(text, prompt_audio, prompt_text)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CosyVoice TTS Backend")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--speaker", type=str, default=None, help="Speaker name")
    parser.add_argument("--model-dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file path")
    parser.add_argument("--opus", action="store_true", help="Output as Opus (for Feishu)")
    parser.add_argument("--clone-audio", type=str, help="Reference audio for cloning")
    parser.add_argument("--clone-text", type=str, help="Transcript of reference audio")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--list-speakers", action="store_true", help="List available speakers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    backend = CosyVoiceBackend(CosyVoiceConfig(model_dir=args.model_dir))
    backend.load()

    if args.list_speakers:
        for spk in backend.list_speakers():
            print(spk)
        exit(0)

    if args.clone_audio:
        if not args.clone_text:
            parser.error("--clone-text required with --clone-audio")
        result = backend.synthesize_clone(
            args.text, args.clone_audio, args.clone_text, speed=args.speed
        )
    else:
        result = backend.synthesize(args.text, speaker=args.speaker, speed=args.speed)

    if args.opus:
        result.save_opus(args.output.replace('.wav', '.opus'))
        print(f"Saved Opus: {args.output.replace('.wav', '.opus')} ({result.duration_ms:.0f}ms)")
    else:
        result.save_wav(args.output)
        print(f"Saved WAV: {args.output} ({result.duration_ms:.0f}ms)")


# ── 音色库 ────────────────────────────────────────────────────────────────

# CosyVoice3 支持的语言和方言
SUPPORTED_LANGUAGES = {
    "zh": "中文（普通话）",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "de": "德语",
    "es": "西班牙语",
    "fr": "法语",
    "it": "意大利语",
    "ru": "俄语",
}

# 中文方言/口音
CHINESE_DIALECTS = {
    "cantonese": "粤语",
    "minnan": "闽南语",
    "sichuan": "四川话",
    "dongbei": "东北话",
    "shanghai": "上海话",
    "tianjin": "天津话",
}

# ── 预训练音色库（60+音色，来自阿里云CosyVoice） ──────────────────────────
# 用户只需选择"场景+风格"，系统自动推荐最佳音色
# 详见: https://help.aliyun.com/zh/model-studio/cosyvoice-voice-list

PRETRAINED_SPEAKERS = {
    # ═══ 社交陪伴 ═══
    "longanyang":    {"name": "龙安洋", "lang": "zh", "gender": "male",   "scene": "社交陪伴", "style": "阳光大男孩", "age": "20-30", "instruct": True},
    "longanhuan":    {"name": "龙安欢", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "欢脱元气女", "age": "20-30", "instruct": True},
    "longanhuan_v3": {"name": "龙安欢V3", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "欢脱元气女+9方言", "age": "20-30", "instruct": True},
    "longantai_v3":  {"name": "龙安台", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "嗲甜台湾女", "age": "20-30"},
    "longhua_v3":    {"name": "龙华", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "元气甜美女", "age": "20-30"},
    "longcheng_v3":  {"name": "龙橙", "lang": "zh", "gender": "male",   "scene": "社交陪伴", "style": "智慧青年男", "age": "25-30"},
    "longze_v3":     {"name": "龙泽", "lang": "zh", "gender": "male",   "scene": "社交陪伴", "style": "温暖元气男", "age": "20-30"},
    "longzhe_v3":    {"name": "龙哲", "lang": "zh", "gender": "male",   "scene": "社交陪伴", "style": "呆板大暖男", "age": "20-30"},
    "longyan_v3":    {"name": "龙颜", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "温暖春风女", "age": "20-30"},
    "longxing_v3":   {"name": "龙星", "lang": "zh", "gender": "male",   "scene": "社交陪伴", "style": "磁性理智男", "age": "25-30"},
    "longtian_v3":   {"name": "龙天", "lang": "zh", "gender": "male",   "scene": "社交陪伴", "style": "浪漫风情男", "age": "25-30"},
    "longwan_v3":    {"name": "龙婉", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "温婉邻家女", "age": "20-30"},
    "longqiang_v3":  {"name": "龙嫱", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "浪漫风情女", "age": "25-30"},
    "longfeifei_v3": {"name": "龙菲菲", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "甜美娇气女", "age": "20-25"},
    "longhao_v3":    {"name": "龙浩", "lang": "zh", "gender": "male",   "scene": "社交陪伴", "style": "多情忧郁男", "age": "25-30"},
    "longanrou_v3":  {"name": "龙安柔", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "温柔闺蜜女", "age": "20-30"},
    "longhan_v3":    {"name": "龙寒", "lang": "zh", "gender": "male",   "scene": "社交陪伴", "style": "温暖痴情男", "age": "25-30"},
    "longanzhi_v3":  {"name": "龙安智", "lang": "zh", "gender": "male",   "scene": "社交陪伴", "style": "睿智轻熟男", "age": "25-35"},
    "longanling_v3": {"name": "龙安灵", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "思维灵动女", "age": "20-30"},
    "longanya_v3":   {"name": "龙安雅", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "高雅气质女", "age": "25-35"},
    "longanqin_v3":  {"name": "龙安亲", "lang": "zh", "gender": "female", "scene": "社交陪伴", "style": "亲和活泼女", "age": "20-30"},

    # ═══ 语音助手 ═══
    "longxiaochun_v3": {"name": "龙小淳", "lang": "zh", "gender": "female", "scene": "语音助手", "style": "知性积极女", "age": "25-30"},
    "longxiaoxia_v3":  {"name": "龙小夏", "lang": "zh", "gender": "female", "scene": "语音助手", "style": "沉稳权威女", "age": "25-30"},
    "longyumi_v3":     {"name": "YUMI", "lang": "zh", "gender": "female", "scene": "语音助手", "style": "正经青年女", "age": "25-30"},
    "longanyun_v3":    {"name": "龙安昀", "lang": "zh", "gender": "male",   "scene": "语音助手", "style": "居家暖男", "age": "25-30"},
    "longanwen_v3":    {"name": "龙安温", "lang": "zh", "gender": "female", "scene": "语音助手", "style": "优雅知性女", "age": "25-35"},
    "longanli_v3":     {"name": "龙安莉", "lang": "zh", "gender": "female", "scene": "语音助手", "style": "利落从容女", "age": "25-35"},
    "longanlang_v3":   {"name": "龙安朗", "lang": "zh", "gender": "male",   "scene": "语音助手", "style": "温暖春风男", "age": "25-30"},
    "longyingmu_v3":   {"name": "龙应沐", "lang": "zh", "gender": "female", "scene": "语音助手", "style": "优雅知性女", "age": "25-30"},

    # ═══ 客服 ═══
    "longyingxun_v3":   {"name": "龙应询", "lang": "zh", "gender": "male",   "scene": "客服", "style": "年轻青涩男", "age": "20-25"},
    "longyingjing_v3":  {"name": "龙应静", "lang": "zh", "gender": "female", "scene": "客服", "style": "低调冷静女", "age": "25-35"},
    "longyingling_v3":  {"name": "龙应聆", "lang": "zh", "gender": "female", "scene": "客服", "style": "温和共情女", "age": "25-30"},
    "longyingtao_v3":   {"name": "龙应桃", "lang": "zh", "gender": "female", "scene": "客服", "style": "温柔淡定女", "age": "25-30"},

    # ═══ 有声书 ═══
    "longmiao_v3":     {"name": "龙妙", "lang": "zh", "gender": "female", "scene": "有声书", "style": "抑扬顿挫女", "age": "25-30"},
    "longsanshu_v3":   {"name": "龙三叔", "lang": "zh", "gender": "male",   "scene": "有声书", "style": "沉稳质感男", "age": "35-45"},
    "longyuan_v3":     {"name": "龙媛", "lang": "zh", "gender": "female", "scene": "有声书", "style": "温暖治愈女", "age": "25-30"},
    "longyue_v3":      {"name": "龙悦", "lang": "zh", "gender": "female", "scene": "有声书", "style": "温暖磁性女", "age": "25-30"},
    "longxiu_v3":      {"name": "龙修", "lang": "zh", "gender": "male",   "scene": "有声书", "style": "博才说书男", "age": "30-40"},
    "longnan_v3":      {"name": "龙楠", "lang": "zh", "gender": "male",   "scene": "有声书", "style": "睿智青年男", "age": "25-30"},
    "longwanjun_v3":   {"name": "龙婉君", "lang": "zh", "gender": "female", "scene": "有声书", "style": "细腻柔声女", "age": "25-30"},
    "longyichen_v3":   {"name": "龙逸尘", "lang": "zh", "gender": "male",   "scene": "有声书", "style": "洒脱活力男", "age": "25-30"},
    "longlaobo_v3":    {"name": "龙老伯", "lang": "zh", "gender": "male",   "scene": "有声书", "style": "沧桑岁月爷", "age": "60+"},
    "longlaoyi_v3":    {"name": "龙老姨", "lang": "zh", "gender": "female", "scene": "有声书", "style": "烟火从容阿姨", "age": "50+"},

    # ═══ 新闻播报 ═══
    "longshuo_v3":    {"name": "龙硕", "lang": "zh", "gender": "male",   "scene": "新闻播报", "style": "博才干练男", "age": "25-30"},
    "longshu_v3":     {"name": "龙书", "lang": "zh", "gender": "male",   "scene": "新闻播报", "style": "沉稳青年男", "age": "25-30"},
    "loongbella_v3":  {"name": "Bella3.0", "lang": "zh", "gender": "female", "scene": "新闻播报", "style": "精准干练女", "age": "25-30"},

    # ═══ 短视频配音 ═══
    "longjiqi_v3":    {"name": "龙机器", "lang": "zh", "gender": "male",   "scene": "短视频", "style": "呆萌机器人", "age": "N/A"},
    "longhouge_v3":   {"name": "龙猴哥", "lang": "zh", "gender": "male",   "scene": "短视频", "style": "经典猴哥", "age": "N/A"},
    "longdaiyu_v3":   {"name": "龙黛玉", "lang": "zh", "gender": "female", "scene": "短视频", "style": "娇率才女", "age": "20-25"},

    # ═══ 直播带货 ═══
    "longanran_v3":   {"name": "龙安燃", "lang": "zh", "gender": "female", "scene": "直播", "style": "活泼质感女", "age": "30-40"},
    "longanxuan_v3":  {"name": "龙安宣", "lang": "zh", "gender": "female", "scene": "直播", "style": "经典直播女", "age": "30-40"},

    # ═══ 电话销售 ═══
    "longyingxiao_v3": {"name": "龙应笑", "lang": "zh", "gender": "female", "scene": "销售", "style": "清甜推销女", "age": "20-25"},

    # ═══ 诗词朗诵 ═══
    "longfei_v3":     {"name": "龙飞", "lang": "zh", "gender": "male",   "scene": "朗诵", "style": "热血磁性男", "age": "30-35"},

    # ═══ 童声 ═══
    "longhuhu_v3":    {"name": "龙呼呼", "lang": "zh", "gender": "female", "scene": "童声", "style": "天真烂漫女童", "age": "6-10", "instruct": True},
    "longpaopao_v3":  {"name": "龙泡泡", "lang": "zh", "gender": "female", "scene": "童声", "style": "飞天泡泡音", "age": "6-10"},
    "longjielidou_v3": {"name": "龙杰力豆", "lang": "zh", "gender": "male", "scene": "童声", "style": "阳光顽皮男", "age": "10"},
    "longxian_v3":    {"name": "龙仙", "lang": "zh", "gender": "female", "scene": "童声", "style": "豪放可爱女", "age": "10-15"},
    "longling_v3":    {"name": "龙铃", "lang": "zh", "gender": "female", "scene": "童声", "style": "稚气呆板女", "age": "10-15"},
    "longshanshan_v3": {"name": "龙闪闪", "lang": "zh", "gender": "female", "scene": "童声", "style": "戏剧化童声", "age": "6-15"},
    "longniuniu_vv3": {"name": "龙牛牛", "lang": "zh", "gender": "male",   "scene": "童声", "style": "阳光男童声", "age": "6-15"},

    # ═══ 方言 ═══
    "longjiaxin_v3":  {"name": "龙嘉欣", "lang": "zh-cantonese", "gender": "female", "scene": "方言", "style": "优雅粤语女", "age": "25-30"},
    "longjiayi_v3":   {"name": "龙嘉怡", "lang": "zh-cantonese", "gender": "female", "scene": "方言", "style": "知性粤语女", "age": "25-30"},
    "longanyue_v3":   {"name": "龙安粤", "lang": "zh-cantonese", "gender": "male",   "scene": "方言", "style": "欢脱粤语男", "age": "25-35"},
    "longlaotie_v3":  {"name": "龙老铁", "lang": "zh-dongbei", "gender": "male",   "scene": "方言", "style": "东北直率男", "age": "25-30"},
    "longshange_v3":  {"name": "龙陕哥", "lang": "zh-shaanxi", "gender": "male",   "scene": "方言", "style": "原味陕北男", "age": "25-35"},
    "longanmin_v3":   {"name": "龙安闽", "lang": "zh-minnan", "gender": "female", "scene": "方言", "style": "清纯闽南女", "age": "18-25"},

    # ═══ 出海营销（英文/日文/韩文） ═══
    "loongabby_v3":   {"name": "Abby", "lang": "en", "gender": "female", "scene": "出海", "style": "美式英文女", "age": "30-35"},
    "loongandy_v3":   {"name": "Andy", "lang": "en", "gender": "male",   "scene": "出海", "style": "美式英文男", "age": "30-35"},
    "loongemily_v3":  {"name": "Emily", "lang": "en", "gender": "female", "scene": "出海", "style": "英式英文女", "age": "25-30"},
    "loongeric_v3":   {"name": "Eric", "lang": "en", "gender": "male",   "scene": "出海", "style": "英式英文男", "age": "30-35"},
    "loongkyong_v3":  {"name": "Kyong", "lang": "ko", "gender": "female", "scene": "出海", "style": "韩语女", "age": "25-30"},
    "loongtomoka_v3": {"name": "Tomoka", "lang": "ja", "gender": "female", "scene": "出海", "style": "日语女", "age": "25-30"},
}


def list_voices(language: str = None, gender: str = None, scene: str = None) -> dict:
    """列出可用音色，支持按语言/性别/场景筛选。

    Args:
        language: 语言代码（zh/en/ja/ko/zh-cantonese/zh-dongbei等）
        gender: 性别（male/female）
        scene: 场景（社交陪伴/语音助手/客服/有声书/新闻播报/短视频/直播/销售/朗诵/童声/方言/出海）

    Returns:
        音色字典 {id: {name, lang, gender, style, scene, ...}}
    """
    voices = PRETRAINED_SPEAKERS.copy()

    if language:
        voices = {k: v for k, v in voices.items() if v["lang"].startswith(language)}
    if gender:
        voices = {k: v for k, v in voices.items() if v["gender"] == gender}
    if scene:
        voices = {k: v for k, v in voices.items() if v.get("scene") == scene}

    return voices


def list_scenes() -> list:
    """列出所有可用场景。"""
    scenes = set()
    for info in PRETRAINED_SPEAKERS.values():
        if "scene" in info:
            scenes.add(info["scene"])
    return sorted(scenes)


def recommend_voice_by_text(text: str, gender: str = "female") -> str:
    """根据文本内容自动推荐最佳音色。

    分析文本特征（长度、语气、场景）推荐最适合的音色。
    """
    # 短句+口语化 → 社交陪伴
    if len(text) < 50 and any(c in text for c in ["！", "？", "哈", "嘿", "哦"]):
        if gender == "female":
            return "longanhuan"  # 欢脱元气女
        return "longanyang"  # 阳光大男孩

    # 长文本 → 有声书
    if len(text) > 200:
        if gender == "female":
            return "longmiao_v3"  # 抑扬顿挫女
        return "longsanshu_v3"  # 沉稳质感男

    # 正式内容 → 语音助手
    if any(word in text for word in ["您好", "请问", "感谢", "服务"]):
        if gender == "female":
            return "longxiaochun_v3"  # 知性积极女
        return "longanyun_v3"  # 居家暖男

    # 默认
    if gender == "female":
        return "longanhuan"  # 欢脱元气女
    return "longanyang"  # 阳光大男孩


def get_voice_by_style(
    language: str = "zh",
    gender: str = "female",
    style: str = "自然",
) -> str:
    """按风格推荐音色。

    Args:
        language: 语言
        gender: 性别
        style: 风格（自然/温柔/沉稳/活泼）

    Returns:
        音色ID
    """
    for vid, info in PRETRAINED_SPEAKERS.items():
        if info["lang"] == language and info["gender"] == gender and info["style"] == style:
            return vid

    # Fallback: 返回同语言同性别的第一个
    for vid, info in PRETRAINED_SPEAKERS.items():
        if info["lang"] == language and info["gender"] == gender:
            return vid

    return "zh_female_1"  # Ultimate fallback


# ── 增强 CosyVoiceBackend ──────────────────────────────────────────────────

# 给 CosyVoiceBackend 添加便捷方法
def _enhance_backend():
    """Add convenience methods to CosyVoiceBackend."""

    def list_voices_for_backend(self, language=None, gender=None):
        """列出可用音色。"""
        return list_voices(language, gender)

    def synthesize_with_style(
        self, text, language="zh", gender="female", style="自然",
        speed=None, seed=None
    ):
        """按风格合成语音。

        Args:
            text: 文本
            language: 语言
            gender: 性别
            style: 风格
            speed: 语速
            seed: 随机种子

        Returns:
            AudioResult
        """
        speaker = get_voice_by_style(language, gender, style)
        return self.synthesize(text, speaker=speaker, speed=speed, seed=seed)

    # 动态添加方法
    CosyVoiceBackend.list_voices = list_voices_for_backend
    CosyVoiceBackend.synthesize_with_style = synthesize_with_style

_enhance_backend()
