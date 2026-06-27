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

# 预训练音色（从 CosyVoice 模型中提取）
PRETRAINED_SPEAKERS = {
    # 中文音色
    "zh_female_1": {"name": "中文女", "lang": "zh", "gender": "female", "style": "自然"},
    "zh_female_2": {"name": "中文女-温柔", "lang": "zh", "gender": "female", "style": "温柔"},
    "zh_male_1": {"name": "中文男", "lang": "zh", "gender": "male", "style": "自然"},
    "zh_male_2": {"name": "中文男-沉稳", "lang": "zh", "gender": "male", "style": "沉稳"},
    # 英文音色
    "en_female_1": {"name": "英文女", "lang": "en", "gender": "female", "style": "自然"},
    "en_male_1": {"name": "英文男", "lang": "en", "gender": "male", "style": "自然"},
}


def list_voices(language: str = None, gender: str = None) -> dict:
    """列出可用音色，支持按语言/性别筛选。

    Args:
        language: 语言代码（zh/en/ja/ko/de/es/fr/it/ru）
        gender: 性别（male/female）

    Returns:
        音色字典 {id: {name, lang, gender, style}}
    """
    voices = PRETRAINED_SPEAKERS.copy()

    if language:
        voices = {k: v for k, v in voices.items() if v["lang"] == language}
    if gender:
        voices = {k: v for k, v in voices.items() if v["gender"] == gender}

    return voices


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
