---
name: awake-differently
description: "基于聊天记录深度蒸馏的数字人系统。人格蒸馏+声音克隆+数字形象+实时交互四合一。当需要从聊天记录中提取人物特征、生成数字分身、创建数字人、实时对话交互时使用。"
argument-hint: "[question or task]"
version: 2.1.0
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, WebSearch, WebFetch
author: "AtomCollide-团队"

triggers:
  - 身份蒸馏
  - AI persona
  - digital twin
  - 数字人
  - digital human
  - 数字分身
  - 虚拟形象
  - 声音克隆
  - voice clone
  - 实时对话
  - 数字人直播
  - awake
  - 别样觉醒
---

# 别样觉醒 · Awake Differently

> 📖 详细文档见 `references/` 目录

> **AtomCollide-团队** 出品

*每个人都是一个独特的认知模型。我们用数据把它唤醒。*

---

## ⚠️ 重要说明

**本技能是工具，不是人格。** 它不会覆盖你的身份或说话风格。

- 加载此技能 = 获得「创建数字分身」的能力
- **不会**让你变成某个特定的人
- 你的人格数据由 `init_digital_human.py` 从你的聊天记录中生成
- 示例人格档案见 `references/sample-persona.md`（仅供参考，不参与运行）

---

## 核心能力

| 模块 | 功能 |
|------|------|
| 人格蒸馏 | 聊天记录→行为模式→人格分层建模 |
| 声音画像 | 人格特征→TTS参数→音色推荐/声音克隆 |
| 数字形象 | 视频/照片→人脸检测→数字人形象生成 |
| 实时交互 | WebRTC/RTMP/虚拟摄像头实时对话 |

## 技能文件说明

| 文件 | 内容 |
|------|------|
| `SKILL.md` | 主技能入口（本文件） |
| `references/sample-persona.md` | 示例人格档案（仅供参考） |
| `scripts/digital_avatar.py` | 数字人引擎（AwakeEngine） |
| `scripts/voice_profile.py` | 声音画像+音色推荐 |
| `scripts/avatar_builder.py` | 数字人形象生成 |
| `scripts/cosyvoice_backend.py` | CosyVoice TTS后端 |
| `scripts/persona_fidelity.py` | 6维人格保真度评估 |
| `scripts/benchmark.py` | 延迟与性能测试 |
| `init_digital_human.py` | 一键初始化（聊天记录→完整数字人） |

## Quick Start

### 从零创建你自己的数字人

```bash
# 一键初始化：聊天记录 → 人格画像 → 声音画像 → 数字人
python3 init_digital_human.py --name "你的名字" --input chat_export.json --platform feishu
```

### 使用已有的数字人

1. 检查引擎状态: `python3 scripts/digital_avatar.py --check`
2. 生成数字人形象: `python3 scripts/avatar_builder.py --video your_video.mp4 --avatar-id my_avatar`
3. 配置声音画像: `python3 scripts/voice_profile.py --recommend --language zh --gender male`
4. 让数字人说话: `python3 scripts/digital_avatar.py --say "你好"`
5. 实时对话: `python3 scripts/digital_avatar.py --chat "你觉得AI能变现吗？"`

**依赖**: Python >= 3.10, requests >= 2.28, GPU(RTX 3060+)

---

## 关于「别样觉醒」

**别样觉醒 · Awake Differently** 是 AtomCollide-团队 的数字分身蒸馏系列。

我们相信：每个人的行为模式、决策逻辑和表达风格，都可以通过大规模对话数据被深度理解并复现。这不是模仿，是觉醒——让 AI 真正"成为"一个人，而不只是"扮演"一个人。

**方法论**：聊天记录采集 → 行为模式提取 → 人格分层建模 → 声音画像 → 数字形象 → 实时数字人

**技术栈**：人格蒸馏(自有) + AwakeEngine(实时数字人引擎) + TTS(声音合成) + 口型同步

**出品方**：AtomCollide-团队

---

*别样觉醒，让每个人的独特认知模型被看见。*

## 工作流

使用此技能时，按以下步骤执行：
- [ ] 1. 确认用户需求和使用场景
- [ ] 2. 加载相关代码和配置
- [ ] 3. 执行核心功能
- [ ] 4. 验证输出结果
- [ ] 5. 反馈给用户

## CosyVoice 集成 (v2.1.0)

### 新增能力
- **CosyVoice TTS 后端** — 阿里FunAudioLLM出品，中文质量最佳
- **4种推理模式**：预训练音色 / 3秒极速克隆 / 跨语种复刻 / 自然语言控制
- **飞书语音气泡** — 自动转Opus格式，支持飞书原生语音消息
- **零样本声音克隆** — 3秒参考音频即可克隆任意声音

### 快速使用

```bash
# 预训练音色
python3 scripts/cosyvoice_backend.py --text "你好世界" --speaker 中文女 --output output.wav

# 声音克隆
python3 scripts/cosyvoice_backend.py --text "你好世界" --clone-audio speaker.wav --clone-text "参考文本" --output cloned.wav

# 飞书语音气泡（Opus格式）
python3 scripts/cosyvoice_backend.py --text "你好世界" --opus --output output.opus

# 列出可用音色
python3 scripts/cosyvoice_backend.py --list-speakers
```

### Python API

```python
from scripts.cosyvoice_backend import CosyVoiceBackend, CosyVoiceConfig

backend = CosyVoiceBackend(CosyVoiceConfig(model_dir="pretrained_models/Fun-CosyVoice3-0.5B"))
backend.load()

# 预训练
result = backend.synthesize("你好世界", speaker="中文女")
result.save_wav("output.wav")
result.save_opus("output.opus")  # 飞书语音气泡

# 克隆
result = backend.synthesize_clone("你好世界", "speaker.wav", "参考文本")
```

### 环境要求
- GPU: RTX 3060+ (推荐) / CPU (可用但慢)
- Python: 3.10+
- ffmpeg: opus转换需要
- CosyVoice: `pip install -e .` from CosyVoice repo

## 音色选择

### 预训练音色
| ID | 名称 | 语言 | 性别 | 风格 |
|----|------|------|------|------|
| zh_female_1 | 中文女 | zh | female | 自然 |
| zh_female_2 | 中文女-温柔 | zh | female | 温柔 |
| zh_male_1 | 中文男 | zh | male | 自然 |
| zh_male_2 | 中文男-沉稳 | zh | male | 沉稳 |
| en_female_1 | 英文女 | en | female | 自然 |
| en_male_1 | 英文男 | en | male | 自然 |

### 支持语言
中文（普通话）、英语、日语、韩语、德语、西班牙语、法语、意大利语、俄语

### 中文方言
粤语、闽南语、四川话、东北话、上海话、天津话

### 使用方式

```python
from scripts.cosyvoice_backend import CosyVoiceBackend, list_voices, get_voice_by_style

# 列出所有音色
voices = list_voices()
voices_zh = list_voices(language='zh')
voices_female = list_voices(gender='female')

# 按风格推荐
voice = get_voice_by_style('zh', 'female', '温柔')

# 使用 Backend
backend = CosyVoiceBackend()
backend.load()
result = backend.synthesize("你好世界", speaker="zh_female_2")
result = backend.synthesize_with_style("你好世界", language='zh', gender='female', style='温柔')
```

### 自然语言控制（Instruct模式）

```python
# 用自然语言控制音色和情感
result = backend.synthesize_instruct(
    "你好世界",
    instruct="用温柔的语气说，语速稍慢",
    speaker="zh_female_1"
)
```

### 声音克隆（3秒极速克隆）

```python
# 用任意音频克隆声音
result = backend.synthesize_clone(
    "你好世界",
    prompt_audio="speaker.wav",  # 3-30秒参考音频
    prompt_text="参考音频的文本内容"
)
```
