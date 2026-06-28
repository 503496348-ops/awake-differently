---
name: awake-differently
description: "基于聊天记录深度蒸馏的数字人系统。人格蒸馏+声音克隆+数字形象+实时交互四合一。当需要从聊天记录中提取人物特征、生成数字分身、创建数字人、实时对话交互时使用。"
argument-hint: "[question or task]"
version: 2.2.0
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

## 🚨 使用边界与法律声明

### 必须遵守的使用边界

1. **只能处理你自己的聊天记录**
   - 禁止未经他人授权处理他人的聊天数据
   - 禁止克隆他人的声音或形象
   - 违反者承担全部法律责任

2. **声音克隆的法律风险**
   - 声音权受法律保护，未经本人授权克隆声音可能构成侵权
   - 本工具仅限用于你自己的声音或已获授权的声音
   - 禁止用于伪造、欺诈等非法用途

3. **数字形象的法律风险**
   - 肖像权受法律保护，未经本人授权使用他人形象可能构成侵权
   - 本工具仅限用于你自己的形象或已获授权的形象
   - 禁止用于伪造、欺诈等非法用途

4. **人格蒸馏的安全边界**
   - 人格蒸馏是**工具**，不是**自动运行的功能**
   - 必须由用户**主动触发**（运行 `init_digital_human.py`）
   - 不会自动读取或分析你的对话
   - 不会覆盖你现有的 MEMORY.md 或 soul.md

### 免责声明

- 本工具仅供合法、合规的个人使用
- 用户需自行确保使用行为符合当地法律法规
- 开发团队不对用户的滥用行为承担责任
- 如有任何法律疑问，请咨询专业律师

---

## 核心能力

| 模块 | 功能 |
|------|------|
| 人格蒸馏 | 聊天记录→行为模式→人格分层建模 |
| 声音画像 | 人格特征→TTS参数→音色推荐/声音克隆 |
| 数字形象 | 视频/照片→人脸检测→数字人形象生成 |
| 实时交互 | WebRTC/RTMP/虚拟摄像头实时对话 |

## 技术架构

| 模块 | 实现 | 职责 |
|------|------|------|
| 人格蒸馏 | `persona_distiller.py` (1188行) | 聊天记录→行为指纹→persona.json |
| 聊天导入 | `chat_importer.py` | 微信/Telegram/飞书多格式解析 |
| 工作流引擎 | `workflow_engine.py` | ComfyUI风格DAG节点编排 |
| 声音画像 | `voice_profile.py` + `cosyvoice_backend.py` | 声纹提取→CosyVoice/GPT-SoVITS TTS |
| 数字分身 | `digital_avatar.py` | 3D形象渲染+表情驱动 |
| 实时交互 | `realtime_interaction.py` | SSE流式对话+多模态响应 |
| 忠实度评估 | `persona_fidelity.py` | 蒸馏质量量化评分 |

Pipeline: 聊天导入 → 人格蒸馏 → 声音画像 → 数字分身 → 实时交互

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

## 快速开始

### 方案A：轻量级（推荐，无需GPU）

使用 D-ID 云端 API，无需本地 GPU：

```bash
# 1. 安装依赖（仅 Python 包，约 2 分钟）
pip install requests

# 2. 配置 D-ID API Key
export DID_API_KEY="your_did_api_key"

# 3. 生成数字人（使用 D-ID 云端）
python3 scripts/digital_avatar.py --backend did --say "你好"
```

**优点**：无需 GPU，无需 ffmpeg，2 分钟内完成
**缺点**：需要 D-ID API Key（免费额度有限）

### 方案B：完整版（需要 GPU）

使用本地 CosyVoice + AwakeEngine：

```bash
# 1. 安装系统依赖
sudo apt install ffmpeg

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 安装 CosyVoice（可选，用于高质量 TTS）
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && pip install -e .

# 4. 生成数字人
python3 scripts/digital_avatar.py --say "你好"
```

**优点**：完全本地化，无 API 费用
**缺点**：需要 RTX 3060+ GPU，安装时间 30-60 分钟

### 方案C：一键初始化（从聊天记录创建）

```bash
# 从聊天记录生成你自己的数字人
python3 init_digital_human.py --name "你的名字" --input chat_export.json --platform feishu
```

**注意**：此命令会：
1. 分析你的聊天记录（需要你主动提供）
2. 生成人格画像（保存在 `data/persona.yaml`）
3. 推荐音色（基于人格特征）
4. 生成 SKILL.md（不会覆盖你现有的身份）

---

## 安装路径指南

### 推荐安装位置

```bash
# 方案1：作为独立技能安装（推荐）
~/.hermes/skills/awake-differently/

# 方案2：作为项目仓库安装
~/projects/awake-differently/
```

### 避免冲突

- **不要**安装在 `~/.hermes/skills/` 的子目录中（会和其他技能冲突）
- **不要**覆盖现有的 `persona.md` 或 `work.md` 文件
- 安装前检查是否有同名文件：

```bash
# 检查是否有冲突
ls ~/.hermes/skills/ | grep -i "persona\|work\|distillation"
```

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
