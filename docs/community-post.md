# 别样觉醒 · AwakeEngine — 你的聊天记录，能变成一个会说话的数字人

---

## 一句话介绍

把你的聊天记录丢进去，出来一个会用你的方式说话、会用你的方式思考、能实时视频对话的数字分身。

**不需要GPU，不需要写代码，一条命令搞定。**

---

## 它能做什么

| 能力 | 说明 |
|------|------|
| 🧬 **人格蒸馏** | 从聊天记录提取你的性格、口头禅、决策风格、回复习惯 |
| 🎙️ **声音画像** | 自动匹配最适合你风格的TTS音色，支持声音克隆 |
| 🎬 **数字形象** | 上传一段视频或照片，生成你的数字人形象 |
| 💬 **实时对话** | 浏览器里看到数字人用你的方式说话，口型同步 |

---

## 谁适合用

- **内容创作者**：批量生成数字人口播视频，不再真人出镜
- **教育工作者**：数字分身录制课程，AI讲师实时授课
- **社群操盘手**：数字人客服，24小时在线，用你的风格回复
- **个人IP**：数字分身代替你做直播、做短视频、做问答
- **技术团队**：集成到自己的产品里，提供数字人对话能力

---

## 怎么用（3步）

### 第1步：初始化人格

```bash
python3 init_digital_human.py --name "你的名字" --input 聊天记录.json --platform feishu
```

支持飞书、微信、Telegram、CSV等格式。自动分析你的聊天数据，生成人格画像。

### 第2步：生成数字人

**方式A：D-ID云端（推荐，无需GPU）**

```bash
python3 scripts/digital_avatar.py --backend did --did-key YOUR_KEY \
  --did-photo https://你的照片.jpg --say "你好"
```

**方式B：本地引擎（高级用户，实时交互）**

```bash
python3 scripts/digital_avatar.py --backend local --say "你好"
```

### 第3步：开始对话

浏览器里打开，看到数字人用你的方式说话。

---

## 技术架构

```
聊天记录 → 行为分析 → 人格蒸馏 → 声音画像 → 数字形象 → 实时数字人
                                                      ↓
                                            WebRTC / RTMP / MP4
```

| 组件 | 技术 | 说明 |
|------|------|------|
| 人格蒸馏 | 自研 | 聊天记录→性格/口头禅/决策框架 |
| 数字人引擎 | D-ID / AwakeEngine | 云端API或本地GPU |
| TTS | Edge TTS / GPT-SoVITS / CosyVoice | 多引擎可选 |
| 口型同步 | Wav2Lip / MuseTalk | 音频驱动口型 |
| 输出 | WebRTC / RTMP | 浏览器/直播/虚拟摄像头 |

---

## 两种部署方式

| 方式 | 需要GPU | 适合人群 | 特点 |
|------|---------|---------|------|
| **D-ID云端** | ❌ | 普通用户 | 注册即用，免费额度 |
| **本地引擎** | ✅ RTX 3060+ | 高级用户 | 实时交互，无限量 |

---

## 开源地址

**GitHub**: https://github.com/503496348-ops/awake-differently

**许可证**: MIT

**技术栈**: Python 3.10+

---

## 快速体验

```bash
# 克隆仓库
git clone https://github.com/503496348-ops/awake-differently.git
cd awake-differently

# 初始化你的人格
python3 init_digital_human.py --name "你的名字" --input 你的聊天记录.json --platform feishu

# 用D-ID生成数字人（无需GPU）
python3 scripts/digital_avatar.py --backend did --did-key YOUR_KEY --did-photo 你的照片.jpg --say "你好世界"
```

---

## 常见问题

**Q: 需要什么硬件？**
A: D-ID云端模式不需要GPU，有网就行。本地引擎需要RTX 3060以上显卡。

**Q: 支持哪些聊天平台？**
A: 飞书、微信、Telegram、CSV。后续会支持更多平台。

**Q: 数字人能做什么？**
A: 口播视频、实时对话、直播、客服。核心是它会用你的方式说话。

**Q: 数据安全吗？**
A: 人格蒸馏在本地完成，聊天数据不会上传。D-ID模式下照片和文本会经过D-ID服务器。

---

> 每个人都是一个独特的认知模型。我们用数据把它唤醒。

**别样觉醒 · AwakeEngine** — AtomCollide-智械工坊
