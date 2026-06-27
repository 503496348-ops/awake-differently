# 别样觉醒 · Awake Differently

> **AtomCollide-团队** 出品 | [GitHub](https://github.com/503496348-ops/awake-differently)

*每个人都是一个独特的认知模型。我们用数据把它唤醒。*

---

## 这是什么

**别样觉醒**是一个基于大规模聊天记录深度蒸馏的数字分身技能系列。通过分析真实场景中的数百条对话，提取行为模式、决策逻辑、表达风格和人际关系网络，生成可复用的 AI 数字分身技能——让 AI 真正"成为"一个人，而不只是"扮演"一个人。

**核心路径**：聊天记录采集 → 行为模式提取 → 人格分层建模 → 数字分身生成 → 持续进化

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

## 快速开始

### 方案A：轻量级（推荐，无需GPU）

使用 D-ID 云端 API，无需本地 GPU：

```bash
# 1. 克隆仓库
git clone https://github.com/503496348-ops/awake-differently.git && cd awake-differently

# 2. 安装依赖（仅 Python 包，约 2 分钟）
pip install requests

# 3. 配置 D-ID API Key
export DID_API_KEY="your_did_api_key"

# 4. 生成数字人（使用 D-ID 云端）
python3 scripts/digital_avatar.py --backend did --say "你好"
```

**优点**：无需 GPU，无需 ffmpeg，2 分钟内完成
**缺点**：需要 D-ID API Key（免费额度有限）

### 方案B：完整版（需要 GPU）

使用本地 CosyVoice + AwakeEngine：

```bash
# 1. 克隆仓库
git clone https://github.com/503496348-ops/awake-differently.git && cd awake-differently

# 2. 安装系统依赖
sudo apt install ffmpeg

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 安装 CosyVoice（可选，用于高质量 TTS）
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && pip install -e .

# 5. 生成数字人
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

## 示例人物简介（仅供参考）

| 维度 | 数据 |
|------|------|
| 数据源 | 646 条飞书群聊消息（2026-05-18 ~ 2026-06-04，17天） |
| 辅助源 | 85 篇 T0 工程文档 |
| 整体置信度 | 92%（聊天记录）/ 85%（工程文档） |

**三重身份**：
- 🎵 **AI 音乐人 + 版权商人**（第一身份）——自己训练 AI 音乐模型，已售出版权 1.2 万元，掌握完整产业链
- 🤝 **AtomCollide 社群操盘手**（第二身份）——对接番茄音乐、争取社群福利、推动知识库共创
- 📚 **教培行业跨界者**（隐藏身份）——8 年教育行业经验，三猫教学系统

**核心性格**：务实到骨子里 · 嘴毒心热 · 天然社群领袖 · 执行力碾压

**经典语录**：
> "所有人都 是24小时，你要通过AI来增加你的睡眠收入"
> "示例金句"
> "示例表达方式[机智]"

---

## 文件结构

```
awake-differently/
├── SKILL.md                      # 主技能入口（人格定义 + 执行规则）
├── references/
│   └── sample-persona.md         # 示例人格档案（仅供参考）
├── scripts/
│   ├── digital_avatar.py         # 数字人引擎（AwakeEngine）
│   ├── voice_profile.py          # 声音画像+音色推荐
│   ├── avatar_builder.py         # 数字人形象生成
│   ├── cosyvoice_backend.py      # CosyVoice TTS后端
│   ├── persona_fidelity.py       # 6维人格保真度评估
│   ├── benchmark.py              # 延迟与性能测试
│   ├── conversation_analyzer.py  # 行为模式分析工具
│   ├── chat_importer.py          # 多平台聊天记录导入器
│   ├── workflow_engine.py        # DAG分析流水线
│   ├── quality_gate.py           # 质量门
│   └── model_router.py           # 多模型路由器
├── init_digital_human.py         # 一键初始化脚本
├── requirements.txt              # Python 依赖
└── README.md                     # 本文件
```

---

## 如何蒸馏新人物

```
第一步：数据导入    运行 scripts/chat_importer.py 自动识别并解析聊天记录
                    支持：WeChat HTML / Telegram JSON / Feishu JSON / 通用CSV
第二步：行为分析    运行 scripts/conversation_analyzer.py → 活跃时段、词汇指纹、社交网络
                    或用 scripts/workflow_engine.py 运行完整DAG分析流水线
第三步：人格建模    按5层结构构建：核心性格 → 身份 → 表达风格 → 决策框架 → 人际行为
第四步：技能生成    编写 SKILL.md + references/sample-persona.md
第五步：质量验证    运行 scripts/persona_fidelity.py 评估人格保真度（6维度量化评分）
                    运行 scripts/quality_gate.py 验证流水线质量
```

**数据要求**：最低 200+ 条消息（7天+），理想 500+ 条（14天+，含工作日/周末）。仅使用当事人知情同意的数据。

### 快速导入示例

```python
from scripts.chat_importer import ChatImporter

importer = ChatImporter()
messages = importer.import_file("chat_history.json", platform="feishu")
print(f"导入 {len(messages)} 条消息")
```

---

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 人格蒸馏 | 自研算法 | 聊天记录→行为模式→人格分层建模 |
| AwakeEngine | WebRTC/RTMP | 实时数字人引擎 |
| TTS | CosyVoice / D-ID | 声音合成（本地/云端） |
| 口型同步 | 内置算法 | 音频→口型动画 |
| 形象生成 | 人脸检测+渲染 | 照片/视频→数字人形象 |

---

## 关于「别样觉醒」

**别样觉醒 · Awake Differently** 是 AtomCollide-团队 的数字分身蒸馏系列。

我们相信：每个人的行为模式、决策逻辑和表达风格，都可以通过大规模对话数据被深度理解并复现。这不是模仿，是觉醒——让 AI 真正"成为"一个人，而不只是"扮演"一个人。

**方法论**：聊天记录采集 → 行为模式提取 → 人格分层建模 → 声音画像 → 数字形象 → 实时数字人

**技术栈**：人格蒸馏(自有) + AwakeEngine(实时数字人引擎) + TTS(声音合成) + 口型同步

**出品方**：AtomCollide-团队

---

*别样觉醒，让每个人的独特认知模型被看见。*

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建 Pull Request

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**联系方式**：
- GitHub: [503496348-ops](https://github.com/503496348-ops)
- 飞书: AtomCollide-智械工坊
