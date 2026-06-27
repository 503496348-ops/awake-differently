# 别样觉醒 · Awake Differently

> **AtomCollide-团队** 出品 | [GitHub](https://github.com/503496348-ops/awake-differently)

*每个人都是一个独特的认知模型。我们用数据把它唤醒。*

---

## 这是什么

**别样觉醒**是一个基于大规模聊天记录深度蒸馏的数字分身技能系列。通过分析真实场景中的数百条对话，提取行为模式、决策逻辑、表达风格和人际关系网络，生成可复用的 AI 数字分身技能——让 AI 真正"成为"一个人，而不只是"扮演"一个人。

**核心路径**：聊天记录采集 → 行为模式提取 → 人格分层建模 → 数字分身生成 → 持续进化

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/503496348-ops/awake-differently.git && cd awake-differently

# 2. 安装（Python 3.10+，无额外依赖）
python3 --version

# 3. 查看人物档案
cat SKILL.md          # 主技能入口 + 完整人格定义
cat persona.md        # 5层人格结构 + 置信度标注
cat work.md           # 工作能力、技术栈、决策流程
cat distillation.md   # 原始蒸馏数据报告

# 4. 激活数字分身（放入 Agent 技能目录）
cp SKILL.md ~/.hermes/skills/awake-differently/SKILL.md
# 支持 Hermes Agent 等兼容宿主

# 5. 自定义新人物
# 准备聊天记录 JSON → 运行 python3 scripts/conversation_analyzer.py
# → 按模板生成 persona.md / work.md / distillation.md → 编写 SKILL.md
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
├── persona.md                    # 5层人格结构 + 置信度标注
├── work.md                       # 工作能力、技术栈、决策流程
├── distillation.md               # 原始蒸馏数据报告
├── scripts/
│   ├── conversation_analyzer.py  # 行为模式分析工具
│   ├── chat_importer.py          # 多平台聊天记录导入器（WeChat/Telegram/Feishu/CSV）
│   ├── persona_fidelity.py       # 人格保真度评估器（5维度量化评分）
│   ├── workflow_engine.py        # DAG分析流水线（可视化节点编排）
│   ├── quality_gate.py           # 质量门（流水线阶段间验证）
│   └── model_router.py           # 多模型路由器（成本/质量/延迟优化）
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
第四步：技能生成    编写 SKILL.md + work.md + distillation.md
第五步：质量验证    运行 scripts/persona_fidelity.py 评估人格保真度（5维度量化评分）
                    运行 scripts/quality_gate.py 验证流水线质量
```

**数据要求**：最低 200+ 条消息（7天+），理想 500+ 条（14天+，含工作日/周末）。仅使用当事人知情同意的数据。

### 快速导入示例

```python
from scripts.chat_importer import ChatImporter

importer = ChatImporter()
result = importer.import_file("chat_export.json")  # 自动检测平台
print(result.summary())

# 获取人物统计，选择分析目标
stats = importer.get_person_stats(result)
for name, info in stats.items():
    print(f"{name}: {info['message_count']}条 ({info['percentage']}%)")
```

### 保真度评估示例

```python
from scripts.persona_fidelity import PersonaFidelityEvaluator

evaluator = PersonaFidelityEvaluator()
report = evaluator.evaluate(
    original_messages=messages,
    persona_config={"catchphrases": ["示例口头禅1", "示例口头禅2"], "decision_priority": ["变现"]},
    persona_outputs=["这个可以搞，具体方案是……"],
)
print(report.summary_text())  # 综合得分 78/100 (等级C)
```

---

## FAQ

**Q: 跟普通 AI 角色扮演有什么区别？**
普通角色扮演基于描述"模拟"，别样觉醒基于数据"唤醒"。分析的是真实场景中的言行，每层人格都有数据支撑和置信度标注。

**Q: 数字分身准确率有多高？**
以示例数据为例，整体置信度 92%。已知局限：飞书 API 最多追溯 17 天、群聊仅反映公开社交面。置信度随数据量增加而提升。

**Q: 可以用在哪些场景？**
个人数字分身（用你的风格替你回复）、团队知识沉淀（核心成员经验数字化）、社群运营（真实人格的社群助手）、内容创作（一致性内容生成）。

**Q: 如何提升蒸馏质量？**
增加数据量和时间跨度；多场景交叉验证（群聊+私聊+工作文档）；引入当事人 Correction 反馈；定期更新数据保持同步进化。

---

**出品方**：AtomCollide-团队

*别样觉醒，让每个人的独特认知模型被看见。*

---



---

## 🚀 加入AtomCollide-AI智能体实验室

**元素碰撞-AtomCollide-AI 智能体实验室** 是一个专注于AI领域的开源组织，汇聚了众多优秀学习者。

### 核心价值

**找工作：更省力，也更精准**
- 一线大厂内推通道（字节、阿里、腾讯等）
- 全链路求职赋能包（面试题库、简历优化、晋升指导）
- 线下技术沙龙 & 人脉网络

**学AI测试：真正落地，拒绝空谈**
- 从0到1实战落地体系（Skills、MCP、RAG、AI IDE等）
- 独家自研资料与工具矩阵
- 前沿技术同步与提效方案

### 知识库

- [踩坑合集](https://vcnvmnln7wit.feishu.cn/wiki/CjV9wG8IHiIpWikCdFEcxfErnne)
- [商业化案例库](https://vcnvmnln7wit.feishu.cn/wiki/LdIxwlrKGibFEVkWMocc2K9KnBh)
- [科普专栏](https://vcnvmnln7wit.feishu.cn/wiki/K1RPwM8zji9ZchkxlOmcivUgnJe)
- [Open Build](https://vcnvmnln7wit.feishu.cn/wiki/CThswol0PiNJJbkhgT1cZIxanLb)
- [LLM/Agent/研究报告知识库](https://vcnvmnln7wit.feishu.cn/wiki/KwGQwS2TciT2EdkSBBtcYnbsnSd)
- [Skill封装合集](https://vcnvmnln7wit.feishu.cn/wiki/PDfpwqJZUibTyBkUa7TcZZ6Onpd)
- [社区治理运营知识库](https://vcnvmnln7wit.feishu.cn/wiki/MSEGwrdnTiiF9Dk8qCVcNW6InJg)

### 加入社群

| 社群 | 链接 |
|------|------|
| AI探索交流1区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=074vd565-6084-455c-ac52-9703e89a0697) |
| AI探索交流2区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=60bj94f0-1a67-48a7-abbb-9172b161c2b0) |
| AI探索交流3区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=13do1920-db46-4444-b635-005680beaf58) |
| AI探索交流4区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=f17o1b86-06f6-4f10-911a-69a299a25fe3) |
| AI探索交流5区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=2bbh6ab6-22c2-4753-b973-74bb1a2edcc9) |
| AI探索交流6区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=d19r19f7-2f47-42ba-b1ec-cb0342cf2e80) |
| AI探索交流7区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=fe9vdacc-7316-4b4d-ae4a-fdbcf56315e6) |
| AI探索交流8区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=103kfae8-1fd7-424f-984f-d66c210e42d1) |
| AI探索交流9区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=239p3cad-2f83-4baa-a230-f40386067548) |
| AI探索交流10区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=880r7cf5-3638-45ff-afb9-7944de991872) |
| AI探索交流-网文作家 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=6a3v579b-ab43-4e1a-87f9-be63bab88da7) |
| AI探索交流群-音乐达人 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=76at299e-73da-4eeb-9eba-32161e98f2f8) |
| AI探索交流群-微笑驿站 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=f2av73d0-6bb4-4a9f-9095-5fbbe83e49ec) |

---

*AtomCollide-智械工坊团队出品*

