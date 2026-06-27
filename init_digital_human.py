"""
别样觉醒 · AwakeEngine — 一键初始化

从聊天记录到完整数字人，一条命令搞定：
  聊天记录 → 行为分析 → 人格蒸馏 → 声音画像 → 数字形象 → 数字人上线

用法：
  # 从飞书聊天记录初始化
  python3 init_digital_human.py --name "张三" --input feishu_export.json --platform feishu

  # 从微信聊天记录初始化
  python3 init_digital_human.py --name "张三" --input wechat_export.html --platform wechat

  # 从CSV初始化
  python3 init_digital_human.py --name "张三" --input chat.csv --platform csv

  # 跳过形象生成（不需要GPU）
  python3 init_digital_human.py --name "张三" --input chat.json --platform feishu --skip-avatar

  # 只生成人格画像（不生成数字人）
  python3 init_digital_human.py --name "张三" --input chat.json --platform feishu --persona-only

作者：AtomCollide-智械工坊
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


# ── 步骤1: 导入聊天记录 ──────────────────────────────────────────────────────

def step_import(input_path: str, platform: str, person_name: str, output_dir: str) -> list:
    """导入聊天记录，返回标准化消息列表"""
    print(f"\n{'='*50}")
    print(f"📥 步骤1/5: 导入聊天记录")
    print(f"{'='*50}")
    print(f"  平台: {platform}")
    print(f"  文件: {input_path}")
    print(f"  目标人物: {person_name}")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
    from chat_importer import ChatImporter

    importer = ChatImporter()
    result = importer.import_file(input_path, platform=platform)

    if not result or not result.messages:
        print("  ❌ 导入失败或无消息")
        return []

    # 筛选目标人物的消息
    person_messages = [m for m in result.messages if person_name in getattr(m, "sender", "")]
    if not person_messages:
        print(f"  ⚠️ 未找到 {person_name} 的消息，使用全部消息")
        person_messages = result.messages

    print(f"  ✅ 导入成功: {len(result.messages)} 条消息, {person_name} 发送 {len(person_messages)} 条")

    # 保存导入结果
    import_path = os.path.join(output_dir, "imported_messages.json")
    with open(import_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"sender": getattr(m, "sender", ""), "content": getattr(m, "content", ""), "timestamp": str(getattr(m, "timestamp", ""))} for m in person_messages],
            f, ensure_ascii=False, indent=2,
        )
    print(f"  💾 保存到: {import_path}")

    return person_messages


# ── 步骤2: 行为分析 ──────────────────────────────────────────────────────────

def step_analyze(messages: list, person_name: str, output_dir: str) -> dict:
    """分析聊天记录，提取行为模式"""
    print(f"\n{'='*50}")
    print(f"🔍 步骤2/5: 行为模式分析")
    print(f"{'='*50}")

    from conversation_analyzer import analyze_messages

    profile = analyze_messages(
        [{"sender": getattr(m, "sender", ""), "content": getattr(m, "content", "")} for m in messages],
        person_name=person_name,
    )

    print(f"  消息总数: {profile.total_messages}")
    print(f"  平均长度: {profile.avg_message_length:.0f}字")
    print(f"  活跃时段: {', '.join(f'{h}:00' for h in profile.peak_hours[:3])}")
    print(f"  口头禅: {', '.join(profile.signature_phrases[:5])}")
    print(f"  回复风格: {profile.response_style}")

    # 保存分析结果
    analysis_path = os.path.join(output_dir, "persona_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": profile.name,
            "total_messages": profile.total_messages,
            "avg_message_length": profile.avg_message_length,
            "peak_hours": profile.peak_hours,
            "signature_phrases": profile.signature_phrases,
            "response_style": profile.response_style,
            "emotional_tone": profile.emotional_tone,
            "relationships": profile.relationships,
            "patterns": [{"type": p.pattern_type, "desc": p.description, "confidence": p.confidence} for p in profile.patterns],
        }, f, ensure_ascii=False, indent=2)
    print(f"  💾 保存到: {analysis_path}")

    print(f"  ✅ 分析完成")

    return {
        "name": profile.name,
        "total_messages": profile.total_messages,
        "avg_length": profile.avg_message_length,
        "peak_hours": profile.peak_hours,
        "signature_phrases": profile.signature_phrases,
        "response_style": profile.response_style,
        "emotional_tone": profile.emotional_tone,
        "relationships": profile.relationships,
        "patterns": profile.patterns,
    }


# ── 步骤3: 生成人格画像 ──────────────────────────────────────────────────────

def step_generate_persona(analysis: dict, person_name: str, output_dir: str) -> str:
    """从分析结果生成 persona.md"""
    print(f"\n{'='*50}")
    print(f"🧬 步骤3/5: 生成人格画像")
    print(f"{'='*50}")

    # 推导人格特征
    style = analysis.get("response_style", "medium")
    phrases = analysis.get("signature_phrases", [])
    tone = analysis.get("emotional_tone", {})
    patterns = analysis.get("patterns", [])

    # 生成性格描述
    personality_traits = []
    if style == "short":
        personality_traits.append("言简意赅，不废话")
    elif style == "long":
        personality_traits.append("表达详细，善于解释")

    if tone.get("positive", 0) > 0.6:
        personality_traits.append("积极乐观")
    if tone.get("humor", 0) > 0.3:
        personality_traits.append("幽默风趣")

    # 生成 persona.md
    persona_md = f"""# {person_name} · 数字人格画像

*由别样觉醒 · AwakeEngine 自动生成*
*生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}*
*数据源: {analysis.get('total_messages', 0)} 条聊天消息*

---

## 基本信息

- **姓名**: {person_name}
- **消息总数**: {analysis.get('total_messages', 0)}
- **平均消息长度**: {analysis.get('avg_length', 0):.0f}字
- **活跃时段**: {', '.join(f'{h}:00' for h in analysis.get('peak_hours', [])[:3])}

## 核心性格

{chr(10).join(f'- {t}' for t in personality_traits) if personality_traits else '- 待补充（需要更多聊天数据）'}

## 表达风格

- **回复风格**: {style}
- **口头禅**: {', '.join(phrases[:5]) if phrases else '待观察'}
- **典型句式**: {', '.join(set(p.description for p in patterns if p.pattern_type == 'vocabulary')[:3]) if patterns else '待分析'}

## 情感倾向

{chr(10).join(f'- **{k}**: {v:.0%}' for k, v in tone.items()) if tone else '- 待分析'}

## 高频互动对象

{chr(10).join(f'- {name}: {count}次' for name, count in sorted(analysis.get('relationships', {}).items(), key=lambda x: -x[1])[:5]) if analysis.get('relationships') else '- 暂无数据'}

## 行为模式

{chr(10).join(f'- [{p.pattern_type}] {p.description} (置信度: {p.confidence:.0%})' for p in patterns[:8]) if patterns else '- 待分析'}

---

## 执行规则

1. 用 {person_name} 的风格说话
2. 用 {person_name} 的框架做判断
3. 保持人格一致性，不跳出角色
4. 遇到未知话题时，按 {person_name} 的性格推测回应方式

---

*此画像由别样觉醒自动生成，可通过更多聊天数据持续优化。*
"""

    persona_path = os.path.join(output_dir, "persona.md")
    with open(persona_path, "w", encoding="utf-8") as f:
        f.write(persona_md)

    print(f"  ✅ persona.md 已生成 ({len(persona_md)}字)")
    print(f"  💾 保存到: {persona_path}")

    return persona_path


# ── 步骤4: 生成声音画像 ──────────────────────────────────────────────────────

def step_voice_profile(analysis: dict, person_name: str, output_dir: str) -> dict:
    """从分析结果生成声音画像"""
    print(f"\n{'='*50}")
    print(f"🎙️ 步骤4/5: 生成声音画像")
    print(f"{'='*50}")

    from voice_profile import create_voice_profile_from_persona, recommend_voice

    # 构造persona数据
    persona_data = {
        "name": person_name,
        "language": "zh",
        "gender": "male",  # 默认，后续可配置
        "speaking_style": {
            "description": analysis.get("response_style", ""),
            "pace": "快" if analysis.get("avg_length", 50) < 30 else "medium",
        },
        "personality": {
            "tone": "直接",
            "energy": "high" if analysis.get("emotional_tone", {}).get("positive", 0) > 0.5 else "medium",
        },
    }

    profile = create_voice_profile_from_persona(persona_data)

    print(f"  推荐音色: {profile.voice_id}")
    print(f"  语速: {profile.speed}")
    print(f"  音调: {profile.pitch}")
    print(f"  情感: {profile.emotion}")

    # 保存
    voice_path = os.path.join(output_dir, "voice_profile.json")
    with open(voice_path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"  ✅ 声音画像已生成")
    print(f"  💾 保存到: {voice_path}")

    return profile.to_dict()


# ── 步骤5: 生成SKILL.md ──────────────────────────────────────────────────────

def step_generate_skill(analysis: dict, person_name: str, output_dir: str) -> str:
    """生成个性化的SKILL.md"""
    print(f"\n{'='*50}")
    print(f"📝 步骤5/5: 生成SKILL.md")
    print(f"{'='*50}")

    phrases = analysis.get("signature_phrases", [])
    style = analysis.get("response_style", "medium")

    skill_md = f"""---
name: awake-differently
description: "基于聊天记录深度蒸馏的数字人系统。{person_name}的数字分身，支持实时对话交互。"
version: 2.0.0
author: "{person_name}"
triggers:
  - 数字人
  - digital human
  - {person_name}
  - 数字分身
  - 别样觉醒
  - awake
---

# 别样觉醒 · {person_name}的数字分身

> 由 AwakeEngine 自动生成 | 基于 {analysis.get('total_messages', 0)} 条聊天消息蒸馏

## 本集人物：{person_name}

### 基本特征

- **消息总数**: {analysis.get('total_messages', 0)}
- **回复风格**: {style}
- **口头禅**: {', '.join(phrases[:5]) if phrases else '待观察'}
- **活跃时段**: {', '.join(f'{h}:00' for h in analysis.get('peak_hours', [])[:3])}

### 表达风格

- {'言简意赅，一两句话说完' if style == 'short' else '表达详细，善于解释' if style == 'long' else '表达适中'}

### 执行规则

1. 用 {person_name} 的风格说话
2. 保持人格一致性
3. 遇到未知话题时按性格推测回应

## 核心能力

| 模块 | 功能 |
|------|------|
| 人格蒸馏 | 聊天记录→行为模式→人格画像 |
| 声音画像 | 人格特征→TTS参数→音色推荐 |
| 数字形象 | 视频/照片→数字人形象 |
| 实时交互 | WebRTC/RTMP实时对话 |

## Quick Start

1. 检查引擎状态: `python3 scripts/digital_avatar.py --check`
2. 生成数字人形象: `python3 scripts/avatar_builder.py --video your_video.mp4 --avatar-id {person_name.lower().replace(' ', '_')}`
3. 让数字人说话: `python3 scripts/digital_avatar.py --say "你好"`
4. 实时对话: `python3 scripts/digital_avatar.py --chat "你好"`

## 工作流

- [ ] 1. 确认引擎服务已启动
- [ ] 2. 上传说话视频生成形象
- [ ] 3. 配置声音画像
- [ ] 4. 开始实时对话

*由别样觉醒 · AwakeEngine 自动生成*
"""

    skill_path = os.path.join(output_dir, "SKILL.md")
    with open(skill_path, "w", encoding="utf-8") as f:
        f.write(skill_md)

    print(f"  ✅ SKILL.md 已生成 ({len(skill_md)}字)")
    print(f"  💾 保存到: {skill_path}")

    return skill_path


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="别样觉醒 · 一键初始化 — 从聊天记录到完整数字人",
        epilog="""
示例:
  # 从飞书聊天记录初始化
  python3 init_digital_human.py --name "张三" --input feishu_export.json --platform feishu

  # 从微信聊天记录初始化
  python3 init_digital_human.py --name "张三" --input wechat_export.html --platform wechat

  # 只生成人格画像（不需要GPU）
  python3 init_digital_human.py --name "张三" --input chat.json --platform feishu --persona-only
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", "-n", required=True, help="目标人物姓名")
    parser.add_argument("--input", "-i", required=True, help="聊天记录文件路径")
    parser.add_argument("--platform", "-p", required=True, choices=["feishu", "wechat", "telegram", "csv"], help="聊天平台")
    parser.add_argument("--output", "-o", default="./digital_human_output", help="输出目录")
    parser.add_argument("--persona-only", action="store_true", help="只生成人格画像，不生成数字人组件")
    parser.add_argument("--skip-avatar", action="store_true", help="跳过形象生成（不需要GPU）")

    args = parser.parse_args()

    # 验证输入
    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════╗
║  别样觉醒 · AwakeEngine — 一键初始化            ║
║  从聊天记录到完整数字人                          ║
╚══════════════════════════════════════════════════╝

  目标人物: {args.name}
  输入文件: {args.input}
  聊天平台: {args.platform}
  输出目录: {args.output}
  模式: {'仅人格画像' if args.persona_only else '完整初始化'}
""")

    # 执行流程
    try:
        # 步骤1: 导入
        messages = step_import(args.input, args.platform, args.name, args.output)
        if not messages:
            print("\n❌ 导入失败，终止")
            sys.exit(1)

        # 步骤2: 分析
        analysis = step_analyze(messages, args.name, args.output)

        # 步骤3: 生成人格
        persona_path = step_generate_persona(analysis, args.name, args.output)

        if args.persona_only:
            print(f"\n{'='*50}")
            print(f"✅ 人格画像生成完成!")
            print(f"{'='*50}")
            print(f"  persona.md: {persona_path}")
            print(f"\n  如需生成完整数字人，去掉 --persona-only 重新运行")
            return

        # 步骤4: 声音画像
        voice = step_voice_profile(analysis, args.name, args.output)

        # 步骤5: SKILL.md
        skill_path = step_generate_skill(analysis, args.name, args.output)

        # 完成
        print(f"\n{'='*50}")
        print(f"🎉 数字人初始化完成!")
        print(f"{'='*50}")
        print(f"""
  输出文件:
    📄 persona.md      — 人格画像
    📄 voice_profile.json — 声音画像
    📄 SKILL.md        — 技能配置
    📄 persona_analysis.json — 分析数据
    📄 imported_messages.json — 导入数据

  下一步:
    1. 检查 persona.md，按需调整人格描述
    2. 部署数字人引擎服务
    3. 上传说话视频生成数字人形象:
       python3 scripts/avatar_builder.py --video your_video.mp4 --avatar-id {args.name.lower().replace(' ', '_')}
    4. 开始实时对话:
       python3 scripts/digital_avatar.py --say "你好，我是{args.name}"
""")

    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
