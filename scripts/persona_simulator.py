"""
别样觉醒 — 人格模拟测试引擎
用蒸馏出的人格画像生成模拟对话，验证蒸馏质量。
"""
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class SimulatedMessage:
    role: str  # "persona" or "interlocutor"
    content: str
    timestamp: str = ""
    confidence: float = 0.0  # how well this matches the persona


@dataclass
class SimulationResult:
    scenario: str
    messages: List[SimulatedMessage]
    persona_consistency_score: float = 0.0  # 0-1
    diversity_score: float = 0.0  # 0-1
    naturalness_score: float = 0.0  # 0-1
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "messages": [{"role": m.role, "content": m.content, "confidence": m.confidence} for m in self.messages],
            "scores": {
                "consistency": round(self.persona_consistency_score, 3),
                "diversity": round(self.diversity_score, 3),
                "naturalness": round(self.naturalness_score, 3),
            },
            "summary": self.summary,
        }


# Scenario templates for testing different persona dimensions
SCENARIO_TEMPLATES = {
    "work_discussion": {
        "description": "工作会议讨论",
        "prompts": [
            "你觉得这个方案怎么样？",
            "时间紧，我们能按时交付吗？",
            "客户反馈不太好，怎么处理？",
            "这个技术选型你同意吗？",
        ],
    },
    "casual_chat": {
        "description": "闲聊",
        "prompts": [
            "最近在忙什么？",
            "周末干嘛了？",
            "推荐个电影/音乐呗",
            "你觉得AI会取代人类吗？",
        ],
    },
    "conflict_resolution": {
        "description": "冲突处理",
        "prompts": [
            "我不同意你的看法",
            "你这样做让我很不舒服",
            "我们得谈谈上次的事",
            "你为什么不提前说？",
        ],
    },
    "knowledge_sharing": {
        "description": "知识分享",
        "prompts": [
            "你能解释一下这个概念吗？",
            "你是怎么学会这个的？",
            "有什么学习建议？",
            "这个领域有什么坑要注意？",
        ],
    },
    "emotional_support": {
        "description": "情感支持",
        "prompts": [
            "今天心情不太好",
            "我遇到瓶颈了，很迷茫",
            "你觉得我应该坚持吗？",
            "压力好大，怎么调节？",
        ],
    },
}


class PersonaSimulator:
    """用蒸馏人格画像生成模拟对话并评分。"""

    def __init__(self, persona_path: str = "persona.md"):
        self.persona_path = persona_path
        self.persona_traits = self._load_persona()

    def _load_persona(self) -> dict:
        """从 persona.md 提取人格特征。"""
        try:
            with open(self.persona_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return {"raw": "", "traits": [], "speech_patterns": [], "topics": []}

        traits = []
        speech_patterns = []
        topics = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Extract personality traits
            if any(kw in line for kw in ["性格", "风格", "特点", "特征"]):
                traits.append(line)
            # Extract speech patterns
            if any(kw in line for kw in ["口头禅", "语气", "说话", "表达", "语言"]):
                speech_patterns.append(line)
            # Extract topics of interest
            if any(kw in line for kw in ["兴趣", "爱好", "关注", "领域", "专业"]):
                topics.append(line)

        return {"raw": content, "traits": traits, "speech_patterns": speech_patterns, "topics": topics}

    def simulate_conversation(
        self,
        scenario: str = "casual_chat",
        turns: int = 6,
        llm_client=None,
    ) -> SimulationResult:
        """模拟一段对话。"""
        template = SCENARIO_TEMPLATES.get(scenario, SCENARIO_TEMPLATES["casual_chat"])
        messages = []

        persona_context = f"""你是以下人物，请用TA的风格回复：
{self.persona_traits['raw'][:2000]}

要求：保持角色一致性，回复要自然、有个性。"""

        for i in range(turns):
            # Interlocutor speaks
            prompt = random.choice(template["prompts"])
            messages.append(SimulatedMessage(role="interlocutor", content=prompt))

            # Persona responds
            if llm_client:
                history = [{"role": m.role, "content": m.content} for m in messages[-4:]]
                response = llm_client(persona_context, history)
            else:
                response = f"[需要LLM客户端生成回复] 场景: {scenario}, 提示: {prompt}"

            messages.append(SimulatedMessage(role="persona", content=response, confidence=0.8))

        return SimulationResult(
            scenario=f"{scenario} ({template['description']})",
            messages=messages,
            persona_consistency_score=self._score_consistency(messages),
            diversity_score=self._score_diversity(messages),
            naturalness_score=0.0,  # needs LLM to evaluate
            summary=f"模拟了 {turns//2} 轮{template['description']}对话",
        )

    def _score_consistency(self, messages: List[SimulatedMessage]) -> float:
        """评估角色一致性（基于关键词匹配）。"""
        persona_msgs = [m for m in messages if m.role == "persona"]
        if not persona_msgs:
            return 0.0

        trait_keywords = set()
        for trait in self.persona_traits.get("traits", []):
            trait_keywords.update(trait.split())

        if not trait_keywords:
            return 0.5  # no traits to match

        matches = 0
        total = 0
        for msg in persona_msgs:
            words = set(msg.content)
            total += 1
            if words & trait_keywords:
                matches += 1

        return min(1.0, matches / max(total, 1) + 0.3)  # baseline 0.3

    def _score_diversity(self, messages: List[SimulatedMessage]) -> float:
        """评估回复多样性（不重复）。"""
        persona_msgs = [m for m in messages if m.role == "persona"]
        if len(persona_msgs) < 2:
            return 1.0

        # Simple: check if responses are different lengths and content
        lengths = [len(m.content) for m in persona_msgs]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        diversity = min(1.0, (variance ** 0.5) / max(avg_len, 1))

        return round(diversity, 3)

    def run_full_test(self, llm_client=None) -> List[SimulationResult]:
        """运行所有场景的完整测试。"""
        results = []
        for scenario in SCENARIO_TEMPLATES:
            result = self.simulate_conversation(scenario=scenario, llm_client=llm_client)
            results.append(result)
        return results


if __name__ == "__main__":
    import sys

    persona_path = sys.argv[1] if len(sys.argv) > 1 else "persona.md"
    simulator = PersonaSimulator(persona_path)
    results = simulator.run_full_test()

    print(f"\n=== 人格模拟测试报告 ===")
    print(f"人格文件: {persona_path}")
    print(f"测试场景: {len(results)}\n")

    for r in results:
        print(f"📋 {r.scenario}")
        print(f"   一致性: {r.persona_consistency_score:.1%}")
        print(f"   多样性: {r.diversity_score:.1%}")
        print(f"   {r.summary}")
        print()

    avg_consistency = sum(r.persona_consistency_score for r in results) / len(results)
    print(f"综合一致性: {avg_consistency:.1%}")
