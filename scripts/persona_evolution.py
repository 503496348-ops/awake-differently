"""
别样觉醒 — 人格进化追踪器
追踪人格画像随时间的变化，发现漂移和稳定特征。
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter


@dataclass
class PersonaSnapshot:
    """某时刻的人格快照。"""
    timestamp: str
    source_period: str  # e.g., "2026-05-18 ~ 2026-06-04"
    message_count: int
    traits: Dict[str, float] = field(default_factory=dict)  # trait_name → strength (0-1)
    topics: Dict[str, float] = field(default_factory=dict)  # topic → frequency
    sentiment: float = 0.0  # -1 to 1
    activity_pattern: Dict[str, int] = field(default_factory=dict)  # hour → count

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "source_period": self.source_period,
            "message_count": self.message_count,
            "traits": self.traits,
            "topics": self.topics,
            "sentiment": round(self.sentiment, 3),
            "activity_pattern": self.activity_pattern,
        }


@dataclass
class EvolutionDelta:
    """两个人格快照之间的变化。"""
    from_period: str
    to_period: str
    trait_changes: Dict[str, float] = field(default_factory=dict)  # trait → delta
    new_traits: List[str] = field(default_factory=list)
    lost_traits: List[str] = field(default_factory=list)
    topic_shifts: Dict[str, float] = field(default_factory=dict)
    sentiment_delta: float = 0.0
    stability_score: float = 0.0  # 0=完全变了, 1=完全没变

    def summary(self) -> str:
        lines = [f"人格变化: {self.from_period} → {self.to_period}"]
        if self.trait_changes:
            lines.append("  特征变化:")
            for trait, delta in sorted(self.trait_changes.items(), key=lambda x: abs(x[1]), reverse=True):
                direction = "↑" if delta > 0 else "↓"
                lines.append(f"    {direction} {trait}: {delta:+.2f}")
        if self.new_traits:
            lines.append(f"  新增特征: {', '.join(self.new_traits)}")
        if self.lost_traits:
            lines.append(f"  消失特征: {', '.join(self.lost_traits)}")
        lines.append(f"  稳定性: {self.stability_score:.1%}")
        return "\n".join(lines)


class PersonaEvolutionTracker:
    """追踪人格画像随时间的进化。"""

    def __init__(self, snapshots_dir: str = "evolution"):
        self.snapshots_dir = snapshots_dir
        self.snapshots: List[PersonaSnapshot] = []
        os.makedirs(snapshots_dir, exist_ok=True)
        self._load_existing()

    def _load_existing(self):
        """加载已有的快照。"""
        path = os.path.join(self.snapshots_dir, "snapshots.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for s in data:
                    self.snapshots.append(PersonaSnapshot(**s))

    def save(self):
        """保存所有快照。"""
        path = os.path.join(self.snapshots_dir, "snapshots.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.snapshots], f, ensure_ascii=False, indent=2)

    def add_snapshot(self, snapshot: PersonaSnapshot):
        """添加新快照。"""
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda s: s.timestamp)
        self.save()

    def create_snapshot_from_messages(
        self,
        messages: List[Dict],
        period_label: str = "",
    ) -> PersonaSnapshot:
        """从聊天记录创建人格快照。"""
        if not messages:
            return PersonaSnapshot(timestamp=datetime.now().isoformat(), source_period=period_label, message_count=0)

        # Extract traits
        traits = self._extract_traits(messages)
        topics = self._extract_topics(messages)
        sentiment = self._compute_sentiment(messages)
        activity = self._extract_activity(messages)

        snapshot = PersonaSnapshot(
            timestamp=datetime.now().isoformat(),
            source_period=period_label,
            message_count=len(messages),
            traits=traits,
            topics=topics,
            sentiment=sentiment,
            activity_pattern=activity,
        )

        self.add_snapshot(snapshot)
        return snapshot

    def _extract_traits(self, messages: List[Dict]) -> Dict[str, float]:
        """从消息中提取人格特征强度。"""
        trait_keywords = {
            "幽默": ["哈哈", "😂", "笑", "搞笑", "有趣", "逗"],
            "专业": ["技术", "方案", "架构", "分析", "方案", "优化"],
            "热情": ["!", "！", "太棒了", "厉害", "牛", "赞"],
            "严谨": ["数据", "证据", "验证", "测试", "确认", "准确"],
            "创新": ["新", "尝试", "实验", "突破", "创意", "想法"],
            "社交": ["大家", "一起", "合作", "分享", "交流", "帮忙"],
            "独立": ["自己", "一个人", "独立", "自主", "不需要"],
            "夜猫子": ["凌晨", "深夜", "0:", "1:", "2:", "3:"],
        }

        trait_scores = {}
        total_msgs = len(messages)

        for trait, keywords in trait_keywords.items():
            count = 0
            for msg in messages:
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                if any(kw in content for kw in keywords):
                    count += 1
            trait_scores[trait] = round(min(1.0, count / max(total_msgs, 1) * 3), 3)

        return trait_scores

    def _extract_topics(self, messages: List[Dict]) -> Dict[str, float]:
        """提取话题分布。"""
        topic_keywords = {
            "AI": ["AI", "GPT", "LLM", "模型", "训练", "Agent"],
            "音乐": ["音乐", "歌曲", "版权", "旋律", "作曲"],
            "教育": ["教学", "课程", "学生", "培训", "教育"],
            "商业": ["商业", "赚钱", "收入", "客户", "市场"],
            "技术": ["代码", "编程", "Python", "开发", "部署"],
            "生活": ["吃饭", "睡觉", "周末", "电影", "游戏"],
        }

        topic_counts = Counter()
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            for topic, keywords in topic_keywords.items():
                if any(kw in content for kw in keywords):
                    topic_counts[topic] += 1

        total = sum(topic_counts.values()) or 1
        return {topic: round(count / total, 3) for topic, count in topic_counts.most_common()}

    def _compute_sentiment(self, messages: List[Dict]) -> float:
        """简单情感倾向计算。"""
        positive = ["好", "棒", "赞", "喜欢", "开心", "感谢", "不错", "厉害"]
        negative = ["差", "烂", "烦", "讨厌", "失望", "不行", "问题"]

        pos_count = 0
        neg_count = 0
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            pos_count += sum(1 for kw in positive if kw in content)
            neg_count += sum(1 for kw in negative if kw in content)

        total = pos_count + neg_count or 1
        return round((pos_count - neg_count) / total, 3)

    def _extract_activity(self, messages: List[Dict]) -> Dict[str, int]:
        """提取活跃时段分布。"""
        hour_counts = Counter()
        for msg in messages:
            ts = msg.get("timestamp", "") if isinstance(msg, dict) else ""
            if ts:
                try:
                    hour = datetime.fromisoformat(ts).hour
                    hour_counts[str(hour)] += 1
                except (ValueError, TypeError):
                    pass
        return dict(hour_counts)

    def compute_evolution(self, from_idx: int = -2, to_idx: int = -1) -> Optional[EvolutionDelta]:
        """计算两个人格快照之间的变化。"""
        if len(self.snapshots) < 2:
            return None

        s1 = self.snapshots[from_idx]
        s2 = self.snapshots[to_idx]

        # Trait changes
        all_traits = set(s1.traits.keys()) | set(s2.traits.keys())
        trait_changes = {}
        new_traits = []
        lost_traits = []

        for trait in all_traits:
            v1 = s1.traits.get(trait, 0)
            v2 = s2.traits.get(trait, 0)
            delta = v2 - v1
            if abs(delta) > 0.05:
                trait_changes[trait] = round(delta, 3)
            if v1 == 0 and v2 > 0:
                new_traits.append(trait)
            if v1 > 0 and v2 == 0:
                lost_traits.append(trait)

        # Topic shifts
        all_topics = set(s1.topics.keys()) | set(s2.topics.keys())
        topic_shifts = {}
        for topic in all_topics:
            delta = s2.topics.get(topic, 0) - s1.topics.get(topic, 0)
            if abs(delta) > 0.02:
                topic_shifts[topic] = round(delta, 3)

        # Stability score
        if trait_changes:
            avg_change = sum(abs(v) for v in trait_changes.values()) / len(trait_changes)
            stability = max(0, 1 - avg_change * 2)
        else:
            stability = 1.0

        return EvolutionDelta(
            from_period=s1.source_period,
            to_period=s2.source_period,
            trait_changes=trait_changes,
            new_traits=new_traits,
            lost_traits=lost_traits,
            topic_shifts=topic_shifts,
            sentiment_delta=round(s2.sentiment - s1.sentiment, 3),
            stability_score=round(stability, 3),
        )

    def get_trend_report(self) -> str:
        """生成进化趋势报告。"""
        if not self.snapshots:
            return "无快照数据"

        lines = ["=== 人格进化趋势报告 ===\n"]
        lines.append(f"快照数: {len(self.snapshots)}")
        lines.append(f"时间跨度: {self.snapshots[0].source_period} → {self.snapshots[-1].source_period}\n")

        # Latest snapshot
        latest = self.snapshots[-1]
        lines.append(f"最新快照 ({latest.source_period}):")
        lines.append(f"  消息数: {latest.message_count}")
        lines.append(f"  情感倾向: {latest.sentiment:+.2f}")
        if latest.traits:
            lines.append("  核心特征:")
            for trait, score in sorted(latest.traits.items(), key=lambda x: x[1], reverse=True)[:5]:
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                lines.append(f"    {trait}: {bar} {score:.1%}")

        # Evolution
        if len(self.snapshots) >= 2:
            delta = self.compute_evolution()
            if delta:
                lines.append(f"\n{delta.summary()}")

        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    tracker = PersonaEvolutionTracker(sys.argv[1] if len(sys.argv) > 1 else "evolution")

    # Demo: create a snapshot from a sample file
    if len(sys.argv) > 2:
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            messages = json.load(f)
        tracker.create_snapshot_from_messages(messages, period_label="demo")
        print(tracker.get_trend_report())
    else:
        print(tracker.get_trend_report())
