"""
Awake Differently — Conversation Pattern Analyzer
===================================================
Extracts behavioral patterns from chat history for digital twin enhancement.

Features:
- Message frequency analysis (peak hours, response patterns)
- Vocabulary fingerprint (signature phrases, sentence patterns)
- Relationship mapping (who talks to whom, how often)
- Emotional tone distribution
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class MessagePattern:
    """Extracted behavioral pattern from conversation."""
    pattern_type: str  # vocabulary, frequency, relationship, tone
    description: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class PersonaProfile:
    """Digital twin persona profile extracted from conversations."""
    name: str
    total_messages: int = 0
    avg_message_length: float = 0
    peak_hours: list[int] = field(default_factory=list)
    signature_phrases: list[str] = field(default_factory=list)
    response_style: str = ""  # short/medium/long
    emotional_tone: dict = field(default_factory=dict)  # {"positive": 0.6, "negative": 0.1, ...}
    relationships: dict = field(default_factory=dict)  # {"name": frequency}
    patterns: list[MessagePattern] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== {self.name} 人格画像 ===",
            f"消息总数: {self.total_messages}",
            f"平均消息长度: {self.avg_message_length:.0f}字",
            f"活跃时段: {', '.join(f'{h}:00' for h in self.peak_hours[:3])}",
            f"口头禅: {', '.join(self.signature_phrases[:5])}",
            f"回复风格: {self.response_style}",
        ]
        if self.relationships:
            top_contacts = sorted(self.relationships.items(), key=lambda x: -x[1])[:5]
            lines.append(f"高频互动: {', '.join(f'{n}({c}次)' for n, c in top_contacts)}")
        return "\n".join(lines)


def analyze_messages(messages: list[dict], person_name: str = "") -> PersonaProfile:
    """Analyze a list of messages to extract behavioral patterns.
    
    Args:
        messages: [{"sender": str, "text": str, "timestamp": str}, ...]
        person_name: Name of the person to profile
    """
    profile = PersonaProfile(name=person_name)
    
    # Filter messages from target person
    target_msgs = [m for m in messages if m.get("sender") == person_name] if person_name else messages
    if not target_msgs:
        target_msgs = messages

    profile.total_messages = len(target_msgs)
    
    if not target_msgs:
        return profile

    # Average message length
    lengths = [len(m.get("text", "")) for m in target_msgs]
    profile.avg_message_length = sum(lengths) / len(lengths) if lengths else 0

    # Response style
    if profile.avg_message_length < 20:
        profile.response_style = "极简(短句为主)"
    elif profile.avg_message_length < 50:
        profile.response_style = "简洁(中等长度)"
    else:
        profile.response_style = "详细(长回复)"

    # Peak hours
    hours = Counter()
    for m in target_msgs:
        ts = m.get("timestamp", "")
        if "T" in ts:
            try:
                h = int(ts.split("T")[1][:2])
                hours[h] += 1
            except (ValueError, IndexError):
                pass
    profile.peak_hours = [h for h, _ in hours.most_common(3)]

    # Signature phrases (2-4 char frequent phrases)
    all_text = " ".join(m.get("text", "") for m in target_msgs)
    phrases = Counter()
    for i in range(len(all_text) - 1):
        bigram = all_text[i:i+2]
        if re.match(r"[一-鿿]{2}", bigram):
            phrases[bigram] += 1
    profile.signature_phrases = [p for p, c in phrases.most_common(10) if c > 3]

    # Relationship mapping
    contacts = Counter()
    for m in messages:
        if m.get("sender") != person_name:
            contacts[m.get("sender", "unknown")] += 1
    profile.relationships = dict(contacts.most_common(20))

    return profile


if __name__ == "__main__":
    test_msgs = [
        {"sender": "陈龙", "text": "好的收到", "timestamp": "2026-05-01T10:30:00"},
        {"sender": "陈龙", "text": "这个方案可以", "timestamp": "2026-05-01T11:00:00"},
        {"sender": "PR", "text": "你觉得怎么样", "timestamp": "2026-05-01T11:05:00"},
        {"sender": "陈龙", "text": "直接说结果吧", "timestamp": "2026-05-01T14:00:00"},
    ]
    profile = analyze_messages(test_msgs, "陈龙")
    print(profile.summary())
