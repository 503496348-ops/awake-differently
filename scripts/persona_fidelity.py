"""
Awake Differently — Persona Fidelity Evaluator
================================================
Evaluates how accurately a distilled persona reproduces the original person's
behavior, style, and decision-making patterns.

Scoring dimensions:
1. Style Fidelity — Does the persona's output match the original's writing style?
2. Behavioral Fidelity — Do temporal/social patterns align?
3. Decision Fidelity — Does the persona make decisions like the original?
4. Vocabulary Fidelity — Does it use the same signature phrases?
5. Emotional Fidelity — Does it match the emotional tone distribution?

Overall score = weighted average of all dimensions (0-100).

Usage:
    evaluator = PersonaFidelityEvaluator()
    score = evaluator.evaluate(original_messages, persona_config, test_outputs)

Author: AtomCollide-智械工坊
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("awake_differently.fidelity_eval")


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """Score for a single fidelity dimension."""
    dimension: str
    score: float  # 0.0 - 1.0
    weight: float
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class FidelityReport:
    """Complete fidelity evaluation report."""
    dimensions: List[DimensionScore]
    overall_score: float  # 0-100
    grade: str  # A/B/C/D/F
    summary: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "grade": self.grade,
            "summary": self.summary,
            "dimensions": {
                d.dimension: {
                    "score": round(d.score * 100, 1),
                    "weight": d.weight,
                    "weighted": round(d.weighted_score * 100, 1),
                    "details": d.details,
                    "evidence": d.evidence[:3],
                }
                for d in self.dimensions
            },
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
        }

    def summary_text(self) -> str:
        lines = [
            f"=== 人格保真度评估报告 ===",
            f"综合得分: {self.overall_score:.1f}/100 (等级: {self.grade})",
            f"",
            f"各维度得分:",
        ]
        for d in self.dimensions:
            bar = "█" * int(d.score * 20) + "░" * (20 - int(d.score * 20))
            lines.append(f"  {d.dimension}: {bar} {d.score * 100:.1f}% (权重{d.weight:.0%})")

        if self.strengths:
            lines.append(f"\n✅ 优势: {', '.join(self.strengths)}")
        if self.weaknesses:
            lines.append(f"⚠ 薄弱: {', '.join(self.weaknesses)}")
        if self.recommendations:
            lines.append(f"\n💡 改进建议:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


# ─── Text Similarity Utilities ────────────────────────────────────────────────

class TextSimilarity:
    """Text similarity computation utilities."""

    @staticmethod
    def jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Cosine similarity between two frequency vectors."""
        all_keys = set(vec_a.keys()) | set(vec_b.keys())
        if not all_keys:
            return 1.0

        dot_product = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_keys)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    @staticmethod
    def char_ngram_similarity(text_a: str, text_b: str, n: int = 2) -> float:
        """Character n-gram similarity (for short text comparison)."""
        if not text_a or not text_b:
            return 0.0

        def get_ngrams(text: str, n: int) -> Counter:
            return Counter(text[i:i + n] for i in range(len(text) - n + 1))

        ngrams_a = get_ngrams(text_a, n)
        ngrams_b = get_ngrams(text_b, n)
        return TextSimilarity.cosine_similarity(
            dict(ngrams_a), dict(ngrams_b)
        )

    @staticmethod
    def distribution_divergence(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
        """Jensen-Shannon divergence between two distributions. Returns 0-1 (0=identical)."""
        all_keys = set(dist_a.keys()) | set(dist_b.keys())
        if not all_keys:
            return 0.0

        # Normalize distributions
        total_a = sum(dist_a.values()) or 1
        total_b = sum(dist_b.values()) or 1

        p = {k: dist_a.get(k, 0) / total_a for k in all_keys}
        q = {k: dist_b.get(k, 0) / total_b for k in all_keys}

        # M = (P + Q) / 2
        m = {k: (p[k] + q[k]) / 2 for k in all_keys}

        # KL divergence
        def kl_div(a: dict, b: dict) -> float:
            return sum(
                a[k] * math.log(a[k] / b[k])
                for k in all_keys if a[k] > 0 and b[k] > 0
            )

        jsd = (kl_div(p, m) + kl_div(q, m)) / 2
        # Normalize to 0-1
        return min(jsd / math.log(2), 1.0) if jsd > 0 else 0.0


# ─── Style Fidelity Evaluator ────────────────────────────────────────────────

class StyleFidelityEvaluator:
    """Evaluate writing style fidelity between original and persona output."""

    # Dimension weights
    WEIGHT = 0.20

    def evaluate(self, original_messages: List[dict], persona_outputs: List[str]) -> DimensionScore:
        """Compare writing style metrics between original and persona."""
        evidence = []
        details = {}

        # Extract style metrics from original
        orig_texts = [m.get("text", "") for m in original_messages if m.get("text")]
        persona_text = " ".join(persona_outputs)

        if not orig_texts or not persona_text:
            return DimensionScore(
                dimension="风格保真度",
                score=0.0,
                weight=self.WEIGHT,
                details={"error": "数据不足"},
                evidence=["原始消息或人格输出为空"],
            )

        scores = []

        # 1. Message length distribution
        orig_lengths = [len(t) for t in orig_texts]
        persona_lengths = [len(s) for s in persona_outputs if s]

        orig_avg_len = sum(orig_lengths) / max(len(orig_lengths), 1)
        persona_avg_len = sum(persona_lengths) / max(len(persona_lengths), 1)

        length_ratio = min(orig_avg_len, persona_avg_len) / max(orig_avg_len, persona_avg_len) if max(orig_avg_len, persona_avg_len) > 0 else 1.0
        scores.append(length_ratio)
        details["avg_message_length"] = {
            "original": round(orig_avg_len, 1),
            "persona": round(persona_avg_len, 1),
            "match": round(length_ratio * 100, 1),
        }
        evidence.append(f"消息长度: 原始{orig_avg_len:.0f}字 vs 人格{persona_avg_len:.0f}字")

        # 2. Punctuation patterns
        orig_excl = sum(t.count("！") + t.count("!") for t in orig_texts) / max(len(orig_texts), 1)
        persona_excl = persona_text.count("！") + persona_text.count("!")
        persona_excl_rate = persona_excl / max(len(persona_outputs), 1)

        punct_similarity = 1.0 - min(abs(orig_excl - persona_excl_rate) / max(orig_excl, persona_excl_rate, 1), 1.0)
        scores.append(punct_similarity)
        details["exclamation_rate"] = {
            "original": round(orig_excl, 2),
            "persona": round(persona_excl_rate, 2),
        }

        # 3. Sentence length distribution
        orig_sentences = []
        for t in orig_texts:
            orig_sentences.extend(re.split(r'[。！？\n]', t))
        orig_sentences = [s.strip() for s in orig_sentences if s.strip() and len(s.strip()) > 1]

        persona_sentences = re.split(r'[。！？\n]', persona_text)
        persona_sentences = [s.strip() for s in persona_sentences if s.strip() and len(s.strip()) > 1]

        orig_sent_avg = sum(len(s) for s in orig_sentences) / max(len(orig_sentences), 1)
        persona_sent_avg = sum(len(s) for s in persona_sentences) / max(len(persona_sentences), 1)

        sent_ratio = min(orig_sent_avg, persona_sent_avg) / max(orig_sent_avg, persona_sent_avg) if max(orig_sent_avg, persona_sent_avg) > 0 else 1.0
        scores.append(sent_ratio)
        details["sentence_length"] = {
            "original": round(orig_sent_avg, 1),
            "persona": round(persona_sent_avg, 1),
        }

        # 4. Short sentence ratio (messages < 20 chars)
        orig_short_ratio = sum(1 for t in orig_texts if len(t) < 20) / max(len(orig_texts), 1)
        persona_short_ratio = sum(1 for s in persona_outputs if len(s) < 20) / max(len(persona_outputs), 1)

        short_sim = 1.0 - abs(orig_short_ratio - persona_short_ratio)
        scores.append(short_sim)
        details["short_sentence_ratio"] = {
            "original": round(orig_short_ratio * 100, 1),
            "persona": round(persona_short_ratio * 100, 1),
        }

        overall = sum(scores) / max(len(scores), 1)

        return DimensionScore(
            dimension="风格保真度",
            score=min(overall, 1.0),
            weight=self.WEIGHT,
            details=details,
            evidence=evidence,
        )


# ─── Vocabulary Fidelity Evaluator ───────────────────────────────────────────

class VocabularyFidelityEvaluator:
    """Evaluate vocabulary and phrase fidelity."""

    WEIGHT = 0.15

    def evaluate(self, original_messages: List[dict], persona_outputs: List[str], persona_config: dict = None) -> DimensionScore:
        """Compare vocabulary fingerprint between original and persona."""
        evidence = []
        details = {}

        orig_texts = [m.get("text", "") for m in original_messages if m.get("text")]
        persona_text = " ".join(persona_outputs)

        if not orig_texts or not persona_text:
            return DimensionScore(
                dimension="词汇保真度",
                score=0.0, weight=self.WEIGHT,
                details={"error": "数据不足"},
            )

        scores = []

        # 1. Signature phrase coverage
        # Extract bigrams from original
        orig_all = " ".join(orig_texts)
        orig_bigrams = Counter()
        for i in range(len(orig_all) - 1):
            bg = orig_all[i:i + 2]
            if re.match(r'[\u4e00-\u9fff]{2}', bg):
                orig_bigrams[bg] += 1

        top_phrases = [p for p, c in orig_bigrams.most_common(20) if c > 3]

        # Check if persona uses these phrases
        persona_phrase_hits = sum(1 for p in top_phrases if p in persona_text)
        phrase_coverage = persona_phrase_hits / max(len(top_phrases), 1)
        scores.append(phrase_coverage)

        details["signature_phrases"] = {
            "top_phrases": top_phrases[:10],
            "persona_hits": persona_phrase_hits,
            "coverage": round(phrase_coverage * 100, 1),
        }
        evidence.append(f"特征词汇覆盖: {persona_phrase_hits}/{len(top_phrases)}")

        # 2. Character bigram cosine similarity
        persona_bigrams = Counter()
        for i in range(len(persona_text) - 1):
            bg = persona_text[i:i + 2]
            if re.match(r'[\u4e00-\u9fff]{2}', bg):
                persona_bigrams[bg] += 1

        bigram_sim = TextSimilarity.cosine_similarity(dict(orig_bigrams), dict(persona_bigrams))
        scores.append(bigram_sim)
        details["bigram_similarity"] = round(bigram_sim * 100, 1)

        # 3. Catchphrase usage from persona config
        if persona_config:
            catchphrases = persona_config.get("catchphrases", [])
            if not catchphrases:
                # Extract from persona.md style definition
                catchphrases = persona_config.get("signature_phrases", [])

            if catchphrases:
                used = sum(1 for cp in catchphrases if cp in persona_text)
                catchphrase_rate = used / max(len(catchphrases), 1)
                scores.append(catchphrase_rate)
                details["catchphrase_usage"] = {
                    "defined": catchphrases,
                    "used": used,
                    "rate": round(catchphrase_rate * 100, 1),
                }
                evidence.append(f"口头禅使用: {used}/{len(catchphrases)}")

        # 4. Vocabulary richness comparison
        orig_vocab = set(orig_all)
        persona_vocab = set(persona_text)
        vocab_overlap = len(orig_vocab & persona_vocab) / max(len(orig_vocab), 1)
        scores.append(min(vocab_overlap, 1.0))
        details["vocabulary_overlap"] = round(vocab_overlap * 100, 1)

        overall = sum(scores) / max(len(scores), 1)

        return DimensionScore(
            dimension="词汇保真度",
            score=min(overall, 1.0),
            weight=self.WEIGHT,
            details=details,
            evidence=evidence,
        )


# ─── Behavioral Fidelity Evaluator ───────────────────────────────────────────

class BehavioralFidelityEvaluator:
    """Evaluate behavioral pattern fidelity."""

    WEIGHT = 0.15

    def evaluate(self, original_messages: List[dict], behavior_patterns: dict = None) -> DimensionScore:
        """Evaluate if extracted behavior patterns match original data."""
        evidence = []
        details = {}
        scores = []

        if not original_messages:
            return DimensionScore(
                dimension="行为保真度", score=0.0, weight=self.WEIGHT,
                details={"error": "无原始消息"},
            )

        # 1. Temporal pattern consistency
        hours = Counter()
        for msg in original_messages:
            ts = msg.get("timestamp", "")
            if "T" in ts:
                try:
                    h = int(ts.split("T")[1][:2])
                    hours[h] += 1
                except (ValueError, IndexError):
                    pass

        if hours:
            total = sum(hours.values())
            night_ratio = sum(hours.get(h, 0) for h in range(0, 6)) / total
            peak_hours = [h for h, _ in hours.most_common(3)]

            # If behavior patterns exist, check alignment
            if behavior_patterns:
                patterns = behavior_patterns.get("patterns", [])
                temporal = next((p for p in patterns if p.get("type") == "temporal"), None)
                if temporal:
                    claimed_night = temporal.get("night_owl", False)
                    actual_night = night_ratio > 0.3
                    temporal_match = claimed_night == actual_night
                    scores.append(1.0 if temporal_match else 0.3)
                    details["temporal"] = {
                        "night_ratio": round(night_ratio * 100, 1),
                        "peak_hours": peak_hours,
                        "pattern_aligned": temporal_match,
                    }
                    evidence.append(f"深夜活跃: {night_ratio:.0%}, 峰值时段: {peak_hours}")

        # 2. Interaction breadth consistency
        contacts = Counter()
        for msg in original_messages:
            sender = msg.get("sender", "")
            if sender:
                contacts[sender] += 1

        unique_contacts = len(contacts)
        if behavior_patterns:
            patterns = behavior_patterns.get("patterns", [])
            social = next((p for p in patterns if p.get("type") == "social"), None)
            if social:
                claimed_breadth = social.get("interaction_breadth", 0)
                breadth_ratio = min(unique_contacts, claimed_breadth) / max(unique_contacts, claimed_breadth, 1)
                scores.append(breadth_ratio)
                details["interaction_breadth"] = {
                    "actual": unique_contacts,
                    "claimed": claimed_breadth,
                }

        # 3. Message type distribution
        msg_types = Counter()
        for msg in original_messages:
            msg_types[msg.get("msg_type", "text")] += 1
        total_msgs = sum(msg_types.values())
        type_dist = {k: v / total_msgs for k, v in msg_types.items()}
        details["message_type_distribution"] = {k: round(v * 100, 1) for k, v in type_dist.items()}

        # Default score if no behavior patterns to compare
        if not scores:
            scores = [0.8]  # Neutral score when no patterns to compare
            evidence.append("行为模式数据不足以进行对比评估")

        overall = sum(scores) / max(len(scores), 1)

        return DimensionScore(
            dimension="行为保真度",
            score=min(overall, 1.0),
            weight=self.WEIGHT,
            details=details,
            evidence=evidence,
        )


# ─── Decision Fidelity Evaluator ─────────────────────────────────────────────

class DecisionFidelityEvaluator:
    """Evaluate decision-making pattern fidelity."""

    WEIGHT = 0.20

    # Decision-related keywords
    ACTION_WORDS = {"搞", "干", "做", "来", "发", "推", "卖", "买", "谈", "定", "搞", "搞起来", "直接"}
    RISK_WORDS = {"风险", "试试", "先试试", "看看", "不一定", "可能", "也许"}
    PRAGMATIC_WORDS = {"变现", "赚钱", "收入", "利润", "成本", "投入", "产出", "ROI", "值不值"}
    HELPER_WORDS = {"帮", "示例称呼", "大家", "团队", "一起", "分享", "福利"}

    WEIGHT = 0.20

    def evaluate(self, original_messages: List[dict], persona_outputs: List[str] = None, persona_config: dict = None) -> DimensionScore:
        """Evaluate decision-making pattern fidelity."""
        evidence = []
        details = {}
        scores = []

        orig_texts = [m.get("text", "") for m in original_messages if m.get("text")]
        if not orig_texts:
            return DimensionScore(
                dimension="决策保真度", score=0.0, weight=self.WEIGHT,
                details={"error": "无原始消息"},
            )

        all_text = " ".join(orig_texts)
        total_msgs = len(orig_texts)

        # 1. Action orientation
        action_count = sum(1 for t in orig_texts if any(w in t for w in self.ACTION_WORDS))
        question_words = {"怎么", "什么", "为什么", "如何", "哪", "吗"}
        question_count = sum(1 for t in orig_texts if any(w in t for w in question_words))

        action_ratio = action_count / max(total_msgs, 1)
        question_ratio = question_count / max(total_msgs, 1)

        if action_ratio > question_ratio * 2:
            decision_style = "行动导向"
        elif question_ratio > action_ratio:
            decision_style = "探究导向"
        else:
            decision_style = "平衡型"

        details["decision_style"] = {
            "type": decision_style,
            "action_ratio": round(action_ratio * 100, 1),
            "question_ratio": round(question_ratio * 100, 1),
        }
        evidence.append(f"决策风格: {decision_style} (行动{action_ratio:.0%} vs 提问{question_ratio:.0%})")

        # 2. Pragmatism score (business/practical orientation)
        pragmatic_count = sum(1 for t in orig_texts if any(w in t for w in self.PRAGMATIC_WORDS))
        pragmatic_ratio = pragmatic_count / max(total_msgs, 1)
        details["pragmatism"] = round(pragmatic_ratio * 100, 1)

        # 3. Helper orientation
        helper_count = sum(1 for t in orig_texts if any(w in t for w in self.HELPER_WORDS))
        helper_ratio = helper_count / max(total_msgs, 1)
        details["helper_orientation"] = round(helper_ratio * 100, 1)

        # Compare with persona config if available
        if persona_config:
            priority = persona_config.get("decision_priority", [])
            if priority:
                # Check if top priority aligns with detected style
                top_priority = priority[0] if priority else ""
                if "变现" in top_priority and pragmatic_ratio > 0.05:
                    scores.append(0.9)
                    evidence.append("变现优先级与实际行为一致")
                elif "影响力" in top_priority and helper_ratio > 0.05:
                    scores.append(0.85)
                else:
                    scores.append(0.6)

        # Default scoring: consistency of decision patterns
        if not scores:
            # Higher score if patterns are clear (not ambiguous)
            clarity = abs(action_ratio - question_ratio)
            scores.append(min(0.5 + clarity * 2, 1.0))

        overall = sum(scores) / max(len(scores), 1)

        return DimensionScore(
            dimension="决策保真度",
            score=min(overall, 1.0),
            weight=self.WEIGHT,
            details=details,
            evidence=evidence,
        )


# ─── Emotional Fidelity Evaluator ────────────────────────────────────────────

class EmotionalFidelityEvaluator:
    """Evaluate emotional tone fidelity."""

    WEIGHT = 0.15

    # Emotion markers
    ENTHUSIASTIC_MARKERS = {"！", "！!", "哈哈", "太好了", "牛", "棒", "赞", "厉害", "示例称呼"}
    CRITICAL_MARKERS = {"示例口头禅", "别扯", "扯淡", "狗屎", "不行", "不对", "错了"}
    WARM_MARKERS = {"兄弟", "加油", "支持", "谢谢", "辛苦", "好的", "收到"}

    def evaluate(self, original_messages: List[dict], persona_outputs: List[str] = None) -> DimensionScore:
        """Evaluate emotional tone fidelity."""
        evidence = []
        details = {}

        orig_texts = [m.get("text", "") for m in original_messages if m.get("text")]
        if not orig_texts:
            return DimensionScore(
                dimension="情感保真度", score=0.0, weight=self.WEIGHT,
                details={"error": "无原始消息"},
            )

        total = len(orig_texts)

        # Compute emotion distribution for original
        enthusiastic = sum(1 for t in orig_texts if any(m in t for m in self.ENTHUSIASTIC_MARKERS)) / total
        critical = sum(1 for t in orig_texts if any(m in t for m in self.CRITICAL_MARKERS)) / total
        warm = sum(1 for t in orig_texts if any(m in t for m in self.WARM_MARKERS)) / total
        neutral = 1.0 - enthusiastic - critical - warm

        orig_dist = {
            "enthusiastic": enthusiastic,
            "critical": critical,
            "warm": warm,
            "neutral": max(neutral, 0),
        }

        details["original_emotion_distribution"] = {k: round(v * 100, 1) for k, v in orig_dist.items()}
        evidence.append(f"原始情感: 热情{enthusiastic:.0%} 批判{critical:.0%} 温暖{warm:.0%}")

        # If persona outputs available, compare
        if persona_outputs:
            persona_text = " ".join(persona_outputs)
            p_total = max(len(persona_outputs), 1)

            p_enthusiastic = sum(1 for t in persona_outputs if any(m in t for m in self.ENTHUSIASTIC_MARKERS)) / p_total
            p_critical = sum(1 for t in persona_outputs if any(m in t for m in self.CRITICAL_MARKERS)) / p_total
            p_warm = sum(1 for t in persona_outputs if any(m in t for m in self.WARM_MARKERS)) / p_total
            p_neutral = 1.0 - p_enthusiastic - p_critical - p_warm

            persona_dist = {
                "enthusiastic": p_enthusiastic,
                "critical": p_critical,
                "warm": p_warm,
                "neutral": max(p_neutral, 0),
            }

            details["persona_emotion_distribution"] = {k: round(v * 100, 1) for k, v in persona_dist.items()}

            # Compare distributions
            divergence = TextSimilarity.distribution_divergence(orig_dist, persona_dist)
            emotion_score = 1.0 - divergence
            evidence.append(f"情感分布偏差: {divergence:.2f}")

            return DimensionScore(
                dimension="情感保真度",
                score=min(emotion_score, 1.0),
                weight=self.WEIGHT,
                details=details,
                evidence=evidence,
            )

        # No persona outputs — score based on data clarity
        entropy = -sum(v * math.log2(v + 1e-10) for v in orig_dist.values() if v > 0)
        max_entropy = math.log2(len(orig_dist))
        clarity = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)

        return DimensionScore(
            dimension="情感保真度",
            score=clarity,
            weight=self.WEIGHT,
            details=details,
            evidence=evidence,
        )



# ─── Robustness Fidelity Evaluator ──────────────────────────────────────────

class RobustnessFidelityEvaluator:
    """Evaluate persona robustness under adversarial/noisy conditions.

    Tests:
    1. Noise Resilience — catchphrase length and specificity
    2. Data Diversity — unique message ratio
    3. Temporal Coverage — days spanned
    4. Length Variance — message pattern diversity
    """

    WEIGHT = 0.15

    def evaluate(self, original_messages: List[dict], persona_outputs: List[str] = None, persona_config: dict = None) -> DimensionScore:
        """Evaluate robustness of persona definition."""
        evidence = []
        details = {}
        scores = []

        if not original_messages:
            return DimensionScore(
                dimension="鲁棒性", score=0.0, weight=self.WEIGHT,
                details={"error": "无原始消息"},
            )

        orig_texts = [m.get("text", "") for m in original_messages if m.get("text")]
        if not orig_texts:
            return DimensionScore(
                dimension="鲁棒性", score=0.0, weight=self.WEIGHT,
                details={"error": "无文本内容"},
            )

        # 1. Noise resilience — robust features vs fragile ones
        if persona_config:
            catchphrases = persona_config.get("catchphrases", [])
            signature_phrases = persona_config.get("signature_phrases", [])

            if catchphrases:
                avg_len = sum(len(p) for p in catchphrases) / len(catchphrases)
                robustness = min(avg_len / 4.0, 1.0)
                scores.append(robustness)
                details["catchphrase_robustness"] = {
                    "avg_length": round(avg_len, 1),
                    "score": round(robustness * 100, 1),
                    "examples": catchphrases[:5],
                }
                evidence.append(f"口头禅平均长度{avg_len:.1f}字，抗噪性{'好' if robustness > 0.7 else '一般'}")

            if signature_phrases:
                common_words = {"的", "是", "不", "在", "有", "我", "你", "他", "她", "这", "那", "了", "吗"}
                fragile = [p for p in signature_phrases if p in common_words]
                fragility = len(fragile) / len(signature_phrases)
                robustness = 1.0 - fragility
                scores.append(robustness)
                details["signature_phrase_robustness"] = {
                    "total": len(signature_phrases),
                    "fragile": len(fragile),
                    "fragile_examples": fragile[:5],
                    "score": round(robustness * 100, 1),
                }
                evidence.append(f"特征词{len(signature_phrases)}个，{len(fragile)}个为高频词(易误判)")

        # 2. Data diversity resilience
        unique_msgs = len(set(orig_texts))
        diversity = unique_msgs / max(len(orig_texts), 1)
        scores.append(diversity)
        details["data_diversity"] = {
            "total": len(orig_texts),
            "unique": unique_msgs,
            "ratio": round(diversity * 100, 1),
        }
        evidence.append(f"消息多样性: {unique_msgs}/{len(orig_texts)} ({diversity:.0%})")

        # 3. Temporal coverage resilience
        dates = set()
        for msg in original_messages:
            ts = msg.get("timestamp", "")
            if ts:
                try:
                    dates.add(ts[:10])
                except (IndexError, ValueError):
                    pass
        temporal_score = min(len(dates) / 14.0, 1.0)
        scores.append(temporal_score)
        details["temporal_coverage"] = {
            "days": len(dates),
            "score": round(temporal_score * 100, 1),
        }
        evidence.append(f"时间跨度: {len(dates)}天")

        # 4. Message length variance
        lengths = [len(t) for t in orig_texts]
        if len(lengths) > 1:
            avg = sum(lengths) / len(lengths)
            variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
            std_dev = math.sqrt(variance)
            cv = std_dev / avg if avg > 0 else 0
            variance_score = min(cv / 1.5, 1.0)
            scores.append(variance_score)
            details["length_variance"] = {
                "mean": round(avg, 1),
                "std_dev": round(std_dev, 1),
                "cv": round(cv, 2),
                "score": round(variance_score * 100, 1),
            }
            evidence.append(f"消息长度变异系数: {cv:.2f}")

        if not scores:
            scores = [0.5]
            evidence.append("数据不足，无法全面评估鲁棒性")

        overall = sum(scores) / max(len(scores), 1)

        return DimensionScore(
            dimension="鲁棒性",
            score=min(overall, 1.0),
            weight=self.WEIGHT,
            details=details,
            evidence=evidence,
        )


# ─── Main Evaluator ──────────────────────────────────────────────────────────

class PersonaFidelityEvaluator:
    """Complete persona fidelity evaluation.

    Usage:
        evaluator = PersonaFidelityEvaluator()
        report = evaluator.evaluate(
            original_messages=[{"sender": "用户A", "text": "...", "timestamp": "..."}],
            persona_config={"catchphrases": ["示例口头禅1", "示例口头禅2"], "decision_priority": ["变现"]},
            persona_outputs=["这个可以搞，具体方案是..."],
        )
        print(report.summary_text())
    """

    def __init__(self):
        self.style_eval = StyleFidelityEvaluator()
        self.vocab_eval = VocabularyFidelityEvaluator()
        self.behavior_eval = BehavioralFidelityEvaluator()
        self.decision_eval = DecisionFidelityEvaluator()
        self.emotional_eval = EmotionalFidelityEvaluator()
        self.robustness_eval = RobustnessFidelityEvaluator()

    def evaluate(
        self,
        original_messages: List[dict],
        persona_config: Optional[dict] = None,
        persona_outputs: Optional[List[str]] = None,
        behavior_patterns: Optional[dict] = None,
    ) -> FidelityReport:
        """Run full fidelity evaluation.

        Args:
            original_messages: Original chat messages (pipeline format)
            persona_config: Persona definition (catchphrases, decision_priority, etc.)
            persona_outputs: Sample outputs from the persona (for style/vocab comparison)
            behavior_patterns: Extracted behavior patterns (from workflow engine)

        Returns:
            FidelityReport with scores, evidence, and recommendations
        """
        dimensions = []

        # 1. Style fidelity
        if persona_outputs:
            dimensions.append(self.style_eval.evaluate(original_messages, persona_outputs))
        else:
            dimensions.append(DimensionScore(
                dimension="风格保真度", score=0.5, weight=0.25,
                details={"status": "未提供人格输出样本，无法评估"},
                evidence=["提供persona输出样本以启用此维度评估"],
            ))

        # 2. Vocabulary fidelity
        dimensions.append(self.vocab_eval.evaluate(original_messages, persona_outputs, persona_config))

        # 3. Behavioral fidelity
        dimensions.append(self.behavior_eval.evaluate(original_messages, behavior_patterns))

        # 4. Decision fidelity
        dimensions.append(self.decision_eval.evaluate(original_messages, persona_outputs, persona_config))

        # 5. Emotional fidelity
        dimensions.append(self.emotional_eval.evaluate(original_messages, persona_outputs))

        # 6. Robustness fidelity
        dimensions.append(self.robustness_eval.evaluate(original_messages, persona_outputs, persona_config))

        # Compute overall score
        total_weight = sum(d.weight for d in dimensions)
        overall = sum(d.weighted_score for d in dimensions) / total_weight * 100 if total_weight > 0 else 0

        # Grade
        if overall >= 90:
            grade = "A"
        elif overall >= 80:
            grade = "B"
        elif overall >= 70:
            grade = "C"
        elif overall >= 60:
            grade = "D"
        else:
            grade = "F"

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        recommendations = []

        for d in dimensions:
            pct = d.score * 100
            if pct >= 80:
                strengths.append(f"{d.dimension} ({pct:.0f}%)")
            elif pct < 60:
                weaknesses.append(f"{d.dimension} ({pct:.0f}%)")

        # Generate recommendations
        if not persona_outputs:
            recommendations.append("提供人格输出样本（模拟对话/回复）以获得更准确的评估")

        data_count = len(original_messages)
        if data_count < 200:
            recommendations.append(f"当前消息量{data_count}条，建议增加到200+以提升评估准确性")

        # Time span check
        dates = set()
        for msg in original_messages:
            ts = msg.get("timestamp", "")
            if ts:
                try:
                    dates.add(ts[:10])
                except (IndexError, ValueError):
                    pass
        if len(dates) < 7:
            recommendations.append(f"数据仅覆盖{len(dates)}天，建议扩展到14天以上")

        for d in dimensions:
            if d.score < 0.6:
                if d.dimension == "风格保真度":
                    recommendations.append("增加更多原始消息样本，让模型更好地学习语言风格")
                elif d.dimension == "词汇保真度":
                    recommendations.append("确保人格配置中包含正确的口头禅和特征词汇")
                elif d.dimension == "行为保真度":
                    recommendations.append("补充更多时间段的数据以完善行为模式提取")
                elif d.dimension == "决策保真度":
                    recommendations.append("在persona中明确定义决策优先级框架")

        summary = f"综合得分{overall:.1f}/100 (等级{grade})，{len(strengths)}项优势，{len(weaknesses)}项薄弱"

        return FidelityReport(
            dimensions=dimensions,
            overall_score=overall,
            grade=grade,
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    # Demo with the Awake Differently example data
    sample_messages = [
        {"sender": "用户A", "text": "好的收到", "timestamp": "2026-05-01T10:30:00"},
        {"sender": "用户A", "text": "这个方案可以，直接干", "timestamp": "2026-05-01T11:00:00"},
        {"sender": "PR", "text": "你觉得怎么样", "timestamp": "2026-05-01T11:05:00"},
        {"sender": "用户A", "text": "别催，正在谈，等我消息", "timestamp": "2026-05-01T14:00:00"},
        {"sender": "用户A", "text": "示例社群消息", "timestamp": "2026-05-01T02:30:00"},
        {"sender": "用户A", "text": "示例口头禅内容", "timestamp": "2026-05-02T01:00:00"},
        {"sender": "用户A", "text": "示例对话内容", "timestamp": "2026-05-02T02:00:00"},
        {"sender": "用户A", "text": "示例表达方式", "timestamp": "2026-05-02T03:00:00"},
        {"sender": "李渔樵", "text": "龙哥这个怎么搞", "timestamp": "2026-05-02T10:00:00"},
        {"sender": "用户A", "text": "示例对话内容", "timestamp": "2026-05-02T10:05:00"},
        {"sender": "用户A", "text": "示例金句", "timestamp": "2026-05-02T02:00:00"},
        {"sender": "用户A", "text": "示例金句", "timestamp": "2026-05-03T01:00:00"},
        {"sender": "用户A", "text": "示例社群消息", "timestamp": "2026-05-03T03:00:00"},
        {"sender": "用户A", "text": "这个可以搞，具体方案是……", "timestamp": "2026-05-03T04:00:00"},
        {"sender": "小乖助理", "text": "音乐模型训练完了", "timestamp": "2026-05-03T01:00:00"},
        {"sender": "用户A", "text": "发来听听", "timestamp": "2026-05-03T01:02:00"},
    ]

    persona_config = {
        "catchphrases": ["示例口头禅1", "示例口头禅2", "示例称呼"],
        "decision_priority": ["变现", "影响力", "帮兄弟", "技术优雅"],
        "signature_phrases": ["福利", "版权", "变现", "脑力收入", "睡觉收入"],
    }

    persona_outputs = [
        "示例社群消息",
        "这个可以搞，具体方案是先对接平台方",
        "你这个死孩子，我让你通过脑力赚钱",
        "别扯了，先干",
        "示例对话内容",
    ]

    evaluator = PersonaFidelityEvaluator()
    report = evaluator.evaluate(
        original_messages=sample_messages,
        persona_config=persona_config,
        persona_outputs=persona_outputs,
    )

    print(report.summary_text())
    print()
    print("=== 详细数据 ===")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
