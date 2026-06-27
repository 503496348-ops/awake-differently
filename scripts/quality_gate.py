"""
Awake Differently — Quality Gate
=================================
Pipeline stage validation inspired by ComfyUI's type system and validation.

Between each analysis stage, quality gates validate:
1. Data completeness (required fields present)
2. Statistical validity (sufficient data points, confidence thresholds)
3. Consistency (cross-stage outputs align)
4. Bias detection (skewed data, sampling issues)

Failing a quality gate can: warn, retry with different parameters, or halt pipeline.

Author: AtomCollide-智械工坊
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("awake_differently.quality_gate")


# ─── Gate Severity Levels ───────────────────────────────────────────────────────

class GateAction(str, Enum):
    """What happens when a gate check fails."""
    PASS = "pass"           # Check passed
    WARN = "warn"           # Warning but continue
    FAIL = "fail"           # Hard failure, halt pipeline
    RETRY = "retry"         # Retry with adjusted parameters


class GateSeverity(str, Enum):
    """Severity of quality issues detected."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class GateCheckResult:
    """Result of a single quality gate check."""
    check_name: str
    gate_action: GateAction
    severity: GateSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    score: float = 1.0  # 0.0 = worst, 1.0 = best
    timestamp: float = field(default_factory=time.time)

    @property
    def severity_weight(self) -> float:
        return {
            GateSeverity.INFO: 0.5,
            GateSeverity.WARNING: 1.0,
            GateSeverity.ERROR: 2.0,
            GateSeverity.CRITICAL: 3.0,
        }[self.severity]


@dataclass
class GateReport:
    """Complete quality report for a pipeline stage."""
    stage_name: str
    checks: List[GateCheckResult] = field(default_factory=list)
    overall_action: GateAction = GateAction.PASS
    overall_score: float = 1.0
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def add_check(self, result: GateCheckResult) -> None:
        self.checks.append(result)
        self._update_overall()

    def _update_overall(self) -> None:
        if not self.checks:
            self.overall_action = GateAction.PASS
            self.overall_score = 1.0
            return
        
        # Overall action is worst of all checks
        action_priority = {GateAction.PASS: 0, GateAction.WARN: 1, GateAction.RETRY: 2, GateAction.FAIL: 3}
        worst = max(self.checks, key=lambda c: action_priority[c.gate_action])
        self.overall_action = worst.gate_action
        
        # Overall score is weighted average
        total_weight = sum(c.severity_weight for c in self.checks) if self.checks else 1
        self.overall_score = sum(c.score * c.severity_weight for c in self.checks) / total_weight

    def to_dict(self) -> dict:
        return {
            "stage": self.stage_name,
            "action": self.overall_action.value,
            "score": round(self.overall_score, 3),
            "checks": [
                {
                    "name": c.check_name,
                    "action": c.gate_action.value,
                    "severity": c.severity.value,
                    "message": c.message,
                    "score": round(c.score, 3),
                    **c.details,
                }
                for c in self.checks
            ],
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


# Add severity_weight property to GateCheckResult


# ─── Individual Quality Checks ──────────────────────────────────────────────────

class DataCompletenessChecks:
    """Validate that analysis data has required fields and sufficient content."""

    @staticmethod
    def check_chat_data_minimum(chat_data: dict, min_messages: int = 50) -> GateCheckResult:
        """Check if we have enough messages for meaningful analysis."""
        target_count = chat_data.get("target_count", 0)
        
        if target_count >= min_messages * 2:
            return GateCheckResult(
                check_name="chat_data_minimum",
                gate_action=GateAction.PASS,
                severity=GateSeverity.INFO,
                message=f"数据量充足: {target_count}条 (最低要求{min_messages})",
                score=1.0,
                details={"count": target_count, "minimum": min_messages},
            )
        elif target_count >= min_messages:
            return GateCheckResult(
                check_name="chat_data_minimum",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message=f"数据量刚好达标: {target_count}条, 建议增加到{min_messages * 2}+以提升置信度",
                score=0.6,
                details={"count": target_count, "minimum": min_messages, "recommended": min_messages * 2},
            )
        else:
            return GateCheckResult(
                check_name="chat_data_minimum",
                gate_action=GateAction.FAIL,
                severity=GateSeverity.ERROR,
                message=f"数据量不足: {target_count}条, 最低要求{min_messages}条",
                score=0.0,
                details={"count": target_count, "minimum": min_messages},
            )

    @staticmethod
    def check_time_span(chat_data: dict, min_days: int = 7) -> GateCheckResult:
        """Check if data covers sufficient time span."""
        all_messages = chat_data.get("all_messages", [])
        if not all_messages:
            return GateCheckResult(
                check_name="time_span",
                gate_action=GateAction.FAIL,
                severity=GateSeverity.ERROR,
                message="无消息数据，无法评估时间跨度",
                score=0.0,
            )

        timestamps = []
        for msg in all_messages:
            ts = msg.get("timestamp", "")
            if ts:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    timestamps.append(dt)
                except (ValueError, TypeError):
                    pass

        if len(timestamps) < 2:
            return GateCheckResult(
                check_name="time_span",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message="时间戳数据不足，无法准确评估时间跨度",
                score=0.3,
            )

        span_days = (max(timestamps) - min(timestamps)).days + 1
        
        if span_days >= min_days * 2:
            return GateCheckResult(
                check_name="time_span",
                gate_action=GateAction.PASS,
                severity=GateSeverity.INFO,
                message=f"时间跨度充足: {span_days}天 (最低{min_days}天)",
                score=1.0,
                details={"span_days": span_days, "minimum": min_days},
            )
        elif span_days >= min_days:
            return GateCheckResult(
                check_name="time_span",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message=f"时间跨度刚好达标: {span_days}天, 建议覆盖{min_days * 2}+天",
                score=0.6,
                details={"span_days": span_days, "minimum": min_days},
            )
        else:
            return GateCheckResult(
                check_name="time_span",
                gate_action=GateAction.FAIL,
                severity=GateSeverity.ERROR,
                message=f"时间跨度不足: {span_days}天, 最低要求{min_days}天",
                score=0.0,
                details={"span_days": span_days, "minimum": min_days},
            )

    @staticmethod
    def check_required_fields(data: dict, required_fields: list, context: str = "") -> GateCheckResult:
        """Check that all required fields are present and non-empty."""
        missing = []
        empty = []
        for field_name in required_fields:
            if field_name not in data:
                missing.append(field_name)
            elif data[field_name] is None or data[field_name] == "" or data[field_name] == []:
                empty.append(field_name)

        if not missing and not empty:
            return GateCheckResult(
                check_name=f"required_fields_{context}",
                gate_action=GateAction.PASS,
                severity=GateSeverity.INFO,
                message=f"所有必需字段完整 ({len(required_fields)}个)",
                score=1.0,
                details={"fields": required_fields},
            )
        
        issues = []
        if missing:
            issues.append(f"缺失: {', '.join(missing)}")
        if empty:
            issues.append(f"为空: {', '.join(empty)}")
        
        severity = GateSeverity.ERROR if missing else GateSeverity.WARNING
        action = GateAction.FAIL if missing else GateAction.WARN
        
        return GateCheckResult(
            check_name=f"required_fields_{context}",
            gate_action=action,
            severity=severity,
            message=f"字段问题: {'; '.join(issues)}",
            score=max(0, 1.0 - (len(missing) + len(empty)) * 0.2),
            details={"missing": missing, "empty": empty},
        )


class StatisticalValidityChecks:
    """Validate statistical properties of analysis data."""

    @staticmethod
    def check_sample_diversity(chat_data: dict) -> GateCheckResult:
        """Check if message distribution is too skewed (e.g., all from one day)."""
        all_messages = chat_data.get("all_messages", [])
        if len(all_messages) < 10:
            return GateCheckResult(
                check_name="sample_diversity",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message="样本量太小，无法评估多样性",
                score=0.3,
            )

        # Day distribution
        from collections import Counter
        day_counts = Counter()
        for msg in all_messages:
            ts = msg.get("timestamp", "")
            if "T" in ts:
                day = ts.split("T")[0]
                day_counts[day] += 1

        if not day_counts:
            return GateCheckResult(
                check_name="sample_diversity",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message="无法解析日期分布",
                score=0.3,
            )

        # Check if any single day has >50% of messages
        total = sum(day_counts.values())
        max_day_count = max(day_counts.values())
        max_day_ratio = max_day_count / total

        # Shannon entropy for diversity
        entropy = -sum((c / total) * math.log2(c / total) for c in day_counts.values() if c > 0)
        max_entropy = math.log2(len(day_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        if max_day_ratio > 0.5:
            return GateCheckResult(
                check_name="sample_diversity",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message=f"数据分布不均: 最高峰日占比{max_day_ratio:.0%}, 可能有偏差",
                score=0.4,
                details={
                    "max_day_ratio": round(max_day_ratio, 3),
                    "normalized_entropy": round(normalized_entropy, 3),
                    "day_count": len(day_counts),
                },
            )
        
        return GateCheckResult(
            check_name="sample_diversity",
            gate_action=GateAction.PASS,
            severity=GateSeverity.INFO,
            message=f"数据分布合理: {len(day_counts)}天, 熵={normalized_entropy:.2f}",
            score=min(1.0, normalized_entropy + 0.3),
            details={
                "max_day_ratio": round(max_day_ratio, 3),
                "normalized_entropy": round(normalized_entropy, 3),
                "day_count": len(day_counts),
            },
        )

    @staticmethod
    def check_confidence_threshold(confidence_report: dict, threshold: float = 0.6) -> GateCheckResult:
        """Check if overall confidence meets threshold."""
        overall = confidence_report.get("overall", 0)
        
        if overall >= threshold * 1.5:
            return GateCheckResult(
                check_name="confidence_threshold",
                gate_action=GateAction.PASS,
                severity=GateSeverity.INFO,
                message=f"置信度优秀: {overall:.1%} (阈值{threshold:.0%})",
                score=1.0,
                details={"confidence": overall, "threshold": threshold},
            )
        elif overall >= threshold:
            return GateCheckResult(
                check_name="confidence_threshold",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message=f"置信度达标但偏低: {overall:.1%} (阈值{threshold:.0%})",
                score=0.6,
                details={"confidence": overall, "threshold": threshold},
            )
        else:
            return GateCheckResult(
                check_name="confidence_threshold",
                gate_action=GateAction.FAIL,
                severity=GateSeverity.ERROR,
                message=f"置信度不足: {overall:.1%} (阈值{threshold:.0%})",
                score=0.3,
                details={"confidence": overall, "threshold": threshold},
            )


class ConsistencyChecks:
    """Validate consistency across pipeline stages."""

    @staticmethod
    def check_personality_style_alignment(personality: dict, writing_style: dict) -> GateCheckResult:
        """Check if personality model and writing style are consistent."""
        traits = personality.get("layers", {}).get("core_traits", [])
        trait_names = [t.get("trait", "") for t in traits]
        
        tone = writing_style.get("dominant_tone", "")
        style_info = writing_style.get("sentence_patterns", {})
        avg_len = style_info.get("avg_length", 50)

        # Consistency rules
        inconsistencies = []
        
        if "言简意赅" in trait_names and avg_len > 100:
            inconsistencies.append(f"性格标签'言简意赅'与平均句长{avg_len}字矛盾")
        
        if "社交活跃" in trait_names and tone == "casual" and writing_style.get("emoji_usage", {}).get("frequency", 0) < 0.1:
            inconsistencies.append("性格标签'社交活跃'与低表情使用率矛盾")

        if inconsistencies:
            return GateCheckResult(
                check_name="personality_style_alignment",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message=f"性格-风格一致性存疑: {'; '.join(inconsistencies)}",
                score=0.5,
                details={"inconsistencies": inconsistencies},
            )

        return GateCheckResult(
            check_name="personality_style_alignment",
            gate_action=GateAction.PASS,
            severity=GateSeverity.INFO,
            message="性格模型与写作风格一致",
            score=0.9,
        )

    @staticmethod
    def check_behavior_personality_alignment(behavior: dict, personality: dict) -> GateCheckResult:
        """Check if behavior patterns support personality claims."""
        patterns = behavior.get("patterns", [])
        layers = personality.get("layers", {})
        
        # Check temporal → personality alignment
        temporal = next((p for p in patterns if p["type"] == "temporal"), None)
        core_traits = layers.get("core_traits", [])
        trait_names = [t.get("trait", "") for t in core_traits]

        unsupported = []
        if "夜行侠" in trait_names:
            if temporal and not temporal.get("night_owl"):
                unsupported.append("标签'夜行侠'但夜间消息占比不足")
        if "社交活跃" in trait_names:
            social = next((p for p in patterns if p["type"] == "social"), None)
            if social and social.get("interaction_breadth", 0) < 3:
                unsupported.append("标签'社交活跃'但互动人数不足3人")

        if unsupported:
            return GateCheckResult(
                check_name="behavior_personality_alignment",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message=f"行为-性格部分不一致: {'; '.join(unsupported)}",
                score=0.4,
                details={"unsupported_claims": unsupported},
            )

        return GateCheckResult(
            check_name="behavior_personality_alignment",
            gate_action=GateAction.PASS,
            severity=GateSeverity.INFO,
            message="行为模式与性格模型一致",
            score=0.9,
        )


class BiasDetectionChecks:
    """Detect potential biases in the analysis data."""

    @staticmethod
    def check_sender_bias(chat_data: dict) -> GateCheckResult:
        """Check if data is too skewed toward one sender (observer bias)."""
        all_messages = chat_data.get("all_messages", [])
        target_name = chat_data.get("person_name", "")
        
        if not all_messages:
            return GateCheckResult(
                check_name="sender_bias",
                gate_action=GateAction.PASS,
                severity=GateSeverity.INFO,
                message="无数据可分析",
                score=0.5,
            )

        from collections import Counter
        sender_counts = Counter(m.get("sender", "unknown") for m in all_messages)
        total = len(all_messages)
        
        if target_name and target_name in sender_counts:
            target_ratio = sender_counts[target_name] / total
            if target_ratio > 0.7:
                return GateCheckResult(
                    check_name="sender_bias",
                    gate_action=GateAction.WARN,
                    severity=GateSeverity.WARNING,
                    message=f"目标人物消息占比过高({target_ratio:.0%}), 可能缺乏多角度数据",
                    score=0.5,
                    details={"target_ratio": round(target_ratio, 3), "total_messages": total},
                )
            elif target_ratio < 0.1:
                return GateCheckResult(
                    check_name="sender_bias",
                    gate_action=GateAction.WARN,
                    severity=GateSeverity.WARNING,
                    message=f"目标人物消息占比过低({target_ratio:.0%}), 数据可能不够针对性",
                    score=0.4,
                    details={"target_ratio": round(target_ratio, 3), "total_messages": total},
                )

        return GateCheckResult(
            check_name="sender_bias",
            gate_action=GateAction.PASS,
            severity=GateSeverity.INFO,
            message="发送者分布合理",
            score=0.8,
            details={"sender_count": len(sender_counts)},
        )

    @staticmethod
    def check_context_bias(chat_data: dict) -> GateCheckResult:
        """Check if data only covers one context (e.g., only group chat, no private)."""
        # This is a meta-check since we can't always know the context
        msg_types = set()
        for msg in chat_data.get("all_messages", []):
            msg_types.add(msg.get("msg_type", "text"))

        if len(msg_types) <= 1:
            return GateCheckResult(
                check_name="context_bias",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message="数据仅包含单一消息类型，可能缺乏场景多样性",
                score=0.4,
                details={"message_types": list(msg_types)},
            )

        return GateCheckResult(
            check_name="context_bias",
            gate_action=GateAction.PASS,
            severity=GateSeverity.INFO,
            message=f"消息类型多样: {', '.join(msg_types)}",
            score=0.8,
            details={"message_types": list(msg_types)},
        )


# ─── Quality Gate Runner ────────────────────────────────────────────────────────

class QualityGateRunner:
    """Run quality gates between pipeline stages.
    
    Analogous to ComfyUI's validate_node_input but for analysis quality.
    Each stage has a set of checks that must pass before proceeding.
    
    Usage:
        gate = QualityGateRunner()
        
        # After loading data
        report = gate.check_data_quality(chat_data)
        if report.overall_action == GateAction.FAIL:
            raise QualityGateError(report)
        
        # After analysis
        report = gate.check_analysis_quality(behavior_patterns, personality_model, ...)
        if report.overall_action == GateAction.FAIL:
            raise QualityGateError(report)
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._reports: List[GateReport] = []
        self._custom_checks: Dict[str, Callable] = {}

    def check_data_quality(self, chat_data: dict) -> GateReport:
        """Gate 1: Validate raw data quality before analysis begins."""
        report = GateReport(stage_name="data_quality")
        start = time.monotonic()

        report.add_check(DataCompletenessChecks.check_chat_data_minimum(chat_data))
        report.add_check(DataCompletenessChecks.check_time_span(chat_data))
        report.add_check(DataCompletenessChecks.check_required_fields(
            chat_data,
            ["all_messages", "target_messages", "person_name"],
            context="chat_data",
        ))
        report.add_check(BiasDetectionChecks.check_sender_bias(chat_data))
        report.add_check(BiasDetectionChecks.check_context_bias(chat_data))
        report.add_check(StatisticalValidityChecks.check_sample_diversity(chat_data))

        report.execution_time_ms = (time.monotonic() - start) * 1000
        self._reports.append(report)
        
        self._log_report(report)
        return report

    def check_pattern_quality(self, behavior_patterns: dict) -> GateReport:
        """Gate 2: Validate behavior pattern extraction quality."""
        report = GateReport(stage_name="pattern_quality")
        start = time.monotonic()

        # Check patterns exist
        patterns = behavior_patterns.get("patterns", [])
        if not patterns:
            report.add_check(GateCheckResult(
                check_name="patterns_exist",
                gate_action=GateAction.FAIL,
                severity=GateSeverity.ERROR,
                message="未提取到任何行为模式",
                score=0.0,
            ))
        else:
            report.add_check(GateCheckResult(
                check_name="patterns_exist",
                gate_action=GateAction.PASS,
                severity=GateSeverity.INFO,
                message=f"提取到{len(patterns)}种行为模式",
                score=1.0,
            ))

        # Check pattern types coverage
        pattern_types = {p.get("type") for p in patterns}
        expected_types = {"temporal", "message_style", "vocabulary", "social"}
        missing_types = expected_types - pattern_types
        
        if missing_types:
            report.add_check(GateCheckResult(
                check_name="pattern_coverage",
                gate_action=GateAction.WARN,
                severity=GateSeverity.WARNING,
                message=f"缺少部分模式类型: {', '.join(missing_types)}",
                score=max(0, 1.0 - len(missing_types) * 0.25),
                details={"missing": list(missing_types)},
            ))
        else:
            report.add_check(GateCheckResult(
                check_name="pattern_coverage",
                gate_action=GateAction.PASS,
                severity=GateSeverity.INFO,
                message="行为模式覆盖完整",
                score=1.0,
            ))

        # Check confidence
        confidence = behavior_patterns.get("confidence", 0)
        report.add_check(GateCheckResult(
            check_name="pattern_confidence",
            gate_action=GateAction.PASS if confidence >= 0.5 else GateAction.WARN,
            severity=GateSeverity.INFO if confidence >= 0.5 else GateSeverity.WARNING,
            message=f"模式提取置信度: {confidence:.1%}",
            score=confidence,
        ))

        report.execution_time_ms = (time.monotonic() - start) * 1000
        self._reports.append(report)
        self._log_report(report)
        return report

    def check_model_quality(
        self,
        personality: dict,
        writing_style: dict,
        behavior_patterns: dict,
    ) -> GateReport:
        """Gate 3: Validate personality model and writing style consistency."""
        report = GateReport(stage_name="model_quality")
        start = time.monotonic()

        # Check personality model completeness
        layers = personality.get("layers", {})
        expected_layers = {"core_traits", "identity", "expression", "decision_framework", "interpersonal"}
        missing_layers = expected_layers - set(layers.keys())
        
        if missing_layers:
            report.add_check(GateCheckResult(
                check_name="personality_completeness",
                gate_action=GateAction.WARN if len(missing_layers) <= 2 else GateAction.FAIL,
                severity=GateSeverity.WARNING,
                message=f"人格模型缺少层级: {', '.join(missing_layers)}",
                score=max(0, 1.0 - len(missing_layers) * 0.2),
                details={"missing_layers": list(missing_layers)},
            ))
        else:
            report.add_check(GateCheckResult(
                check_name="personality_completeness",
                gate_action=GateAction.PASS,
                severity=GateSeverity.INFO,
                message="人格模型5层结构完整",
                score=1.0,
            ))

        # Check consistency
        report.add_check(ConsistencyChecks.check_personality_style_alignment(personality, writing_style))
        report.add_check(ConsistencyChecks.check_behavior_personality_alignment(behavior_patterns, personality))

        # Check writing style has key components
        style_fields = ["sentence_patterns", "dominant_tone", "punctuation_style"]
        report.add_check(DataCompletenessChecks.check_required_fields(
            writing_style, style_fields, context="writing_style",
        ))

        report.execution_time_ms = (time.monotonic() - start) * 1000
        self._reports.append(report)
        self._log_report(report)
        return report

    def check_output_quality(self, confidence_report: dict) -> GateReport:
        """Gate 4: Final output quality validation."""
        report = GateReport(stage_name="output_quality")
        start = time.monotonic()

        report.add_check(StatisticalValidityChecks.check_confidence_threshold(confidence_report))

        # Check that recommendations exist
        recs = confidence_report.get("recommendations", [])
        report.add_check(GateCheckResult(
            check_name="recommendations_present",
            gate_action=GateAction.PASS if recs else GateAction.WARN,
            severity=GateSeverity.INFO if recs else GateSeverity.WARNING,
            message=f"生成了{len(recs)}条改进建议" if recs else "未生成改进建议",
            score=0.9 if recs else 0.4,
        ))

        # Check layer confidences
        layer_conf = confidence_report.get("layer_confidences", {})
        if layer_conf:
            low_conf_layers = [k for k, v in layer_conf.items() if v < 0.5]
            if low_conf_layers:
                report.add_check(GateCheckResult(
                    check_name="layer_confidence",
                    gate_action=GateAction.WARN,
                    severity=GateSeverity.WARNING,
                    message=f"部分层级置信度偏低: {', '.join(low_conf_layers)}",
                    score=0.5,
                    details={"low_confidence_layers": low_conf_layers},
                ))
            else:
                report.add_check(GateCheckResult(
                    check_name="layer_confidence",
                    gate_action=GateAction.PASS,
                    severity=GateSeverity.INFO,
                    message="所有层级置信度达标",
                    score=0.9,
                ))

        report.execution_time_ms = (time.monotonic() - start) * 1000
        self._reports.append(report)
        self._log_report(report)
        return report

    def run_all_gates(
        self,
        chat_data: dict,
        behavior_patterns: dict,
        personality: dict,
        writing_style: dict,
        confidence_report: dict,
    ) -> List[GateReport]:
        """Run all quality gates in sequence. Returns list of reports."""
        reports = []
        
        reports.append(self.check_data_quality(chat_data))
        reports.append(self.check_pattern_quality(behavior_patterns))
        reports.append(self.check_model_quality(personality, writing_style, behavior_patterns))
        reports.append(self.check_output_quality(confidence_report))

        return reports

    def register_custom_check(self, name: str, check_fn: Callable[..., GateCheckResult]) -> None:
        """Register a custom quality check function."""
        self._custom_checks[name] = check_fn
        logger.info("Registered custom quality check: %s", name)

    def get_all_reports(self) -> List[dict]:
        """Return all reports as serializable dicts."""
        return [r.to_dict() for r in self._reports]

    def get_summary(self) -> dict:
        """Get summary of all gate reports."""
        if not self._reports:
            return {"total_reports": 0, "overall": "no_data"}
        
        total_checks = sum(len(r.checks) for r in self._reports)
        failed = sum(1 for r in self._reports if r.overall_action == GateAction.FAIL)
        warned = sum(1 for r in self._reports if r.overall_action == GateAction.WARN)
        passed = sum(1 for r in self._reports if r.overall_action == GateAction.PASS)
        avg_score = sum(r.overall_score for r in self._reports) / len(self._reports)

        return {
            "total_reports": len(self._reports),
            "total_checks": total_checks,
            "passed": passed,
            "warned": warned,
            "failed": failed,
            "average_score": round(avg_score, 3),
            "overall": "PASS" if failed == 0 else "FAIL",
        }

    def _log_report(self, report: GateReport) -> None:
        """Log gate report."""
        level = {
            GateAction.PASS: logging.INFO,
            GateAction.WARN: logging.WARNING,
            GateAction.RETRY: logging.WARNING,
            GateAction.FAIL: logging.ERROR,
        }[report.overall_action]
        
        logger.log(
            level,
            "Quality Gate [%s]: %s (score=%.2f, checks=%d)",
            report.stage_name,
            report.overall_action.value,
            report.overall_score,
            len(report.checks),
        )
        for check in report.checks:
            if check.gate_action != GateAction.PASS:
                logger.log(
                    logging.WARNING if check.gate_action == GateAction.WARN else logging.ERROR,
                    "  ├─ %s: %s",
                    check.check_name,
                    check.message,
                )


# ─── Exception Class ────────────────────────────────────────────────────────────

class QualityGateError(Exception):
    """Raised when a quality gate fails in strict mode."""
    
    def __init__(self, report: GateReport):
        self.report = report
        failed_checks = [c for c in report.checks if c.gate_action == GateAction.FAIL]
        msg = f"Quality gate '{report.stage_name}' failed with {len(failed_checks)} error(s):"
        for check in failed_checks:
            msg += f"\n  - {check.check_name}: {check.message}"
        super().__init__(msg)


# ─── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    
    gate = QualityGateRunner()
    
    # Test with sample data
    sample_chat = {
        "all_messages": [
            {"sender": "用户A", "text": "好的", "timestamp": "2026-05-01T10:30:00", "msg_type": "text"},
            {"sender": "用户A", "text": "可以", "timestamp": "2026-05-01T11:00:00", "msg_type": "text"},
            {"sender": "PR", "text": "你觉得", "timestamp": "2026-05-01T11:05:00", "msg_type": "text"},
            {"sender": "用户A", "text": "别催", "timestamp": "2026-05-01T14:00:00", "msg_type": "text"},
        ],
        "target_messages": [
            {"sender": "用户A", "text": "好的", "timestamp": "2026-05-01T10:30:00"},
            {"sender": "用户A", "text": "可以", "timestamp": "2026-05-01T11:00:00"},
            {"sender": "用户A", "text": "别催", "timestamp": "2026-05-01T14:00:00"},
        ],
        "person_name": "用户A",
        "target_count": 3,
        "total_count": 4,
    }

    report = gate.check_data_quality(sample_chat)
    print("\n=== Data Quality Report ===")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    
    print("\n=== Gate Summary ===")
    print(json.dumps(gate.get_summary(), ensure_ascii=False, indent=2))
