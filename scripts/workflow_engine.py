"""
Awake Differently — Workflow Engine
====================================
Composable node-based analysis pipeline inspired by ComfyUI's DAG execution model.

Instead of a single monolithic analysis function, identity analysis is broken into
typed nodes that connect via a directed acyclic graph (DAG). Each node declares its
input/output types, enabling validation and caching at every stage.

Pipeline: ChatData → BehaviorPatterns → PersonalityModel → WritingStyle → VisualPersona

Author: AtomCollide-智械工坊
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

logger = logging.getLogger("awake_differently.workflow")


# ─── Type System (inspired by ComfyUI's IO types) ───────────────────────────────

class AnalysisType(str, Enum):
    """Typed data flowing between analysis nodes, analogous to ComfyUI's IO types."""
    CHAT_DATA = "CHAT_DATA"
    BEHAVIOR_PATTERNS = "BEHAVIOR_PATTERNS"
    PERSONALITY_MODEL = "PERSONALITY_MODEL"
    WRITING_STYLE = "WRITING_STYLE"
    VISUAL_PERSONA = "VISUAL_PERSONA"
    RELATIONSHIP_GRAPH = "RELATIONSHIP_GRAPH"
    CONFIDENCE_REPORT = "CONFIDENCE_REPORT"
    ANY = "ANY"  # wildcard, like ComfyUI's '*' type


def validate_type(received: AnalysisType, expected: AnalysisType) -> bool:
    """Validate that output type from one node matches input type of next.
    Mirrors ComfyUI's validate_node_input logic."""
    if expected == AnalysisType.ANY or received == AnalysisType.ANY:
        return True
    return received == expected


# ─── Cache (inspired by ComfyUI's HierarchicalCache) ────────────────────────────

class AnalysisCache:
    """Content-addressable cache for intermediate analysis results.
    
    Keyed by input signature hash, so identical inputs always hit cache.
    Mirrors ComfyUI's CacheKeySetInputSignature pattern.
    """

    def __init__(self, max_size: int = 128):
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._max_size = max_size

    def _make_key(self, node_id: str, inputs: dict) -> str:
        raw = json.dumps({"node": node_id, "inputs": inputs}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, node_id: str, inputs: dict) -> Optional[Any]:
        key = self._make_key(node_id, inputs)
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug("Cache HIT for node %s (key=%s)", node_id, key)
            return self._cache[key]
        return None

    def put(self, node_id: str, inputs: dict, output: Any) -> None:
        key = self._make_key(node_id, inputs)
        if len(self._cache) >= self._max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[key] = output
        self._access_order.append(key)
        logger.debug("Cache STORE for node %s (key=%s)", node_id, key)

    def invalidate(self, node_id: str) -> int:
        """Remove all cache entries for a specific node. Returns count removed."""
        to_remove = [k for k, v in self._cache.items() if k.startswith("")]
        # More precise: rebuild
        removed = 0
        keys_to_remove = []
        for k in self._cache:
            # We can't easily reverse the hash, so clear all on invalidate
            keys_to_remove.append(k)
        for k in keys_to_remove:
            del self._cache[k]
            self._access_order.remove(k)
            removed += 1
        return removed

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()


# ─── Node Definition (inspired by ComfyNodeABC) ────────────────────────────────

@dataclass
class NodePort:
    """An input or output port on a node."""
    name: str
    type: AnalysisType
    required: bool = True
    description: str = ""


@dataclass
class NodeResult:
    """Result from executing a node, analogous to ComfyUI's cached output."""
    outputs: Dict[str, Any]
    execution_time_ms: float
    from_cache: bool
    node_id: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnalysisNode:
    """Base class for all analysis nodes. Mirrors ComfyNodeABC.
    
    Subclasses must define:
        INPUT_PORTS: list of NodePort (what data this node consumes)
        OUTPUT_PORTS: list of NodePort (what data this node produces)
        CATEGORY: str (for grouping in UI)
        execute(): the actual analysis logic
    """

    INPUT_PORTS: List[NodePort] = []
    OUTPUT_PORTS: List[NodePort] = []
    CATEGORY: str = "analysis"
    DESCRIPTION: str = ""

    @property
    def node_id(self) -> str:
        return self.__class__.__name__

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Run the analysis. kwargs match INPUT_PORTS names."""
        raise NotImplementedError(f"{self.node_id} must implement execute()")

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate inputs against port definitions. Returns list of errors."""
        errors = []
        for port in self.INPUT_PORTS:
            if port.required and port.name not in inputs:
                errors.append(f"Missing required input '{port.name}' ({port.type.value})")
            if port.name in inputs and inputs[port.name] is None and port.required:
                errors.append(f"Input '{port.name}' is None but required")
        return errors


# ─── Concrete Analysis Nodes ────────────────────────────────────────────────────

class ChatDataLoader(AnalysisNode):
    """Load and normalize raw chat data into structured format."""

    INPUT_PORTS = [
        NodePort("raw_messages", AnalysisType.ANY, required=True, description="Raw chat messages (list of dicts)"),
        NodePort("person_name", AnalysisType.ANY, required=False, description="Target person name"),
    ]
    OUTPUT_PORTS = [
        NodePort("chat_data", AnalysisType.CHAT_DATA, required=True),
    ]
    CATEGORY = "data_loader"
    DESCRIPTION = "Load, validate, and normalize chat data for downstream analysis."

    def execute(self, **kwargs) -> Dict[str, Any]:
        raw_messages = kwargs["raw_messages"]
        person_name = kwargs.get("person_name", "")

        if not isinstance(raw_messages, list):
            raise ValueError(f"raw_messages must be a list, got {type(raw_messages)}")

        # Normalize: ensure all messages have required fields
        normalized = []
        for i, msg in enumerate(raw_messages):
            if not isinstance(msg, dict):
                logger.warning("Skipping non-dict message at index %d", i)
                continue
            normalized.append({
                "sender": msg.get("sender", "unknown"),
                "text": msg.get("text", ""),
                "timestamp": msg.get("timestamp", ""),
                "msg_type": msg.get("msg_type", "text"),
                "raw": msg,
            })

        # Filter for target person if specified
        target_msgs = [m for m in normalized if m["sender"] == person_name] if person_name else normalized

        return {
            "chat_data": {
                "all_messages": normalized,
                "target_messages": target_msgs,
                "person_name": person_name,
                "total_count": len(normalized),
                "target_count": len(target_msgs),
            }
        }


class BehaviorPatternExtractor(AnalysisNode):
    """Extract behavioral patterns: active hours, message frequency, response style."""

    INPUT_PORTS = [
        NodePort("chat_data", AnalysisType.CHAT_DATA, required=True),
    ]
    OUTPUT_PORTS = [
        NodePort("behavior_patterns", AnalysisType.BEHAVIOR_PATTERNS, required=True),
    ]
    CATEGORY = "analysis"
    DESCRIPTION = "Extract behavioral patterns from chat data: temporal, linguistic, and interaction patterns."

    def execute(self, **kwargs) -> Dict[str, Any]:
        chat_data = kwargs["chat_data"]
        target_msgs = chat_data["target_messages"]

        if not target_msgs:
            return {"behavior_patterns": {"patterns": [], "confidence": 0.0}}

        patterns = []

        # Temporal pattern: active hours
        hours = Counter()
        weekdays = Counter()
        for msg in target_msgs:
            ts = msg.get("timestamp", "")
            if "T" in ts:
                try:
                    h = int(ts.split("T")[1][:2])
                    hours[h] += 1
                except (ValueError, IndexError):
                    pass
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    weekdays[dt.strftime("%A")] += 1
                except (ValueError, TypeError):
                    pass

        total = len(target_msgs)
        if hours:
            peak_hours = [h for h, _ in hours.most_common(3)]
            night_ratio = sum(hours.get(h, 0) for h in range(0, 6)) / total
            patterns.append({
                "type": "temporal",
                "peak_hours": peak_hours,
                "night_owl": night_ratio > 0.3,
                "night_ratio": round(night_ratio, 3),
                "hourly_distribution": dict(sorted(hours.items())),
            })

        # Message length pattern
        lengths = [len(msg.get("text", "")) for msg in target_msgs]
        avg_len = sum(lengths) / len(lengths)
        median_len = sorted(lengths)[len(lengths) // 2]
        if avg_len < 30:
            style = "极简(短句为主)"
        elif avg_len < 80:
            style = "简洁(中等长度)"
        else:
            style = "详细(长回复)"
        patterns.append({
            "type": "message_style",
            "avg_length": round(avg_len, 1),
            "median_length": median_len,
            "style": style,
        })

        # Vocabulary fingerprint
        import re
        all_text = " ".join(msg.get("text", "") for msg in target_msgs)
        bigrams = Counter()
        for i in range(len(all_text) - 1):
            bigram = all_text[i:i + 2]
            if re.match(r"[\u4e00-\u9fff]{2}", bigram):
                bigrams[bigram] += 1
        signature = [p for p, c in bigrams.most_common(20) if c > 3]
        patterns.append({
            "type": "vocabulary",
            "signature_phrases": signature[:10],
            "vocabulary_size": len(set(all_text)),
        })

        # Interaction pattern
        contacts = Counter()
        for msg in chat_data["all_messages"]:
            if msg["sender"] != chat_data["person_name"]:
                contacts[msg["sender"]] += 1
        patterns.append({
            "type": "social",
            "top_contacts": dict(contacts.most_common(10)),
            "interaction_breadth": len(contacts),
        })

        return {
            "behavior_patterns": {
                "patterns": patterns,
                "confidence": min(0.95, total / 500),  # confidence scales with data
                "data_points": total,
            }
        }


class PersonalityModeler(AnalysisNode):
    """Build multi-layer personality model from behavior patterns."""

    INPUT_PORTS = [
        NodePort("behavior_patterns", AnalysisType.BEHAVIOR_PATTERNS, required=True),
        NodePort("chat_data", AnalysisType.CHAT_DATA, required=True),
    ]
    OUTPUT_PORTS = [
        NodePort("personality_model", AnalysisType.PERSONALITY_MODEL, required=True),
    ]
    CATEGORY = "modeling"
    DESCRIPTION = "Build a 5-layer personality model: core traits, identity, expression, decision framework, interpersonal."

    def execute(self, **kwargs) -> Dict[str, Any]:
        behavior = kwargs["behavior_patterns"]
        chat_data = kwargs["chat_data"]
        patterns = behavior.get("patterns", [])

        personality = {
            "layers": {},
            "overall_confidence": behavior.get("confidence", 0.0),
        }

        # Layer 0: Core traits (inferred from patterns)
        core_traits = []
        temporal = next((p for p in patterns if p["type"] == "temporal"), None)
        if temporal and temporal.get("night_owl"):
            core_traits.append({"trait": "夜行侠", "evidence": f"深夜消息占比{temporal['night_ratio']:.0%}"})
        
        style = next((p for p in patterns if p["type"] == "message_style"), None)
        if style and style["median_length"] < 30:
            core_traits.append({"trait": "言简意赅", "evidence": f"中位数消息长度{style['median_length']}字"})
        
        social = next((p for p in patterns if p["type"] == "social"), None)
        if social and social["interaction_breadth"] > 5:
            core_traits.append({"trait": "社交活跃", "evidence": f"与{social['interaction_breadth']}人有互动"})

        personality["layers"]["core_traits"] = core_traits

        # Layer 1: Identity (from vocabulary analysis)
        vocab = next((p for p in patterns if p["type"] == "vocabulary"), None)
        if vocab:
            personality["layers"]["identity"] = {
                "signature_phrases": vocab["signature_phrases"],
                "vocabulary_richness": vocab["vocabulary_size"],
            }

        # Layer 2: Expression style
        if style:
            personality["layers"]["expression"] = {
                "style": style["style"],
                "avg_length": style["avg_length"],
            }

        # Layer 3: Decision framework (inferred from message content patterns)
        target_msgs = chat_data.get("target_messages", [])
        action_words = ["搞", "干", "做", "来", "发", "推", "卖", "买", "谈"]
        question_words = ["怎么", "什么", "为什么", "如何", "哪"]
        action_count = sum(1 for m in target_msgs if any(w in m.get("text", "") for w in action_words))
        question_count = sum(1 for m in target_msgs if any(w in m.get("text", "") for w in question_words))
        
        if action_count > question_count * 2:
            decision_style = "行动导向"
        elif question_count > action_count:
            decision_style = "探究导向"
        else:
            decision_style = "平衡型"
        
        personality["layers"]["decision_framework"] = {
            "style": decision_style,
            "action_ratio": action_count / max(len(target_msgs), 1),
            "question_ratio": question_count / max(len(target_msgs), 1),
        }

        # Layer 4: Interpersonal behavior
        if social:
            personality["layers"]["interpersonal"] = {
                "network_size": social["interaction_breadth"],
                "top_contacts": social["top_contacts"],
            }

        return {"personality_model": personality}


class WritingStyleAnalyzer(AnalysisNode):
    """Analyze writing style for persona reproduction."""

    INPUT_PORTS = [
        NodePort("personality_model", AnalysisType.PERSONALITY_MODEL, required=True),
        NodePort("chat_data", AnalysisType.CHAT_DATA, required=True),
    ]
    OUTPUT_PORTS = [
        NodePort("writing_style", AnalysisType.WRITING_STYLE, required=True),
    ]
    CATEGORY = "analysis"
    DESCRIPTION = "Analyze and codify writing style: sentence patterns, tone markers, emoji usage."

    def execute(self, **kwargs) -> Dict[str, Any]:
        personality = kwargs["personality_model"]
        chat_data = kwargs["chat_data"]
        target_msgs = chat_data.get("target_messages", [])

        texts = [m.get("text", "") for m in target_msgs if m.get("text")]
        all_text = "\n".join(texts)

        # Sentence structure analysis
        import re
        sentences = re.split(r'[。！？\n]', all_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sentence_lengths = [len(s) for s in sentences]
        avg_sentence_len = sum(sentence_lengths) / max(len(sentence_lengths), 1)

        # Punctuation patterns
        exclamation_count = all_text.count("！") + all_text.count("!")
        question_count = all_text.count("？") + all_text.count("?")
        
        # Emoji/symbol patterns
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
            "\U0001f900-\U0001f9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]+",
            flags=re.UNICODE
        )
        emoji_matches = emoji_pattern.findall(all_text)

        # Common opening patterns
        openers = Counter()
        for s in sentences[:100]:
            if len(s) > 2:
                openers[s[:3]] += 1

        # Tone analysis
        tone_markers = {
            "enthusiastic": exclamation_count / max(len(texts), 1),
            "questioning": question_count / max(len(texts), 1),
            "casual": len(emoji_matches) / max(len(texts), 1),
        }
        dominant_tone = max(tone_markers, key=lambda k: tone_markers[k])

        style = {
            "sentence_patterns": {
                "avg_length": round(avg_sentence_len, 1),
                "short_sentence_ratio": sum(1 for l in sentence_lengths if l < 15) / max(len(sentence_lengths), 1),
            },
            "punctuation_style": {
                "exclamation_rate": round(exclamation_count / max(len(texts), 1), 2),
                "question_rate": round(question_count / max(len(texts), 1), 2),
            },
            "emoji_usage": {
                "frequency": round(len(emoji_matches) / max(len(texts), 1), 2),
                "unique_emojis": list(set(emoji_matches))[:20],
            },
            "dominant_tone": dominant_tone,
            "tone_scores": tone_markers,
            "common_openers": dict(openers.most_common(5)),
            "personality_alignment": self._align_with_personality(personality, tone_markers),
        }

        return {"writing_style": style}

    def _align_with_personality(self, personality: dict, tone: dict) -> str:
        """Check if writing style aligns with personality model."""
        core = personality.get("layers", {}).get("core_traits", [])
        trait_names = [t.get("trait", "") for t in core]
        
        if "言简意赅" in trait_names and tone.get("casual", 0) > 0.3:
            return "高度一致：简洁+活跃风格"
        elif "社交活跃" in trait_names and tone.get("enthusiastic", 0) > 0.5:
            return "高度一致：社交型+热情风格"
        return "需要更多数据验证"


class VisualPersonaGenerator(AnalysisNode):
    """Generate visual persona description from all upstream analysis."""

    INPUT_PORTS = [
        NodePort("personality_model", AnalysisType.PERSONALITY_MODEL, required=True),
        NodePort("writing_style", AnalysisType.WRITING_STYLE, required=True),
        NodePort("behavior_patterns", AnalysisType.BEHAVIOR_PATTERNS, required=True),
    ]
    OUTPUT_PORTS = [
        NodePort("visual_persona", AnalysisType.VISUAL_PERSONA, required=True),
        NodePort("confidence_report", AnalysisType.CONFIDENCE_REPORT, required=True),
    ]
    CATEGORY = "output"
    DESCRIPTION = "Generate comprehensive visual persona description with confidence metrics."

    def execute(self, **kwargs) -> Dict[str, Any]:
        personality = kwargs["personality_model"]
        writing = kwargs["writing_style"]
        behavior = kwargs["behavior_patterns"]

        # Build persona summary
        traits = personality.get("layers", {}).get("core_traits", [])
        trait_desc = "、".join(t.get("trait", "") for t in traits)
        
        style_desc = writing.get("dominant_tone", "平衡")
        tone_map = {
            "enthusiastic": "热情洋溢型",
            "questioning": "探究思考型",
            "casual": "轻松随性型",
        }
        tone_label = tone_map.get(style_desc, style_desc)

        # Decision framework
        decision = personality.get("layers", {}).get("decision_framework", {})
        decision_label = decision.get("style", "未知")

        # Visual persona
        visual = {
            "summary": f"核心特质：{trait_desc}。表达风格：{tone_label}。决策方式：{decision_label}。",
            "personality_layers": personality.get("layers", {}),
            "expression_style": writing.get("sentence_patterns", {}),
            "social_profile": behavior.get("patterns", [{}])[-1] if behavior.get("patterns") else {},
        }

        # Confidence report
        behavior_conf = behavior.get("confidence", 0.0)
        data_points = behavior.get("data_points", 0)
        
        layer_confidences = {}
        for layer_name, layer_data in personality.get("layers", {}).items():
            if isinstance(layer_data, list):
                layer_confidences[layer_name] = min(0.95, len(layer_data) * 0.2)
            elif isinstance(layer_data, dict):
                layer_confidences[layer_name] = min(0.95, len(layer_data) * 0.15 + 0.3)

        overall_conf = sum(layer_confidences.values()) / max(len(layer_confidences), 1)
        
        confidence_report = {
            "overall": round(overall_conf, 3),
            "data_quality": round(behavior_conf, 3),
            "data_points": data_points,
            "layer_confidences": layer_confidences,
            "recommendations": self._get_recommendations(data_points, overall_conf),
        }

        return {
            "visual_persona": visual,
            "confidence_report": confidence_report,
        }

    def _get_recommendations(self, data_points: int, confidence: float) -> List[str]:
        recs = []
        if data_points < 200:
            recs.append(f"数据量不足（{data_points}条），建议增加到200+条以提升置信度")
        if data_points < 500:
            recs.append("建议增加数据时间跨度到14天以上")
        if confidence < 0.7:
            recs.append("整体置信度偏低，建议增加多场景数据（群聊+私聊+工作文档）")
        if not recs:
            recs.append("数据质量良好，可进入持续进化阶段")
        return recs


# ─── Topological Sort (directly from ComfyUI's pattern) ─────────────────────────

class DAGCycleError(Exception):
    pass


class TopologicalSorter:
    """Sort nodes in dependency order. Direct adaptation of ComfyUI's TopologicalSort."""

    def __init__(self, adjacency: Dict[str, Set[str]]):
        self.adjacency = adjacency
        self.in_degree: Dict[str, int] = defaultdict(int)
        for node in adjacency:
            if node not in self.in_degree:
                self.in_degree[node] = 0
            for dep in adjacency[node]:
                self.in_degree[dep] += 1

    def sort(self) -> List[str]:
        """Kahn's algorithm. Returns execution order. Raises DAGCycleError on cycles."""
        queue = [n for n in self.adjacency if self.in_degree[n] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in self.adjacency.get(node, []):
                self.in_degree[neighbor] -= 1
                if self.in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.adjacency):
            raise DAGCycleError(
                f"Cycle detected in workflow DAG. "
                f"Processed {len(result)} of {len(self.adjacency)} nodes."
            )
        return result


# ─── Workflow Definition (analogous to ComfyUI's prompt format) ─────────────────

@dataclass
class WorkflowEdge:
    """A connection between two nodes: from_node.output_port → to_node.input_port."""
    from_node: str
    from_port: str
    to_node: str
    to_port: str


class WorkflowDefinition:
    """Defines a workflow as a set of nodes and edges. Analogous to ComfyUI's prompt JSON.
    
    Example workflow JSON (ComfyUI-style):
    {
        "nodes": {
            "loader": {"class": "ChatDataLoader", "inputs": {"raw_messages": "..."}},
            "behavior": {"class": "BehaviorPatternExtractor"},
            "personality": {"class": "PersonalityModeler"},
            "style": {"class": "WritingStyleAnalyzer"},
            "visual": {"class": "VisualPersonaGenerator"}
        },
        "edges": [
            {"from": "loader", "from_port": "chat_data", "to": "behavior", "to_port": "chat_data"},
            {"from": "behavior", "from_port": "behavior_patterns", "to": "personality", "to_port": "behavior_patterns"},
            ...
        ]
    }
    """

    def __init__(self):
        self.nodes: Dict[str, AnalysisNode] = {}
        self.edges: List[WorkflowEdge] = []
        self._node_outputs: Dict[str, Dict[str, Any]] = {}  # cached outputs

    def add_node(self, instance_id: str, node: AnalysisNode) -> None:
        self.nodes[instance_id] = node

    def add_edge(self, from_node: str, from_port: str, to_node: str, to_port: str) -> None:
        edge = WorkflowEdge(from_node, from_port, to_node, to_port)
        self.edges.append(edge)

    def validate(self) -> List[str]:
        """Validate the workflow definition. Returns list of errors."""
        errors = []
        
        # Check all edge references are valid
        for edge in self.edges:
            if edge.from_node not in self.nodes:
                errors.append(f"Edge references unknown source node: {edge.from_node}")
            if edge.to_node not in self.nodes:
                errors.append(f"Edge references unknown target node: {edge.to_node}")
            if edge.from_node in self.nodes:
                node = self.nodes[edge.from_node]
                if not any(p.name == edge.from_port for p in node.OUTPUT_PORTS):
                    errors.append(f"Node '{edge.from_node}' has no output port '{edge.from_port}'")
            if edge.to_node in self.nodes:
                node = self.nodes[edge.to_node]
                if not any(p.name == edge.to_port for p in node.INPUT_PORTS):
                    errors.append(f"Node '{edge.to_node}' has no input port '{edge.to_port}'")

        # Type validation (like ComfyUI's validate_node_input)
        for edge in self.edges:
            if edge.from_node in self.nodes and edge.to_node in self.nodes:
                from_type = next(
                    (p.type for p in self.nodes[edge.from_node].OUTPUT_PORTS if p.name == edge.from_port),
                    None
                )
                to_type = next(
                    (p.type for p in self.nodes[edge.to_node].INPUT_PORTS if p.name == edge.to_port),
                    None
                )
                if from_type and to_type and not validate_type(from_type, to_type):
                    errors.append(
                        f"Type mismatch: {edge.from_node}.{edge.from_port}({from_type.value}) "
                        f"→ {edge.to_node}.{edge.to_port}({to_type.value})"
                    )

        return errors

    def get_execution_order(self) -> List[str]:
        """Topological sort of nodes based on edges."""
        adjacency: Dict[str, Set[str]] = {nid: set() for nid in self.nodes}
        for edge in self.edges:
            adjacency[edge.from_node].add(edge.to_node)
        
        sorter = TopologicalSorter(adjacency)
        return sorter.sort()

    @classmethod
    def from_json(cls, workflow_json: dict, node_registry: Dict[str, Type[AnalysisNode]]) -> "WorkflowDefinition":
        """Load workflow from JSON definition (ComfyUI prompt format)."""
        wf = cls()
        
        for node_id, node_spec in workflow_json.get("nodes", {}).items():
            class_name = node_spec.get("class")
            if class_name not in node_registry:
                raise ValueError(f"Unknown node class: {class_name}")
            wf.add_node(node_id, node_registry[class_name]())
        
        for edge_spec in workflow_json.get("edges", []):
            wf.add_edge(
                edge_spec["from"], edge_spec["from_port"],
                edge_spec["to"], edge_spec["to_port"],
            )
        
        return wf


# ─── Workflow Executor (analogous to ComfyUI's execution.py) ────────────────────

class WorkflowExecutor:
    """Execute a workflow by running nodes in topological order.
    
    Features inspired by ComfyUI:
    - Topological execution order
    - Content-addressable caching
    - Type validation between stages
    - Error isolation per node
    """

    def __init__(self, workflow: WorkflowDefinition, cache: Optional[AnalysisCache] = None):
        self.workflow = workflow
        self.cache = cache or AnalysisCache()
        self._execution_log: List[NodeResult] = []

    def execute(self, initial_inputs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, NodeResult]:
        """Execute the full workflow. Returns results keyed by node instance_id."""
        initial_inputs = initial_inputs or {}
        
        # Validate
        errors = self.workflow.validate()
        if errors:
            raise ValueError(f"Workflow validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        # Get execution order
        exec_order = self.workflow.get_execution_order()
        logger.info("Execution order: %s", " → ".join(exec_order))

        results: Dict[str, NodeResult] = {}

        for node_id in exec_order:
            node = self.workflow.nodes[node_id]

            # Gather inputs from upstream outputs or initial inputs
            inputs = {}
            for port in node.INPUT_PORTS:
                # Check if this port is connected via an edge
                edge = next(
                    (e for e in self.workflow.edges if e.to_node == node_id and e.to_port == port.name),
                    None
                )
                if edge:
                    if edge.from_node in results:
                        inputs[port.name] = results[edge.from_node].outputs.get(edge.from_port)
                    else:
                        raise RuntimeError(
                            f"Node '{node_id}' expects input '{port.name}' from "
                            f"'{edge.from_node}.{edge.from_port}', but that node hasn't executed yet"
                        )
                elif port.name in initial_inputs.get(node_id, {}):
                    inputs[port.name] = initial_inputs[node_id][port.name]
                elif port.required:
                    raise RuntimeError(f"Node '{node_id}' missing required input '{port.name}'")

            # Validate inputs
            validation_errors = node.validate_inputs(inputs)
            if validation_errors:
                raise RuntimeError(f"Node '{node_id}' input validation failed:\n" + "\n".join(validation_errors))

            # Check cache
            cached = self.cache.get(node_id, inputs)
            if cached is not None:
                cached.from_cache = True
                results[node_id] = cached
                self._execution_log.append(cached)
                logger.info("Node '%s': CACHE HIT", node_id)
                continue

            # Execute
            start_time = time.monotonic()
            try:
                outputs = node.execute(**inputs)
            except Exception as e:
                logger.error("Node '%s' failed: %s", node_id, str(e))
                raise RuntimeError(f"Node '{node_id}' execution failed: {e}") from e
            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Build result
            result = NodeResult(
                outputs=outputs,
                execution_time_ms=round(elapsed_ms, 2),
                from_cache=False,
                node_id=node_id,
            )
            results[node_id] = result
            self.cache.put(node_id, inputs, result)
            self._execution_log.append(result)
            logger.info("Node '%s': executed in %.1fms", node_id, elapsed_ms)

        return results

    def get_execution_log(self) -> List[dict]:
        """Return execution log as serializable dicts."""
        return [
            {
                "node_id": r.node_id,
                "execution_time_ms": r.execution_time_ms,
                "from_cache": r.from_cache,
                "output_keys": list(r.outputs.keys()),
            }
            for r in self._execution_log
        ]


# ─── Default Workflow Builder ───────────────────────────────────────────────────

# Registry of all built-in analysis nodes
NODE_REGISTRY: Dict[str, Type[AnalysisNode]] = {
    "ChatDataLoader": ChatDataLoader,
    "BehaviorPatternExtractor": BehaviorPatternExtractor,
    "PersonalityModeler": PersonalityModeler,
    "WritingStyleAnalyzer": WritingStyleAnalyzer,
    "VisualPersonaGenerator": VisualPersonaGenerator,
}


def build_default_identity_workflow() -> WorkflowDefinition:
    """Build the standard 5-stage identity analysis pipeline.
    
    Pipeline:
    ChatDataLoader → BehaviorPatternExtractor → PersonalityModeler 
                                                ↘ WritingStyleAnalyzer 
                                                → VisualPersonaGenerator
    """
    wf = WorkflowDefinition()
    
    wf.add_node("loader", ChatDataLoader())
    wf.add_node("behavior", BehaviorPatternExtractor())
    wf.add_node("personality", PersonalityModeler())
    wf.add_node("style", WritingStyleAnalyzer())
    wf.add_node("visual", VisualPersonaGenerator())

    # loader → behavior
    wf.add_edge("loader", "chat_data", "behavior", "chat_data")
    # behavior → personality
    wf.add_edge("behavior", "behavior_patterns", "personality", "behavior_patterns")
    # loader → personality (needs chat_data too)
    wf.add_edge("loader", "chat_data", "personality", "chat_data")
    # personality → style
    wf.add_edge("personality", "personality_model", "style", "personality_model")
    # loader → style (needs chat_data)
    wf.add_edge("loader", "chat_data", "style", "chat_data")
    # personality → visual
    wf.add_edge("personality", "personality_model", "visual", "personality_model")
    # style → visual
    wf.add_edge("style", "writing_style", "visual", "writing_style")
    # behavior → visual
    wf.add_edge("behavior", "behavior_patterns", "visual", "behavior_patterns")

    return wf


# ─── CLI Entry Point ────────────────────────────────────────────────────────────

def run_analysis(messages: list, person_name: str = "") -> Dict[str, Any]:
    """Convenience function to run the full identity analysis pipeline."""
    workflow = build_default_identity_workflow()
    executor = WorkflowExecutor(workflow)

    initial_inputs = {
        "loader": {
            "raw_messages": messages,
            "person_name": person_name,
        }
    }

    results = executor.execute(initial_inputs)

    # Extract final outputs
    final = results.get("visual")
    if final:
        return {
            "visual_persona": final.outputs.get("visual_persona"),
            "confidence_report": final.outputs.get("confidence_report"),
            "execution_log": executor.get_execution_log(),
        }
    return {"error": "Pipeline did not produce final output"}


if __name__ == "__main__":
    # Demo with sample data
    test_messages = [
        {"sender": "陈龙", "text": "好的收到", "timestamp": "2026-05-01T10:30:00"},
        {"sender": "陈龙", "text": "这个方案可以，直接干", "timestamp": "2026-05-01T11:00:00"},
        {"sender": "PR", "text": "你觉得怎么样", "timestamp": "2026-05-01T11:05:00"},
        {"sender": "陈龙", "text": "别催，正在谈，等我消息", "timestamp": "2026-05-01T14:00:00"},
        {"sender": "陈龙", "text": "兄弟们！福利来了！", "timestamp": "2026-05-01T02:30:00"},
        {"sender": "陈龙", "text": "这个死孩子，我说了多少遍", "timestamp": "2026-05-02T01:00:00"},
        {"sender": "陈龙", "text": "版权5积分，你卖给学姐学哥10块一份", "timestamp": "2026-05-02T02:00:00"},
        {"sender": "陈龙", "text": "我不会，但是我可以学", "timestamp": "2026-05-02T03:00:00"},
        {"sender": "李渔樵", "text": "龙哥这个怎么搞", "timestamp": "2026-05-02T10:00:00"},
        {"sender": "陈龙", "text": "你是真的打算一点脑子不动纯躺", "timestamp": "2026-05-02T10:05:00"},
        {"sender": "小乖助理", "text": "音乐模型训练完了", "timestamp": "2026-05-03T01:00:00"},
        {"sender": "陈龙", "text": "发来听听", "timestamp": "2026-05-03T01:02:00"},
    ]

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    result = run_analysis(test_messages, "陈龙")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
