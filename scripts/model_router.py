"""
Awake Differently — Model Router
=================================
Multi-model pipeline routing inspired by ComfyUI's model management.

Different analysis tasks have different requirements:
- Deep personality modeling needs high-capability models (GPT-4, Claude)
- Fast pattern extraction can use lightweight models (local LLM, GPT-3.5)
- Validation/gates can use rule-based checks without any LLM

This module routes each analysis task to the optimal model based on:
- Task complexity
- Required accuracy
- Cost budget
- Latency requirements

Author: AtomCollide-智械工坊
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger("awake_differently.model_router")


# ─── Model Capability Tiers ─────────────────────────────────────────────────────

class ModelTier(str, Enum):
    """Model capability tiers, from lightweight to heavy-duty."""
    RULE_BASED = "rule_based"     # No LLM needed (regex, stats)
    LIGHTWEIGHT = "lightweight"   # Local/small LLM (7B-13B params)
    STANDARD = "standard"         # Mid-tier (GPT-3.5, Claude Haiku)
    PREMIUM = "premium"           # Top-tier (GPT-4, Claude Opus, DeepSeek)


class TaskComplexity(str, Enum):
    """Complexity classification for analysis tasks."""
    TRIVIAL = "trivial"         # Pattern matching, counting
    LOW = "low"                 # Simple extraction, classification
    MEDIUM = "medium"           # Multi-step reasoning, summarization
    HIGH = "high"               # Deep analysis, creative synthesis


# ─── Model Configuration ────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for a specific model endpoint."""
    model_id: str
    tier: ModelTier
    api_base: str
    api_key_env: str  # environment variable name for API key
    max_tokens: int = 4096
    temperature: float = 0.3
    cost_per_1k_input: float = 0.0   # USD per 1K input tokens
    cost_per_1k_output: float = 0.0  # USD per 1K output tokens
    latency_ms_estimate: int = 1000
    supports_chinese: bool = True
    supports_json_mode: bool = False
    description: str = ""


@dataclass
class TaskRequirement:
    """Requirements for a specific analysis task."""
    task_name: str
    complexity: TaskComplexity
    min_tier: ModelTier
    preferred_tier: ModelTier
    max_latency_ms: int = 30000
    max_cost_usd: float = 0.50
    requires_structured_output: bool = False
    requires_chinese: bool = True
    description: str = ""


@dataclass
class RoutingDecision:
    """Record of a routing decision for audit/replay."""
    task_name: str
    selected_model: str
    reason: str
    tier_used: ModelTier
    estimated_cost_usd: float
    estimated_latency_ms: int
    timestamp: float = field(default_factory=time.time)


# ─── LLM Provider Protocol ─────────────────────────────────────────────────────

class LLMProvider(Protocol):
    """Protocol for LLM providers. Any conforming class can be used."""
    
    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate a response from the model."""
        ...

    def generate_json(self, prompt: str, system_prompt: str = "", schema: Optional[dict] = None, **kwargs) -> dict:
        """Generate a structured JSON response."""
        ...


# ─── Built-in Providers ────────────────────────────────────────────────────────

class RuleBasedProvider:
    """No-LLM provider that uses pure Python logic for trivial tasks."""

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        return prompt  # passthrough — the "analysis" is done in the node itself

    def generate_json(self, prompt: str, system_prompt: str = "", schema: Optional[dict] = None, **kwargs) -> dict:
        return {"result": prompt, "provider": "rule_based"}


class OpenAICompatibleProvider:
    """Provider for any OpenAI-compatible API (OpenAI, DeepSeek, local models)."""

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self._api_key = os.environ.get(model_config.api_key_env, "")

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Call the OpenAI-compatible API."""
        if not self._api_key:
            logger.warning("No API key for %s, falling back to rule-based", self.config.model_id)
            return f"[FALLBACK] {prompt[:500]}"

        try:
            import urllib.request
            import urllib.error

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            body = json.dumps({
                "model": self.config.model_id,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            }).encode()

            req = urllib.request.Request(
                f"{self.config.api_base}/chat/completions",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error("API call to %s failed: %s", self.config.model_id, e)
            return f"[ERROR] API call failed: {e}"

    def generate_json(self, prompt: str, system_prompt: str = "", schema: Optional[dict] = None, **kwargs) -> dict:
        """Call with JSON mode if supported, otherwise parse response."""
        json_system = (system_prompt + "\n\nYou MUST respond with valid JSON only, no markdown.") if system_prompt else \
                      "You MUST respond with valid JSON only, no markdown."
        
        raw = self.generate(prompt, json_system, **kwargs)
        
        # Try to parse JSON from response
        try:
            # Strip markdown code blocks if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_response": raw, "parse_error": True}


# ─── Default Model Configurations ───────────────────────────────────────────────

DEFAULT_MODELS: Dict[str, ModelConfig] = {
    "rule_based": ModelConfig(
        model_id="rule_based",
        tier=ModelTier.RULE_BASED,
        api_base="",
        api_key_env="",
        latency_ms_estimate=10,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
        description="Pure Python rule-based analysis, no LLM needed",
    ),
    "deepseek_chat": ModelConfig(
        model_id="deepseek-chat",
        tier=ModelTier.STANDARD,
        api_base="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        latency_ms_estimate=2000,
        supports_chinese=True,
        supports_json_mode=True,
        description="DeepSeek Chat - excellent Chinese support, good value",
    ),
    "deepseek_reasoner": ModelConfig(
        model_id="deepseek-reasoner",
        tier=ModelTier.PREMIUM,
        api_base="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        max_tokens=8192,
        temperature=0.1,
        cost_per_1k_input=0.00055,
        cost_per_1k_output=0.00219,
        latency_ms_estimate=8000,
        supports_chinese=True,
        description="DeepSeek Reasoner - for deep personality analysis",
    ),
    "gpt4o_mini": ModelConfig(
        model_id="gpt-4o-mini",
        tier=ModelTier.LIGHTWEIGHT,
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        latency_ms_estimate=1500,
        supports_chinese=True,
        supports_json_mode=True,
        description="GPT-4o Mini - fast and affordable",
    ),
    "gpt4o": ModelConfig(
        model_id="gpt-4o",
        tier=ModelTier.PREMIUM,
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4096,
        temperature=0.2,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        latency_ms_estimate=3000,
        supports_chinese=True,
        supports_json_mode=True,
        description="GPT-4o - highest quality analysis",
    ),
    "claude_haiku": ModelConfig(
        model_id="claude-3-haiku-20240307",
        tier=ModelTier.LIGHTWEIGHT,
        api_base="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        latency_ms_estimate=1200,
        supports_chinese=True,
        description="Claude Haiku - fast and capable",
    ),
}


# ─── Task Requirements ─────────────────────────────────────────────────────────

TASK_PROFILES: Dict[str, TaskRequirement] = {
    "chat_data_loading": TaskRequirement(
        task_name="chat_data_loading",
        complexity=TaskComplexity.TRIVIAL,
        min_tier=ModelTier.RULE_BASED,
        preferred_tier=ModelTier.RULE_BASED,
        max_latency_ms=100,
        max_cost_usd=0.0,
        description="Load and normalize raw chat data",
    ),
    "behavior_extraction": TaskRequirement(
        task_name="behavior_extraction",
        complexity=TaskComplexity.LOW,
        min_tier=ModelTier.RULE_BASED,
        preferred_tier=ModelTier.LIGHTWEIGHT,
        max_latency_ms=5000,
        max_cost_usd=0.05,
        requires_structured_output=True,
        description="Extract behavioral patterns (temporal, social, linguistic)",
    ),
    "personality_modeling": TaskRequirement(
        task_name="personality_modeling",
        complexity=TaskComplexity.HIGH,
        min_tier=ModelTier.STANDARD,
        preferred_tier=ModelTier.PREMIUM,
        max_latency_ms=15000,
        max_cost_usd=0.30,
        requires_structured_output=True,
        requires_chinese=True,
        description="Build multi-layer personality model with deep reasoning",
    ),
    "writing_style_analysis": TaskRequirement(
        task_name="writing_style_analysis",
        complexity=TaskComplexity.MEDIUM,
        min_tier=ModelTier.LIGHTWEIGHT,
        preferred_tier=ModelTier.STANDARD,
        max_latency_ms=8000,
        max_cost_usd=0.10,
        requires_structured_output=True,
        requires_chinese=True,
        description="Analyze writing style, tone, and expression patterns",
    ),
    "visual_persona_generation": TaskRequirement(
        task_name="visual_persona_generation",
        complexity=TaskComplexity.MEDIUM,
        min_tier=ModelTier.LIGHTWEIGHT,
        preferred_tier=ModelTier.STANDARD,
        max_latency_ms=10000,
        max_cost_usd=0.15,
        requires_chinese=True,
        description="Generate comprehensive persona description",
    ),
    "quality_validation": TaskRequirement(
        task_name="quality_validation",
        complexity=TaskComplexity.LOW,
        min_tier=ModelTier.RULE_BASED,
        preferred_tier=ModelTier.LIGHTWEIGHT,
        max_latency_ms=3000,
        max_cost_usd=0.02,
        description="Validate analysis output quality",
    ),
    "distillation_report": TaskRequirement(
        task_name="distillation_report",
        complexity=TaskComplexity.MEDIUM,
        min_tier=ModelTier.STANDARD,
        preferred_tier=ModelTier.STANDARD,
        max_latency_ms=12000,
        max_cost_usd=0.20,
        requires_chinese=True,
        description="Generate detailed distillation report",
    ),
}


# ─── Model Router ───────────────────────────────────────────────────────────────

class ModelRouter:
    """Routes analysis tasks to the optimal model based on requirements and constraints.
    
    Inspired by ComfyUI's model management where different nodes can use different
    models (CLIP, VAE, UNet) based on what they need. Similarly, our identity
    analysis pipeline routes different tasks to different LLM tiers.
    
    Usage:
        router = ModelRouter()
        provider = router.get_provider("personality_modeling")
        result = provider.generate(prompt, system_prompt)
    """

    def __init__(
        self,
        models: Optional[Dict[str, ModelConfig]] = None,
        budget_per_run_usd: float = 1.0,
        prefer_speed: bool = False,
    ):
        self._models = models or DEFAULT_MODELS.copy()
        self._providers: Dict[str, LLMProvider] = {}
        self._budget_remaining = budget_per_run_usd
        self._budget_total = budget_per_run_usd
        self._prefer_speed = prefer_speed
        self._routing_log: List[RoutingDecision] = []
        
        # Initialize providers
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize LLM providers based on available API keys."""
        self._providers["rule_based"] = RuleBasedProvider()
        
        for model_id, config in self._models.items():
            if config.tier == ModelTier.RULE_BASED:
                continue
            
            api_key = os.environ.get(config.api_key_env, "")
            if api_key:
                self._providers[model_id] = OpenAICompatibleProvider(config)
                logger.info("Registered model: %s (tier=%s)", model_id, config.tier.value)
            else:
                logger.debug("Skipping model %s: no API key (%s)", model_id, config.api_key_env)

    def get_provider(self, task_name: str) -> Tuple[LLMProvider, RoutingDecision]:
        """Get the best provider for a given task.
        
        Returns (provider, routing_decision) for audit trail.
        """
        requirement = TASK_PROFILES.get(task_name)
        if not requirement:
            logger.warning("Unknown task '%s', using standard tier", task_name)
            requirement = TaskRequirement(
                task_name=task_name,
                complexity=TaskComplexity.MEDIUM,
                min_tier=ModelTier.LIGHTWEIGHT,
                preferred_tier=ModelTier.STANDARD,
                description="Unknown task",
            )

        # Find best model matching requirements
        decision = self._select_model(requirement)
        self._routing_log.append(decision)

        provider = self._providers.get(decision.selected_model)
        if not provider:
            logger.warning("Selected model '%s' not available, falling back to rule_based", decision.selected_model)
            provider = self._providers["rule_based"]
            decision = RoutingDecision(
                task_name=task_name,
                selected_model="rule_based",
                reason=f"Fallback: {decision.selected_model} unavailable",
                tier_used=ModelTier.RULE_BASED,
                estimated_cost_usd=0,
                estimated_latency_ms=10,
            )

        return provider, decision

    def _select_model(self, requirement: TaskRequirement) -> RoutingDecision:
        """Select the best model for a task based on constraints."""
        candidates = []
        
        for model_id, config in self._models.items():
            # Must meet minimum tier
            if not self._meets_tier(config.tier, requirement.min_tier):
                continue
            
            # Must support Chinese if required
            if requirement.requires_chinese and not config.supports_chinese:
                continue
            
            # Must fit within budget
            estimated_cost = self._estimate_cost(config, requirement)
            if estimated_cost > requirement.max_cost_usd:
                continue
            if estimated_cost > self._budget_remaining:
                continue
            
            # Must fit within latency
            if config.latency_ms_estimate > requirement.max_latency_ms:
                continue

            # Score: prefer speed or prefer quality
            score = self._score_model(config, requirement, estimated_cost)
            candidates.append((model_id, config, estimated_cost, score))

        if not candidates:
            # Fallback to rule-based
            return RoutingDecision(
                task_name=requirement.task_name,
                selected_model="rule_based",
                reason="No model meets all constraints, using rule-based fallback",
                tier_used=ModelTier.RULE_BASED,
                estimated_cost_usd=0,
                estimated_latency_ms=10,
            )

        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[3], reverse=True)
        best_id, best_config, best_cost, _ = candidates[0]

        # Deduct from budget
        self._budget_remaining -= best_cost

        reason_parts = [
            f"complexity={requirement.complexity.value}",
            f"tier={best_config.tier.value}",
        ]
        if self._prefer_speed:
            reason_parts.append("speed-optimized")

        return RoutingDecision(
            task_name=requirement.task_name,
            selected_model=best_id,
            reason=f"Best match: {', '.join(reason_parts)}",
            tier_used=best_config.tier,
            estimated_cost_usd=round(best_cost, 6),
            estimated_latency_ms=best_config.latency_ms_estimate,
        )

    def _meets_tier(self, actual: ModelTier, minimum: ModelTier) -> bool:
        """Check if actual tier meets minimum requirement."""
        tier_order = {
            ModelTier.RULE_BASED: 0,
            ModelTier.LIGHTWEIGHT: 1,
            ModelTier.STANDARD: 2,
            ModelTier.PREMIUM: 3,
        }
        return tier_order[actual] >= tier_order[minimum]

    def _estimate_cost(self, config: ModelConfig, requirement: TaskRequirement) -> float:
        """Estimate cost for a task with this model."""
        # Rough estimate: typical persona analysis ~2K input + 1K output tokens
        input_tokens = 2000
        output_tokens = 1000
        if requirement.complexity in (TaskComplexity.HIGH, TaskComplexity.MEDIUM):
            input_tokens = 4000
            output_tokens = 2000
        
        cost = (input_tokens / 1000 * config.cost_per_1k_input +
                output_tokens / 1000 * config.cost_per_1k_output)
        return cost

    def _score_model(self, config: ModelConfig, requirement: TaskRequirement, cost: float) -> float:
        """Score a model for a task. Higher is better."""
        tier_score = {
            ModelTier.RULE_BASED: 1.0,
            ModelTier.LIGHTWEIGHT: 2.0,
            ModelTier.STANDARD: 3.0,
            ModelTier.PREMIUM: 4.0,
        }
        
        quality_score = tier_score[config.tier]
        
        # Bonus for matching preferred tier exactly
        if config.tier == requirement.preferred_tier:
            quality_score += 1.0
        
        # Cost efficiency
        cost_score = 1.0 - (cost / max(requirement.max_cost_usd, 0.01))
        
        # Speed score
        speed_score = 1.0 - (config.latency_ms_estimate / max(requirement.max_latency_ms, 100))
        
        if self._prefer_speed:
            return quality_score * 0.3 + cost_score * 0.3 + speed_score * 0.4
        else:
            return quality_score * 0.5 + cost_score * 0.3 + speed_score * 0.2

    def get_routing_log(self) -> List[dict]:
        """Return routing decisions as serializable dicts."""
        return [
            {
                "task": d.task_name,
                "model": d.selected_model,
                "tier": d.tier_used.value,
                "cost_usd": d.estimated_cost_usd,
                "latency_ms": d.estimated_latency_ms,
                "reason": d.reason,
            }
            for d in self._routing_log
        ]

    def get_budget_summary(self) -> dict:
        return {
            "total_budget_usd": self._budget_total,
            "remaining_usd": round(self._budget_remaining, 6),
            "spent_usd": round(self._budget_total - self._budget_remaining, 6),
            "models_available": list(self._providers.keys()),
        }

    def add_model(self, model_id: str, config: ModelConfig, provider: Optional[LLMProvider] = None) -> None:
        """Register a custom model (for users who want to bring their own)."""
        self._models[model_id] = config
        if provider:
            self._providers[model_id] = provider
        elif config.api_key_env:
            api_key = os.environ.get(config.api_key_env, "")
            if api_key:
                self._providers[model_id] = OpenAICompatibleProvider(config)


# ─── LLM-Enhanced Analysis Nodes ───────────────────────────────────────────────

class LLMEnhancedAnalyzer:
    """Wraps analysis logic with LLM enhancement when available.
    
    For tasks like personality modeling, the LLM can provide deeper insights
    than pure rule-based analysis. This class provides the bridge.
    
    Usage:
        router = ModelRouter()
        analyzer = LLMEnhancedAnalyzer(router)
        
        # Enhanced personality analysis
        result = analyzer.analyze_personality(chat_data, behavior_patterns)
    """

    SYSTEM_PROMPTS = {
        "personality_modeling": (
            "你是一个专业的人格分析师。基于提供的聊天数据行为模式，构建5层人格模型。\n"
            "输出格式必须是JSON，包含：core_traits, identity, expression, decision_framework, interpersonal\n"
            "每个层级需要标注置信度(0-1)和数据支撑。"
        ),
        "writing_style_analysis": (
            "你是一个语言风格分析师。分析聊天消息的写作风格特征。\n"
            "输出格式必须是JSON，包含：sentence_patterns, punctuation_style, tone_markers, signature_phrases"
        ),
        "visual_persona_generation": (
            "你是一个数字分身设计师。基于人格模型和写作风格，生成完整的数字分身描述。\n"
            "输出应该是结构化的人物画像，包含：summary, personality_layers, expression_style, social_profile"
        ),
        "distillation_report": (
            "你是一个数据分析专家。生成详细的蒸馏数据报告。\n"
            "报告需要包含：数据统计、行为模式、性格分析、社交关系、置信度评估"
        ),
    }

    def __init__(self, router: ModelRouter):
        self.router = router

    def analyze_personality(self, chat_data: dict, behavior_patterns: dict) -> dict:
        """LLM-enhanced personality analysis."""
        provider, decision = self.router.get_provider("personality_modeling")
        
        prompt = self._build_personality_prompt(chat_data, behavior_patterns)
        result = provider.generate_json(prompt, self.SYSTEM_PROMPTS["personality_modeling"])
        
        result["_routing"] = {
            "model": decision.selected_model,
            "tier": decision.tier_used.value,
            "cost_usd": decision.estimated_cost_usd,
        }
        return result

    def analyze_writing_style(self, messages: list) -> dict:
        """LLM-enhanced writing style analysis."""
        provider, decision = self.router.get_provider("writing_style_analysis")
        
        # Sample messages for analysis (avoid token limits)
        sample = messages[:50] if len(messages) > 50 else messages
        prompt = "分析以下聊天消息的写作风格：\n\n"
        for msg in sample:
            prompt += f"[{msg.get('sender', '?')}]: {msg.get('text', '')}\n"
        
        result = provider.generate_json(prompt, self.SYSTEM_PROMPTS["writing_style_analysis"])
        
        result["_routing"] = {
            "model": decision.selected_model,
            "tier": decision.tier_used.value,
        }
        return result

    def generate_persona_description(self, personality: dict, writing_style: dict, behavior: dict) -> dict:
        """LLM-enhanced persona description generation."""
        provider, decision = self.router.get_provider("visual_persona_generation")
        
        prompt = (
            f"基于以下分析结果，生成完整的数字分身描述：\n\n"
            f"## 人格模型\n{json.dumps(personality, ensure_ascii=False, indent=2)}\n\n"
            f"## 写作风格\n{json.dumps(writing_style, ensure_ascii=False, indent=2)}\n\n"
            f"## 行为模式\n{json.dumps(behavior, ensure_ascii=False, indent=2)}"
        )
        
        result = provider.generate_json(prompt, self.SYSTEM_PROMPTS["visual_persona_generation"])
        
        result["_routing"] = {
            "model": decision.selected_model,
            "tier": decision.tier_used.value,
        }
        return result

    def generate_distillation_report(self, all_data: dict) -> str:
        """LLM-enhanced distillation report generation."""
        provider, decision = self.router.get_provider("distillation_report")
        
        prompt = (
            f"基于以下完整分析数据，生成详细的蒸馏报告：\n\n"
            f"{json.dumps(all_data, ensure_ascii=False, indent=2)}"
        )
        
        result = provider.generate(prompt, self.SYSTEM_PROMPTS["distillation_report"])
        return result

    def _build_personality_prompt(self, chat_data: dict, behavior_patterns: dict) -> str:
        target_count = chat_data.get("target_count", 0)
        person_name = chat_data.get("person_name", "目标人物")
        
        prompt = f"## 分析目标\n人物：{person_name}\n消息数量：{target_count}条\n\n"
        prompt += "## 行为模式数据\n"
        
        for pattern in behavior_patterns.get("patterns", []):
            prompt += f"### {pattern.get('type', '未知')}\n"
            prompt += json.dumps(pattern, ensure_ascii=False, indent=2) + "\n\n"
        
        prompt += (
            "\n## 任务\n"
            "请构建5层人格模型，每层包含具体特征、置信度和数据支撑。\n"
            "特别关注：核心性格、表达风格、决策框架、人际行为模式。"
        )
        return prompt


# ─── Convenience Function ───────────────────────────────────────────────────────

def create_router(
    budget_usd: float = 1.0,
    prefer_speed: bool = False,
    custom_models: Optional[Dict[str, ModelConfig]] = None,
) -> ModelRouter:
    """Create a pre-configured model router."""
    return ModelRouter(
        models=custom_models,
        budget_per_run_usd=budget_usd,
        prefer_speed=prefer_speed,
    )


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    
    router = create_router(budget_usd=0.50)
    
    print("=== Model Router Demo ===")
    print(f"\nBudget: {router.get_budget_summary()}")
    
    # Route each task
    for task_name in TASK_PROFILES:
        provider, decision = router.get_provider(task_name)
        print(f"\nTask: {task_name}")
        print(f"  → Model: {decision.selected_model} ({decision.tier_used.value})")
        print(f"  → Cost: ${decision.estimated_cost_usd:.6f}")
        print(f"  → Latency: ~{decision.estimated_latency_ms}ms")
        print(f"  → Reason: {decision.reason}")
    
    print(f"\n=== Budget After Routing ===")
    print(json.dumps(router.get_budget_summary(), indent=2))
    
    print(f"\n=== Routing Log ===")
    for entry in router.get_routing_log():
        print(f"  {entry['task']}: {entry['model']} ({entry['tier']})")
