"""
AwakeEngine Benchmark — 延迟与性能测试
========================================

测试各组件的实际延迟，帮助用户评估硬件是否满足实时对话需求。

用法：
  # 完整benchmark
  python3 scripts/benchmark.py

  # 只测试TTS
  python3 scripts/benchmark.py --tts-only

  # 只测试人格评估
  python3 scripts/benchmark.py --persona-only

  # 自定义迭代次数
  python3 scripts/benchmark.py --iterations 50

作者：AtomCollide-智械工坊
"""

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

# ── 延迟数据结构 ─────────────────────────────────────────────────────────────

@dataclass
class LatencyResult:
    """单个延迟测试结果"""
    component: str
    operation: str
    times_ms: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.times_ms)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0

    @property
    def median(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0

    @property
    def p95(self) -> float:
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99(self) -> float:
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def std_dev(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "operation": self.operation,
            "iterations": self.count,
            "mean_ms": round(self.mean, 2),
            "median_ms": round(self.median, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "std_dev_ms": round(self.std_dev, 2),
            "min_ms": round(min(self.times_ms), 2) if self.times_ms else 0,
            "max_ms": round(max(self.times_ms), 2) if self.times_ms else 0,
        }


@dataclass
class BenchmarkReport:
    """完整benchmark报告"""
    results: List[LatencyResult] = field(default_factory=list)
    system_info: dict = field(default_factory=dict)
    timestamp: str = ""

    def add(self, result: LatencyResult):
        self.results.append(result)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "system": self.system_info,
            "results": [r.to_dict() for r in self.results],
            "summary": self._summary(),
        }

    def _summary(self) -> dict:
        if not self.results:
            return {"status": "no data"}

        # Check if we can do real-time conversation
        # Real-time = total latency < 200ms for TTS + avatar
        tts_results = [r for r in self.results if r.component == "TTS"]
        avatar_results = [r for r in self.results if r.component in ("Avatar", "D-ID")]

        tts_latency = sum(r.mean for r in tts_results) if tts_results else 0
        avatar_latency = sum(r.mean for r in avatar_results) if avatar_results else 0
        total_pipeline = tts_latency + avatar_latency

        realtime_capable = total_pipeline < 200
        quality_threshold = total_pipeline < 500

        return {
            "tts_latency_ms": round(tts_latency, 2),
            "avatar_latency_ms": round(avatar_latency, 2),
            "total_pipeline_ms": round(total_pipeline, 2),
            "realtime_capable": realtime_capable,
            "quality_threshold": quality_threshold,
            "verdict": (
                "✅ 实时对话就绪" if realtime_capable else
                "⚠️ 可用但有延迟感" if quality_threshold else
                "❌ 延迟过高，建议使用云端模式"
            ),
        }

    def summary_text(self) -> str:
        lines = [
            "=" * 60,
            "AwakeEngine Benchmark Report",
            "=" * 60,
            f"Time: {self.timestamp}",
            "",
        ]

        # System info
        if self.system_info:
            lines.append("System:")
            for k, v in self.system_info.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        # Results table
        lines.append(f"{'Component':<15} {'Operation':<20} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'StdDev':>8}")
        lines.append("-" * 85)
        for r in self.results:
            lines.append(
                f"{r.component:<15} {r.operation:<20} "
                f"{r.mean:>7.1f}ms {r.median:>7.1f}ms "
                f"{r.p95:>7.1f}ms {r.p99:>7.1f}ms "
                f"{r.std_dev:>7.1f}ms"
            )

        # Summary
        summary = self._summary()
        lines.append("")
        lines.append("=" * 60)
        lines.append("Summary:")
        lines.append(f"  TTS Latency:      {summary['tts_latency_ms']:.1f}ms")
        lines.append(f"  Avatar Latency:   {summary['avatar_latency_ms']:.1f}ms")
        lines.append(f"  Total Pipeline:   {summary['total_pipeline_ms']:.1f}ms")
        lines.append(f"  Verdict:          {summary['verdict']}")
        lines.append("=" * 60)

        return "\n".join(lines)


# ── 系统信息收集 ─────────────────────────────────────────────────────────────

def collect_system_info() -> dict:
    """Collect system information for benchmark context."""
    info = {
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
    }

    # GPU info
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(",")
            info["gpu"] = gpu_info[0].strip()
            info["gpu_memory_total"] = gpu_info[1].strip() if len(gpu_info) > 1 else "N/A"
            info["gpu_memory_free"] = gpu_info[2].strip() if len(gpu_info) > 2 else "N/A"
        else:
            info["gpu"] = "Not available"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["gpu"] = "Not available"

    # CPU info
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        info["cpu"] = "Unknown"

    # RAM info
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    total_kb = int(line.split()[1])
                    info["ram_gb"] = f"{total_kb / 1024 / 1024:.1f} GB"
                    break
    except FileNotFoundError:
        info["ram_gb"] = "Unknown"

    return info


# ── Benchmark 测试 ───────────────────────────────────────────────────────────

def benchmark_persona_evaluation(iterations: int = 10) -> LatencyResult:
    """Benchmark persona fidelity evaluation."""
    result = LatencyResult("Persona", "Fidelity Eval")

    sys.path.insert(0, os.path.dirname(__file__))
    from persona_fidelity import PersonaFidelityEvaluator

    evaluator = PersonaFidelityEvaluator()

    sample_messages = [
        {"sender": "测试用户", "text": "好的收到", "timestamp": "2026-05-01T10:30:00"},
        {"sender": "测试用户", "text": "这个方案可以，直接干", "timestamp": "2026-05-01T11:00:00"},
        {"sender": "测试用户", "text": "别催，正在谈，等我消息", "timestamp": "2026-05-01T14:00:00"},
        {"sender": "测试用户", "text": "兄弟们！福利来了！", "timestamp": "2026-05-01T02:30:00"},
        {"sender": "测试用户", "text": "这个死孩子，我说了多少遍", "timestamp": "2026-05-02T01:00:00"},
        {"sender": "测试用户", "text": "版权5积分，你卖给学姐学哥10块一份", "timestamp": "2026-05-02T02:00:00"},
        {"sender": "测试用户", "text": "我不会，但是我可以学", "timestamp": "2026-05-02T03:00:00"},
        {"sender": "测试用户", "text": "你是真的打算一点脑子不动纯躺", "timestamp": "2026-05-02T10:05:00"},
        {"sender": "测试用户", "text": "所有人都是24小时，你要通过AI来增加你的睡眠收入", "timestamp": "2026-05-02T02:00:00"},
        {"sender": "测试用户", "text": "脑力收入≠体力收入", "timestamp": "2026-05-03T01:00:00"},
    ]

    persona_config = {
        "catchphrases": ["机智", "死孩子", "兄弟们"],
        "decision_priority": ["变现", "影响力", "帮兄弟"],
        "signature_phrases": ["福利", "版权", "变现", "脑力收入"],
    }

    persona_outputs = [
        "兄弟们！福利来了！",
        "这个可以搞，具体方案是先对接平台方",
        "你这个死孩子，我让你通过脑力赚钱",
        "别扯了，先干",
        "版权5积分，你卖给学姐学哥10块一份",
    ]

    for _ in range(iterations):
        start = time.perf_counter()
        report = evaluator.evaluate(
            original_messages=sample_messages,
            persona_config=persona_config,
            persona_outputs=persona_outputs,
        )
        elapsed = (time.perf_counter() - start) * 1000
        result.times_ms.append(elapsed)

    return result


def benchmark_chat_import(iterations: int = 5) -> LatencyResult:
    """Benchmark chat import (simulated)."""
    result = LatencyResult("Import", "Parse Messages")

    # Simulate message parsing
    sample_data = [{"sender": f"user{i}", "text": f"测试消息{i}" * 10, "timestamp": "2026-05-01T10:00:00"} for i in range(100)]

    for _ in range(iterations):
        start = time.perf_counter()
        # Simulate parsing
        _ = [json.dumps(msg, ensure_ascii=False) for msg in sample_data]
        elapsed = (time.perf_counter() - start) * 1000
        result.times_ms.append(elapsed)

    return result


def benchmark_text_similarity(iterations: int = 100) -> LatencyResult:
    """Benchmark text similarity calculations."""
    result = LatencyResult("TextSim", "Cosine Similarity")

    sys.path.insert(0, os.path.dirname(__file__))
    from persona_fidelity import TextSimilarity

    vec_a = {"hello": 0.5, "world": 0.3, "test": 0.2}
    vec_b = {"hello": 0.4, "world": 0.4, "test": 0.2}

    for _ in range(iterations):
        start = time.perf_counter()
        _ = TextSimilarity.cosine_similarity(vec_a, vec_b)
        elapsed = (time.perf_counter() - start) * 1000
        result.times_ms.append(elapsed)

    return result


def benchmark_distribution_divergence(iterations: int = 100) -> LatencyResult:
    """Benchmark distribution divergence calculation."""
    result = LatencyResult("TextSim", "JS Divergence")

    sys.path.insert(0, os.path.dirname(__file__))
    from persona_fidelity import TextSimilarity

    dist_a = {"enthusiastic": 0.3, "critical": 0.2, "warm": 0.3, "neutral": 0.2}
    dist_b = {"enthusiastic": 0.25, "critical": 0.25, "warm": 0.25, "neutral": 0.25}

    for _ in range(iterations):
        start = time.perf_counter()
        _ = TextSimilarity.distribution_divergence(dist_a, dist_b)
        elapsed = (time.perf_counter() - start) * 1000
        result.times_ms.append(elapsed)

    return result


def benchmark_tts_simulation(iterations: int = 10) -> LatencyResult:
    """Benchmark TTS latency (simulated — replace with actual TTS call)."""
    result = LatencyResult("TTS", "Text-to-Speech")

    # This is a placeholder — in production, replace with actual TTS call
    # For now, simulate with sleep
    for _ in range(iterations):
        start = time.perf_counter()
        # Simulate TTS processing time (50-150ms)
        time.sleep(0.05 + (iterations % 10) * 0.01)
        elapsed = (time.perf_counter() - start) * 1000
        result.times_ms.append(elapsed)

    return result


# ── 主入口 ───────────────────────────────────────────────────────────────────

def run_benchmark(args) -> BenchmarkReport:
    """Run all benchmarks."""
    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        system_info=collect_system_info(),
    )

    print("🚀 Starting AwakeEngine Benchmark...")
    print(f"   Iterations: {args.iterations}")
    print()

    # Always run persona evaluation
    print("📊 Testing persona evaluation...")
    report.add(benchmark_persona_evaluation(args.iterations))

    print("📊 Testing text similarity...")
    report.add(benchmark_text_similarity(args.iterations * 10))

    print("📊 Testing distribution divergence...")
    report.add(benchmark_distribution_divergence(args.iterations * 10))

    print("📊 Testing chat import...")
    report.add(benchmark_chat_import(args.iterations))

    if not args.persona_only:
        print("📊 Testing TTS (simulated)...")
        report.add(benchmark_tts_simulation(args.iterations))

    return report


def main():
    parser = argparse.ArgumentParser(description="AwakeEngine Benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per test")
    parser.add_argument("--tts-only", action="store_true", help="Only test TTS")
    parser.add_argument("--persona-only", action="store_true", help="Only test persona evaluation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=str, help="Save results to file")
    args = parser.parse_args()

    report = run_benchmark(args)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.summary_text())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), ensure_ascii=False, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
