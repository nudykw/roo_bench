"""Benchmark result data class for formatting and display."""

from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum
from i18n import get_text


# --- ENUMS ---

class Capability(str, Enum):
    """Enum for model capabilities."""
    VISION = "vision"
    TOOLS = "tools"
    THINKING = "thinking"
    AUDIO = "audio"


# --- MODEL INFO ---

class ModelInfo(BaseModel):
    """Інформація про модель для бенчмарку."""

    name: str
    size_gb: float = Field(..., description="Розмір моделі в GB (обов'язкове поле).")
    params: str = "N/A"
    quant: str = "N/A"
    architecture: str = "N/A"
    max_ctx: int = 131072
    moe: Optional[dict] = None
    vision: Capability = Capability.VISION
    tools: Capability = Capability.TOOLS
    thinking: Capability = Capability.THINKING
    audio: Capability = Capability.AUDIO

    # Властивості для форматування (повертають емоджі для UI)
    @property
    def vision_str(self) -> str:
        """Повертає '✅' якщо True, '❌' якщо False."""
        return "✅" if self.vision == Capability.VISION else "❌"

    @property
    def tools_str(self) -> str:
        """Повертає '✅' якщо True, '❌' якщо False."""
        return "✅" if self.tools == Capability.TOOLS else "❌"

    @property
    def thinking_str(self) -> str:
        """Повертає '✅' якщо True, '❌' якщо False."""
        return "✅" if self.thinking == Capability.THINKING else "❌"

    @property
    def audio_str(self) -> str:
        """Повертає '✅' якщо True, '❌' якщо False."""
        return "✅" if self.audio == Capability.AUDIO else "❌"

    @property
    def size_gb_str(self) -> str:
        """Форматує розмір для відображення."""
        return f"{self.size_gb:.1f} GB"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"ModelInfo(name={self.name}, size_gb={self.size_gb}, "
            f"params={self.params}, vision={self.vision})"
        )


# --- BENCHMARK METRICS ---

class BenchmarkMetrics(BaseModel):
    """Метрики одного бенчмарк-тесту."""

    ctx: int
    temperature: float
    avg_tps: float
    min_tps: float
    max_tps: float
    std_dev: float
    vram: Optional[int] = None
    prompt_id: Optional[str] = None
    prompt_name: Optional[str] = None
    duration_sec: float = 0.0
    prompt_tokens: int = 0
    response_tokens: int = 0
    mode: Optional[str] = None
    chain_id: Optional[str] = None
    chain_name: Optional[str] = None
    expert_score: Optional[float] = None
    response: Optional[str] = None

    # Властивості для форматування
    @property
    def ctx_str(self) -> str:
        """Get formatted context size string."""
        if self.ctx >= 1024:
            return f"{self.ctx // 1024}K"
        return str(self.ctx)

    @property
    def vram_str(self) -> str:
        """Get formatted VRAM string."""
        if self.vram is None:
            return "N/A"
        return f"{self.vram / 1024 / 1024:.1f} MiB"

    def to_summary_line(self) -> str:
        """Format result as summary line for console output.

        Returns:
            Formatted string with benchmark metrics.
        """
        return (
            f"{get_text('context')}: {self.ctx_str} | "
            f"{get_text('temperature')}: {self.temperature:.1f} | "
            f"{get_text('duration')}: {self.duration_sec:.2f}s | "
            f"{get_text('prompt_tokens')}: {self.prompt_tokens} | "
            f"{get_text('response_tokens')}: {self.response_tokens} | "
            f"{get_text('avg_tps')}: {self.avg_tps:.2f} | "
            f"{get_text('min_tps')}: {self.min_tps:.2f} | "
            f"{get_text('max_tps')}: {self.max_tps:.2f} | "
            f"{get_text('std_dev')}: {self.std_dev:.2f} | "
            f"VRAM: {self.vram_str}"
        )

    def to_recommendation_line(self, rank: int = 1) -> str:
        """Format result as recommendation line.

        Args:
            rank: Recommendation rank (1 = ★, 2+ = indented)

        Returns:
            Formatted string like:
            "★ dev-qwen3.5-9b:latest (9.7B, 6.1 GB)"
            "    Варіант 1: Context = 256K | Температура = 1.0 | Швидкість: 42.5 t/s"
        """
        if rank == 1:
            prefix = "★ "
        else:
            prefix = "    "

        return (
            f"{prefix}{get_text('variant', i=rank, ctx=self.ctx_str, tps=self.avg_tps)} | "
            f"{get_text('temperature')}: {self.temperature:.1f}"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export.

        Returns:
            Dictionary with all result fields.
        """
        return {
            'ctx': self.ctx,
            'ctx_str': self.ctx_str,
            'temperature': round(self.temperature, 3),
            'avg_tps': round(self.avg_tps, 3),
            'min_tps': round(self.min_tps, 3),
            'max_tps': round(self.max_tps, 3),
            'std_dev': round(self.std_dev, 3),
            'vram': self.vram,
            'vram_str': self.vram_str,
            'prompt_id': self.prompt_id,
            'prompt_name': self.prompt_name,
            'duration_sec': round(self.duration_sec, 3),
            'prompt_tokens': self.prompt_tokens,
            'response_tokens': self.response_tokens,
            'mode': self.mode,
            'chain_id': self.chain_id,
            'chain_name': self.chain_name,
            'expert_score': round(self.expert_score, 1) if self.expert_score is not None else None,
            'response': self.response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BenchmarkMetrics':
        """Create BenchmarkMetrics from dictionary.

        Args:
            data: Dictionary with result fields.

        Returns:
            BenchmarkMetrics instance.
        """
        return cls(**data)

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"BenchmarkMetrics(ctx={self.ctx}, temp={self.temperature}, "
            f"avg_tps={self.avg_tps:.2f})"
        )


# --- BENCHMARK RESULT ---

class BenchmarkResult(BaseModel):
    """Результати бенчмарку для моделі."""

    model: ModelInfo
    results: List[BenchmarkMetrics] = []

    # Властивості зручності
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.model.name

    @property
    def all_contexts(self) -> List[int]:
        """Get all context sizes tested."""
        return [r.ctx for r in self.results]

    @property
    def all_temperatures(self) -> List[float]:
        """Get all temperatures tested."""
        return [r.temperature for r in self.results]

    # Методи фільтрації
    def filter_by_context(self, ctx: int) -> List[BenchmarkMetrics]:
        """Filter results by context size."""
        return [r for r in self.results if r.ctx == ctx]

    def filter_by_temperature(self, temp: float) -> List[BenchmarkMetrics]:
        """Filter results by temperature."""
        return [r for r in self.results if r.temperature == temp]

    def filter_by_mode(self, mode: str) -> List[BenchmarkMetrics]:
        """Filter results by mode."""
        return [r for r in self.results if r.mode == mode]

    # Методи агрегації
    def get_best_result(self) -> Optional[BenchmarkMetrics]:
        """Get the result with the highest average TPS."""
        return max(self.results, key=lambda r: r.avg_tps) if self.results else None

    def get_average_tps(self) -> float:
        """Calculate overall average TPS across all runs."""
        total_tps = sum(r.avg_tps for r in self.results)
        return total_tps / len(self.results) if self.results else 0.0

    # Методи трансформації
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export.

        Returns:
            Dictionary with nested model and results structure.
        """
        return {
            'model': self.model.model_dump(),
            'results': [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BenchmarkResult':
        """Create BenchmarkResult from dictionary.

        Args:
            data: Dictionary with model and results.

        Returns:
            BenchmarkResult instance.
        """
        model_data = data.get('model', {})
        results_data = data.get('results', [])

        model = ModelInfo(**model_data)
        metrics_list = [BenchmarkMetrics(**m) for m in results_data]

        return cls(model=model, results=metrics_list)

    # Методи форматування (для сумісності)
    def to_summary_lines(self) -> List[str]:
        """Generate summary lines for all results."""
        return [r.to_summary_line() for r in self.results]

    def to_recommendation_lines(self, rank: int = 1) -> List[str]:
        """Generate recommendation lines for the top result."""
        best_result = self.get_best_result()
        if best_result:
            return [best_result.to_recommendation_line(rank)]
        return []

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"BenchmarkResult(model={self.model_name}, "
            f"results_count={len(self.results)}, "
            f"avg_tps={self.get_average_tps():.2f})"
        )
