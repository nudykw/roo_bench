"""Benchmark result data class for formatting and display."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

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
    moe: dict[str, Any] | None = None
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
    vram: int | None = None
    prompt_id: str | None = None
    prompt_name: str | None = None
    duration_sec: float = 0.0
    prompt_tokens: int = 0
    response_tokens: int = 0
    mode: str | None = None
    chain_id: str | None = None
    chain_name: str | None = None
    expert_score: float | None = None
    response: str | None = None
    
    # Расширенная статистика ресурсов
    cpu_stats: dict[str, Any] | None = None
    ram_stats: dict[str, Any] | None = None
    vram_stats: dict[str, Any] | None = None
    
    # Агрегированные метрики ресурсов
    avg_cpu_percent: float | None = None
    max_cpu_percent: float | None = None
    avg_ram_percent: float | None = None
    max_ram_percent: float | None = None
    avg_vram_percent: float | None = None
    max_vram_percent: float | None = None

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
        lines = [
            f"{get_text('context')}: {self.ctx_str}",
            f"{get_text('temperature')}: {self.temperature:.1f}",
            f"{get_text('duration')}: {self.duration_sec:.2f}s",
            f"{get_text('prompt_tokens')}: {self.prompt_tokens}",
            f"{get_text('response_tokens')}: {self.response_tokens}",
            f"{get_text('avg_tps')}: {self.avg_tps:.2f}",
            f"{get_text('min_tps')}: {self.min_tps:.2f}",
            f"{get_text('max_tps')}: {self.max_tps:.2f}",
            f"{get_text('std_dev')}: {self.std_dev:.2f}",
            f"VRAM: {self.vram_str}"
        ]
        
        # Добавляем информацию о ресурсах, если она доступна
        if self.avg_cpu_percent is not None:
            lines.append(f"CPU: {self.avg_cpu_percent:.1f}% (max: {self.max_cpu_percent:.1f}%)")
        if self.avg_ram_percent is not None:
            lines.append(f"RAM: {self.avg_ram_percent:.1f}% (max: {self.max_ram_percent:.1f}%)")
        if self.avg_vram_percent is not None:
            lines.append(f"VRAM: {self.avg_vram_percent:.1f}% (max: {self.max_vram_percent:.1f}%)")
        
        return " | ".join(lines)

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
            f"{prefix}{get_text('variant', i=str(rank), ctx=self.ctx_str, tps=str(self.avg_tps))} | "
            f"{get_text('temperature')}: {self.temperature:.1f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export.

        Returns:
            Dictionary with all result fields.
        """
        result: dict[str, Any] = {
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
                        # NOTE: 'response' is intentionally excluded from JSON export to keep file size small.
                        # Responses are stored in export/expert_results.md instead.
                    }
        
        # Добавляем расширенную статистику ресурсов
        if self.cpu_stats is not None:
            result['cpu_stats'] = self.cpu_stats
        if self.ram_stats is not None:
            result['ram_stats'] = self.ram_stats
        if self.vram_stats is not None:
            result['vram_stats'] = self.vram_stats
            
        # Добавляем агрегированные метрики ресурсов
        if self.avg_cpu_percent is not None:
            result['avg_cpu_percent'] = round(self.avg_cpu_percent, 3)
        if self.max_cpu_percent is not None:
            result['max_cpu_percent'] = round(self.max_cpu_percent, 3)
        if self.avg_ram_percent is not None:
            result['avg_ram_percent'] = round(self.avg_ram_percent, 3)
        if self.max_ram_percent is not None:
            result['max_ram_percent'] = round(self.max_ram_percent, 3)
        if self.avg_vram_percent is not None:
            result['avg_vram_percent'] = round(self.avg_vram_percent, 3)
        if self.max_vram_percent is not None:
            result['max_vram_percent'] = round(self.max_vram_percent, 3)
            
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'BenchmarkMetrics':
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
    results: list[BenchmarkMetrics] = []

    # Властивості зручності
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.model.name

    @property
    def all_contexts(self) -> list[int]:
        """Get all context sizes tested."""
        return [r.ctx for r in self.results]

    @property
    def all_temperatures(self) -> list[float]:
        """Get all temperatures tested."""
        return [r.temperature for r in self.results]

    # Методи фільтрації
    def filter_by_context(self, ctx: int) -> list[BenchmarkMetrics]:
        """Filter results by context size."""
        return [r for r in self.results if r.ctx == ctx]

    def filter_by_temperature(self, temp: float) -> list[BenchmarkMetrics]:
        """Filter results by temperature."""
        return [r for r in self.results if r.temperature == temp]

    def filter_by_mode(self, mode: str) -> list[BenchmarkMetrics]:
        """Filter results by mode."""
        return [r for r in self.results if r.mode == mode]

    # Методи агрегації
    def get_best_result(self) -> BenchmarkMetrics | None:
        """Get the result with the highest average TPS."""
        return max(self.results, key=lambda r: r.avg_tps) if self.results else None

    def get_average_tps(self) -> float:
        """Calculate overall average TPS across all runs."""
        total_tps = sum(r.avg_tps for r in self.results)
        return total_tps / len(self.results) if self.results else 0.0

    # Методи трансформації
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export.

        Returns:
            Dictionary with nested model and results structure.
        """
        return {
            'model': self.model.model_dump(),
            'results': [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'BenchmarkResult':
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
    def to_summary_lines(self) -> list[str]:
        """Generate summary lines for all results."""
        return [r.to_summary_line() for r in self.results]

    def to_recommendation_lines(self, rank: int = 1) -> list[str]:
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
