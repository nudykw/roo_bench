"""Benchmark result data class for formatting and display."""

from typing import Optional
from i18n import get_text


class BenchmarkResult:
    """Represents a single benchmark result with formatting methods."""
    
    def __init__(
        self,
        model_name: str,
        ctx: int,
        temperature: float,
        avg_tps: float,
        min_tps: float,
        max_tps: float,
        std_dev: float,
        vram: Optional[int] = None,
        prompt_id: Optional[str] = None,
        prompt_name: Optional[str] = None,
        duration_sec: float = 0,
        prompt_tokens: int = 0,
        response_tokens: int = 0
    ):
        """Initialize benchmark result.
        
        Args:
            model_name: Model name
            ctx: Context size
            temperature: Temperature value
            avg_tps: Average tokens per second
            min_tps: Minimum TPS
            max_tps: Maximum TPS
            std_dev: Standard deviation
            vram: VRAM usage in bytes (optional)
            prompt_id: Prompt ID (optional)
            prompt_name: Prompt name (optional)
            duration_sec: Duration in seconds
            prompt_tokens: Number of prompt tokens
            response_tokens: Number of response tokens
        """
        self.model_name = model_name
        self.ctx = ctx
        self.temperature = temperature
        self.avg_tps = avg_tps
        self.min_tps = min_tps
        self.max_tps = max_tps
        self.std_dev = std_dev
        self.vram = vram
        self.prompt_id = prompt_id
        self.prompt_name = prompt_name
        self.duration_sec = duration_sec
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
    
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
            Formatted string like:
            "Контекст: 16K | Температура: 0.7 | Час: 1.68s | Вхідні: 97 | Вихідні: 100 | Середнє TPS: 48.47 | Мін: 8.56 | Макс: 68.75 | СтД: 2.15 | VRAM: 10142.0 MiB"
        """
        # Use i18n for localized output
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
            Dictionary with all result fields
        """
        return {
            'model_name': self.model_name,
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
            'response_tokens': self.response_tokens
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BenchmarkResult':
        """Create BenchmarkResult from dictionary.
        
        Args:
            data: Dictionary with result fields
        
        Returns:
            BenchmarkResult instance
        """
        return cls(
            model_name=data.get('model_name', 'unknown'),
            ctx=data.get('ctx', 0),
            temperature=data.get('temperature', 0.0),
            avg_tps=data.get('avg_tps', 0.0),
            min_tps=data.get('min_tps', 0.0),
            max_tps=data.get('max_tps', 0.0),
            std_dev=data.get('std_dev', 0.0),
            vram=data.get('vram'),
            prompt_id=data.get('prompt_id'),
            prompt_name=data.get('prompt_name'),
            duration_sec=data.get('duration_sec', 0.0),
            prompt_tokens=data.get('prompt_tokens', 0),
            response_tokens=data.get('response_tokens', 0)
        )
    
    def __str__(self) -> str:
        """String representation."""
        return self.to_summary_line()
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"BenchmarkResult(model={self.model_name}, ctx={self.ctx}, "
            f"temp={self.temperature}, avg_tps={self.avg_tps:.2f})"
        )
