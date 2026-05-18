"""Expert evaluator data types and structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.result import BenchmarkMetrics


@dataclass
class ExpertEvaluationEntry:
    """Stores context needed for expert evaluation of a single response."""

    model_name: str
    ctx: int
    temperature: float
    prompt_id: str
    prompt_name: str
    mode: str | None
    chain_id: str | None
    chain_name: str | None
    response: str
    avg_tps: float

    metrics_ref: BenchmarkMetrics | None = None

    chain_context: dict[str, str] = field(default_factory=dict)
