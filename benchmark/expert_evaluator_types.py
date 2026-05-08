"""Expert evaluator data types and structures."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, TYPE_CHECKING

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
    mode: Optional[str]
    chain_id: Optional[str]
    chain_name: Optional[str]
    response: str
    avg_tps: float

    metrics_ref: Optional['BenchmarkMetrics'] = None

    chain_context: Dict[str, str] = field(default_factory=dict)
