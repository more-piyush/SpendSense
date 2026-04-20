from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict


@dataclass
class TaskMetrics:
    total_requests: int = 0
    total_errors: int = 0
    total_feedback_events: int = 0
    total_abstentions: int = 0
    total_overrides: int = 0
    total_accepts: int = 0
    total_rejections: int = 0
    cumulative_latency_ms: float = 0.0


@dataclass
class MetricsRegistry:
    _lock: Lock = field(default_factory=Lock)
    _tasks: Dict[str, TaskMetrics] = field(default_factory=lambda: defaultdict(TaskMetrics))

    def record_prediction(self, task: str, latency_ms: float, abstained: bool) -> None:
        with self._lock:
            metrics = self._tasks[task]
            metrics.total_requests += 1
            metrics.cumulative_latency_ms += latency_ms
            if abstained:
                metrics.total_abstentions += 1

    def record_error(self, task: str) -> None:
        with self._lock:
            self._tasks[task].total_errors += 1

    def record_feedback(self, task: str, action: str) -> None:
        with self._lock:
            metrics = self._tasks[task]
            metrics.total_feedback_events += 1
            if action == "accepted":
                metrics.total_accepts += 1
            elif action == "overridden":
                metrics.total_overrides += 1
            elif action in {"rejected", "dismissed"}:
                metrics.total_rejections += 1

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            summary: Dict[str, Dict[str, float]] = {}
            for task, metrics in self._tasks.items():
                avg_latency = 0.0
                if metrics.total_requests > 0:
                    avg_latency = metrics.cumulative_latency_ms / metrics.total_requests
                summary[task] = {
                    "total_requests": metrics.total_requests,
                    "total_errors": metrics.total_errors,
                    "total_feedback_events": metrics.total_feedback_events,
                    "total_abstentions": metrics.total_abstentions,
                    "total_overrides": metrics.total_overrides,
                    "total_accepts": metrics.total_accepts,
                    "total_rejections": metrics.total_rejections,
                    "avg_latency_ms": round(avg_latency, 2),
                }
            return summary
