"""
training_jobs.py – simple in-memory job registry.

Each training run gets a unique job_id.  The job object records the current
status, per-epoch metrics, and (once finished) a final summary.
"""

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EpochMetric:
    model: str
    epoch: int
    loss: float
    mae: float
    val_loss: Optional[float]
    val_mae: Optional[float]


@dataclass
class TrainingJob:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    config: dict = field(default_factory=dict)
    metrics: List[EpochMetric] = field(default_factory=list)
    summary: Optional[dict] = None
    error: Optional[str] = None
    # subscribers waiting for new metric events (used by SSE)
    _subscribers: List = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_metric(self, metric: EpochMetric) -> None:
        with self._lock:
            self.metrics.append(metric)
            for queue in list(self._subscribers):
                try:
                    queue.put_nowait(metric)
                except Exception:
                    pass

    def subscribe(self, queue) -> None:
        with self._lock:
            self._subscribers.append(queue)

    def unsubscribe(self, queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass

    def finish(self, summary: dict) -> None:
        with self._lock:
            self.status = JobStatus.COMPLETED
            self.summary = summary
            for queue in list(self._subscribers):
                try:
                    queue.put_nowait(None)  # sentinel
                except Exception:
                    pass

    def fail(self, error: str) -> None:
        with self._lock:
            self.status = JobStatus.FAILED
            self.error = error
            for queue in list(self._subscribers):
                try:
                    queue.put_nowait(None)
                except Exception:
                    pass

    def cancel(self) -> None:
        with self._lock:
            if self.status == JobStatus.RUNNING:
                self.status = JobStatus.CANCELLED
                for queue in list(self._subscribers):
                    try:
                        queue.put_nowait(None)
                    except Exception:
                        pass


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

_registry: Dict[str, TrainingJob] = {}
_registry_lock = threading.Lock()


def create_job(config: dict) -> TrainingJob:
    job_id = f"train_{uuid.uuid4().hex[:12]}"
    job = TrainingJob(job_id=job_id, config=config)
    with _registry_lock:
        _registry[job_id] = job
    return job


def get_job(job_id: str) -> Optional[TrainingJob]:
    with _registry_lock:
        return _registry.get(job_id)


def list_jobs() -> List[TrainingJob]:
    with _registry_lock:
        return list(_registry.values())
