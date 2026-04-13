"""Fono mokymo job'ų būsena (in-memory), skirta UI progresui be naršyklės užšalimo."""

from __future__ import annotations

import threading
import uuid
from typing import Any, Callable


_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def create_job() -> str:
    job_id = uuid.uuid4().hex[:16]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "",
            "result": None,
            "error": None,
        }
    return job_id


def update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        row = _jobs.get(job_id)
        return dict(row) if row else None


def run_in_thread(target: Callable[..., None], args: tuple[Any, ...]) -> None:
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
