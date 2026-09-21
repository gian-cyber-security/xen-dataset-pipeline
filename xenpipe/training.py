from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from .pipeline import iter_xen_samples


def stream_to_trainer(callback: Callable[[dict[str, Any]], None], dataset_id: str, *, target: str, **kwargs: Any) -> int:
    """Push XEN samples directly into a trainer callback.

    No permanent dataset file is created by this function. The trainer owns
    the sample immediately and can convert the in-memory sample to tensors.
    """
    count = 0
    for sample in iter_xen_samples(dataset_id, target=target, **kwargs):
        callback(sample)
        count += 1
    return count


def iter_training_samples(dataset_id: str, *, target: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    """Public iterator intended to be consumed by XEN model trainers."""
    yield from iter_xen_samples(dataset_id, target=target, **kwargs)
