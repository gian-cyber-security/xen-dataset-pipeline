from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
import os

from .normalize import resolve_video
from .pipeline import iter_xen_samples


def _iter_video_prefetched(samples: Iterator[dict[str, Any]], dataset_id: str, *,
                           revision: str | None, cache_dir: str | None,
                           workers: int, prefetch: int) -> Iterator[dict[str, Any]]:
    """Resolve remote videos concurrently with a bounded in-flight window."""
    workers=max(1, int(workers))
    prefetch=max(workers, int(prefetch))
    pending: deque[tuple[dict[str, Any], Future[str]]] = deque()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="xen-video") as pool:
        source=iter(samples)

        def fill():
            while len(pending) < prefetch:
                try:
                    sample=next(source)
                except StopIteration:
                    return
                sample_revision = (
                    revision
                    or sample.get("provenance", {}).get("revision")
                )
                pending.append((sample,pool.submit(
                    resolve_video,
                    sample["video_ref"],
                    dataset_id,
                    sample_revision,
                    cache_dir,
                )))

        fill()
        while pending:
            sample,future=pending.popleft()
            try:
                sample["video"]=future.result()
            except Exception:
                fill()
                continue
            sample.pop("video_ref",None)
            yield sample
            fill()


def stream_to_trainer(callback: Callable[[dict[str, Any]], None], dataset_id: str, *, target: str, **kwargs: Any) -> int:
    """Push XEN samples directly into a trainer callback.

    GEN1-V uses bounded background prefetch. Videos are downloaded only when
    needed and kept in the temporary XEN cache; no processed corpus is built.
    """
    if target == "gen1-v":
        workers=int(kwargs.pop("prefetch_workers",os.getenv("XEN_PREFETCH_WORKERS","4")))
        prefetch=int(kwargs.pop("prefetch_size",os.getenv("XEN_PREFETCH_SIZE","8")))
        revision=kwargs.get("revision")
        cache_dir=kwargs.get("cache_dir")
        samples=iter_xen_samples(dataset_id,target=target,resolve_media=False,**kwargs)
        iterator=_iter_video_prefetched(samples,dataset_id,revision=revision,cache_dir=cache_dir,workers=workers,prefetch=prefetch)
    else:
        iterator=iter_xen_samples(dataset_id,target=target,**kwargs)

    count=0
    for sample in iterator:
        callback(sample)
        count+=1
    return count


def iter_training_samples(dataset_id: str, *, target: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    """Yield samples ready for XEN trainers with optimized GEN1-V loading."""
    if target != "gen1-v":
        yield from iter_xen_samples(dataset_id,target=target,**kwargs)
        return

    workers=int(kwargs.pop("prefetch_workers",os.getenv("XEN_PREFETCH_WORKERS","4")))
    prefetch=int(kwargs.pop("prefetch_size",os.getenv("XEN_PREFETCH_SIZE","8")))
    revision=kwargs.get("revision")
    cache_dir=kwargs.get("cache_dir")
    samples=iter_xen_samples(dataset_id,target=target,resolve_media=False,**kwargs)
    yield from _iter_video_prefetched(samples,dataset_id,revision=revision,cache_dir=cache_dir,workers=workers,prefetch=prefetch)
