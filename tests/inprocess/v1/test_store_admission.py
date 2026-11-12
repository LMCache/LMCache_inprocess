# SPDX-License-Identifier: Apache-2.0
"""Reproduction + regression tests for LMCache #3634.

Under concurrent store pressure the store-side CPU allocation can fail
transiently because eviction candidates are pinned by in-flight requests.
The legacy behavior broke out of the chunk loop and silently dropped the
failed chunk *and every later chunk*. These tests drive the REAL
``cache_engine.store()`` path (via a lightweight engine harness) to show:

  * default ``store_admission_retry_ms=0``: the chunks are still dropped, but
    the drop event is now recorded in ``update_store_admission_failure_count``
    (visible).
  * ``store_admission_retry_ms>0``: a bounded retry window lets an in-flight
    request release its pin; the chunk is stored and nothing is dropped.
  * an oversized retry budget is capped, and concurrent drop events do not
    lose counts or trip the warning rate limiter more than once.

``transformers`` and ``aiofile`` are declared in ``requirements/common.txt``;
they are imported transitively by the in-process engine. No dependency stubs
or checkout paths belong in this file -- install the declared requirements
instead.
"""

# Standard
from types import SimpleNamespace
from typing import Optional
import contextlib
import threading
import time

# Third Party
import torch

# First Party
from lmcache.inprocess.v1.cache_engine import (
    STORE_ADMISSION_RETRY_MAX_MS,
    STORE_DROP_WARNING_INTERVAL_SEC,
    LMCacheEngine,
)
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.inprocess.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import AddressManager, MemoryFormat, MemoryObj


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="m",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


def _make_backend(cap: int) -> tuple[LocalCPUBackend, LMCacheMetadata, int]:
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=256, local_cpu=True, lmcache_instance_id="t"
    )
    md = _metadata()
    pa = MixedMemoryAllocator(1024 * 1024 * 1024)
    pb = LocalCPUBackend(config=cfg, memory_allocator=pa)
    probe = pb.allocate(md.get_shapes(256), md.get_dtypes(), fmt=None)
    cb = probe.get_size()
    pa.close()
    # ``cb`` already includes the allocator's alignment. Reserve one explicit
    # alignment unit per chunk (plus one for the pool boundary) instead of a
    # magic multiplier, so alignment changes cannot silently shrink the pool.
    pool_bytes = cb * cap + AddressManager.ALIGN_BYTES * (cap + 1)
    alloc = MixedMemoryAllocator(pool_bytes)
    b = LocalCPUBackend(config=cfg, memory_allocator=alloc)
    return b, md, cb


def _put(
    backend: LocalCPUBackend, md: LMCacheMetadata, key_id: str
) -> Optional[MemoryObj]:
    o = backend.allocate(md.get_shapes(256), md.get_dtypes(), fmt=None)
    if o is None:
        return None
    backend.submit_put_task(CacheEngineKey("m", 1, 0, hash(key_id), torch.bfloat16), o)
    o.ref_count_down()  # model: cache owns exactly one reference
    return o


def _build_engine(
    backend: LocalCPUBackend, md: LMCacheMetadata, retry_ms: int
) -> tuple[LMCacheEngine, dict]:
    # __new__ bypasses __init__, so every runtime field has to be injected.
    engine = LMCacheEngine.__new__(LMCacheEngine)

    store_stats = SimpleNamespace(
        profile_process_tokens=lambda: contextlib.nullcontext(),
        profile_from_gpu=lambda: contextlib.nullcontext(),
        profile_put=lambda: contextlib.nullcontext(),
        process_tokens_time=0.0,
        from_gpu_time=0.0,
        put_time=0.0,
        time_to_store=lambda: 0.001,
    )
    recorder = {"count": 0}
    recorder_lock = threading.Lock()

    def _record(dropped_count):
        with recorder_lock:
            recorder["count"] += dropped_count

    engine.stats_monitor = SimpleNamespace(
        on_store_request=lambda n: store_stats,
        on_store_finished=lambda *a, **k: None,
        update_store_admission_failure_count=_record,
    )
    engine.gpu_connector = SimpleNamespace(batched_from_gpu=lambda *a, **k: None)
    engine.token_database = SimpleNamespace(
        process_tokens=lambda *a, **k: [
            (0, 256, CacheEngineKey("m", 1, 0, hash("new_a"), torch.bfloat16)),
            (256, 512, CacheEngineKey("m", 1, 0, hash("new_b"), torch.bfloat16)),
            (512, 768, CacheEngineKey("m", 1, 0, hash("new_c"), torch.bfloat16)),
        ]
    )

    class _Proxy:
        def __init__(self, b):
            self.b = b

        def allocate(self, shapes, dtypes, busy_loop=False, fmt=None):
            return self.b.allocate(shapes, dtypes, busy_loop=busy_loop, fmt=fmt)

        def batched_allocate(self, *a, **k):
            return self.b.batched_allocate(*a, **k)

        def batched_put(self, keys, memory_objs, transfer_spec=None, location=None):
            for k, o in zip(keys, memory_objs, strict=False):
                self.b.submit_put_task(k, o)

    engine.storage_manager = _Proxy(backend)
    engine.is_healthy = lambda: True
    engine._is_passive = lambda: False
    engine._get_req_id = lambda *a, **k: ""
    engine._log_kvcache_for_check = lambda *a, **k: None
    engine.is_frozen = lambda: False
    engine.metadata = md
    engine.fmt = MemoryFormat.KV_2LTD
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=256, local_cpu=True, lmcache_instance_id="t"
    )
    cfg.store_admission_retry_ms = retry_ms
    engine.config = cfg
    engine.kv_events_enabled = False
    engine.kv_events = []
    engine.store_location = "cpu"
    engine._init_failed = False
    engine._health_monitor = None
    engine._store_drop_lock = threading.Lock()
    engine._last_store_drop_warning_time = 0.0
    engine._store_drops_since_warning = 0
    return engine, recorder


def _fill_and_pin(backend: LocalCPUBackend, md: LMCacheMetadata) -> list[MemoryObj]:
    base = []
    for i in range(4):
        o = _put(backend, md, f"base_{i}")
        assert o is not None
        base.append(o)
    # in-flight requests hold references -> candidates cannot be evicted
    for o in base:
        o.ref_count_up()
    return base


def test_store_drop_on_admission_failure_default():
    """Default (retry_ms=0): chunks dropped, but the drop event is counted."""
    backend, md, _ = _make_backend(4)
    _fill_and_pin(backend, md)
    engine, recorder = _build_engine(backend, md, retry_ms=0)

    engine.store(tokens=list(range(768)), mask=None)

    present = {k.chunk_hash for k in backend.hot_cache.keys()}
    assert hash("new_a") not in present
    assert hash("new_b") not in present
    assert hash("new_c") not in present
    assert recorder["count"] == 1


def test_store_admission_retry_success():
    """retry_ms>0: an in-flight release lets the store succeed, nothing dropped."""
    backend, md, _ = _make_backend(4)
    base = _fill_and_pin(backend, md)
    # Equal to the cap: a realistic budget must still cover a pin release.
    engine, recorder = _build_engine(backend, md, retry_ms=STORE_ADMISSION_RETRY_MAX_MS)

    def _release():
        time.sleep(0.05)
        for o in base:
            o.ref_count_down()

    threading.Thread(target=_release).start()

    engine.store(tokens=list(range(768)), mask=None)

    present = {k.chunk_hash for k in backend.hot_cache.keys()}
    assert hash("new_a") in present
    assert hash("new_b") in present
    assert hash("new_c") in present
    assert recorder["count"] == 0


def test_retry_preserves_force_store_wait() -> None:
    """Retries must reuse the same ``busy_loop`` as the first allocation.

    Regression guard: ``force_store_wait`` lives in ``extra_config`` and is
    read via ``get_extra_config_value``, so it must be resolved and passed
    explicitly on every retry attempt. This stub records the value passed
    to each attempt.
    """
    backend, md, _ = _make_backend(4)
    _fill_and_pin(backend, md)
    engine, recorder = _build_engine(backend, md, retry_ms=5)
    engine.config.extra_config = {"force_store_wait": True}

    seen_busy_loop: list[bool] = []

    class _AlwaysFailingStorageManager:
        def allocate(
            self,
            shapes: object,
            dtypes: object,
            busy_loop: bool = False,
            fmt: object = None,
        ) -> None:
            seen_busy_loop.append(busy_loop)
            return None

    engine.storage_manager = _AlwaysFailingStorageManager()

    engine.store(tokens=list(range(768)), mask=None)

    # first attempt + at least one retry, all honouring the configured value
    assert len(seen_busy_loop) >= 2
    assert all(value is True for value in seen_busy_loop)
    assert recorder["count"] == 1


def test_retry_budget_is_capped() -> None:
    """An oversized retry_ms must not stall a doomed store for seconds."""
    backend, md, _ = _make_backend(4)
    _fill_and_pin(backend, md)
    engine, recorder = _build_engine(backend, md, retry_ms=10_000)

    class _AlwaysFailingStorageManager:
        def allocate(self, *args, **kwargs) -> None:
            return None

    engine.storage_manager = _AlwaysFailingStorageManager()

    started = time.monotonic()
    engine.store(tokens=list(range(768)), mask=None)
    elapsed = time.monotonic() - started

    # Without the cap this would take the full 10 seconds.
    assert elapsed < STORE_ADMISSION_RETRY_MAX_MS / 1000.0 + 0.5
    assert recorder["count"] == 1


def test_drop_warning_state_is_thread_safe() -> None:
    """Concurrent drops must neither lose counts nor trip the limiter twice."""
    backend, md, _ = _make_backend(4)
    _fill_and_pin(backend, md)
    engine, recorder = _build_engine(backend, md, retry_ms=0)
    # Force an expired window so the first contender would emit the warning.
    engine._last_store_drop_warning_time = (
        time.monotonic() - 2 * STORE_DROP_WARNING_INTERVAL_SEC
    )

    n_threads = 32
    barrier = threading.Barrier(n_threads)

    def _drop(i: int) -> None:
        barrier.wait()
        engine._on_store_chunk_dropped(f"req_{i}", 0)

    threads = [threading.Thread(target=_drop, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every event reached the counter; exactly one contender won the warning
    # and reset the aggregate, so the remaining n-1 stayed accumulated.
    assert recorder["count"] == n_threads
    assert engine._store_drops_since_warning == n_threads - 1
