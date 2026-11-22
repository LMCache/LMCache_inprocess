# SPDX-License-Identifier: Apache-2.0
"""RemoteBackend prefetch with ``remote_prefetch_write_back``."""

# Standard
from typing import List
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.inprocess.v1.storage_backend.connector import RemoteConnector
from lmcache.inprocess.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.inprocess.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from tests.inprocess.utils import (
    close_asyncio_loop,
    dumb_cache_engine_key,
    init_asyncio_loop,
)

CHUNK_SHAPE = torch.Size([2, 4, 256, 128])
CHUNK_DTYPE = torch.bfloat16


class PrefetchConnector(RemoteConnector):
    """Holds a set of keys and hands out a freshly allocated chunk per hit."""

    def __init__(self, local_cpu_backend):
        self.local_cpu_backend = local_cpu_backend
        self.stored = set()
        self.fetches: List[list] = []
        # Set to make every fetch wait until the test cancels it.
        self.hang = False

    async def exists(self, key):
        return key in self.stored

    def exists_sync(self, key):
        return key in self.stored

    async def get(self, key):
        return None

    async def put(self, key, memory_obj):
        self.stored.add(key)

    async def list(self) -> List[str]:
        return []

    async def close(self):
        pass

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def batched_get_non_blocking(self, lookup_id, keys):
        self.fetches.append(list(keys))
        if self.hang:
            await asyncio.Future()
        return [
            self.local_cpu_backend.allocate(CHUNK_SHAPE, CHUNK_DTYPE)
            if key in self.stored
            else None
            for key in keys
        ]


@pytest.fixture
def async_loop():
    loop, thread = init_asyncio_loop()
    yield loop
    close_asyncio_loop(loop, thread)


def _create_remote_backend(loop, write_back):
    extra_config = {"remote_prefetch_write_back": True} if write_back else {}
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=1.0,
        remote_url="mock://100",
        remote_serde="naive",
        extra_config=extra_config,
    )
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=CHUNK_DTYPE,
        kv_shape=(4, 2, 256, 8, 16),
    )
    local_cpu_backend = LocalCPUBackend(
        config, metadata, memory_allocator=PinMemoryAllocator(1024 * 1024 * 1024)
    )
    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop,
        local_cpu_backend=local_cpu_backend,
    )
    backend.connection = PrefetchConnector(local_cpu_backend)
    return backend


def _prefetch(backend, loop, keys):
    return asyncio.run_coroutine_threadsafe(
        backend.batched_get_non_blocking("lookup", keys), loop
    ).result(timeout=10)


def test_prefetched_chunks_are_fetched_once(async_loop):
    backend = _create_remote_backend(async_loop, write_back=True)
    connector = backend.connection
    local_cpu_backend = backend.local_cpu_backend
    keys = [dumb_cache_engine_key(i) for i in range(3)]
    connector.stored.update(keys)

    first = _prefetch(backend, async_loop, keys)
    assert len(first) == 3
    assert connector.fetches == [keys]
    assert all(local_cpu_backend.contains(key) for key in keys)
    # One reference for the caller, one for the local CPU backend.
    assert [memory_obj.get_ref_count() for memory_obj in first] == [2, 2, 2]

    second = _prefetch(backend, async_loop, keys)
    assert connector.fetches == [keys]
    assert [id(memory_obj) for memory_obj in second] == [
        id(memory_obj) for memory_obj in first
    ]

    for memory_obj in first + second:
        memory_obj.ref_count_down()
    assert [memory_obj.get_ref_count() for memory_obj in first] == [1, 1, 1]


def test_only_missing_chunks_are_fetched(async_loop):
    backend = _create_remote_backend(async_loop, write_back=True)
    connector = backend.connection
    keys = [dumb_cache_engine_key(i) for i in range(3)]
    connector.stored.update(keys)

    for memory_obj in _prefetch(backend, async_loop, keys[:1]):
        memory_obj.ref_count_down()
    memory_objs = _prefetch(backend, async_loop, keys)

    assert len(memory_objs) == 3
    assert connector.fetches == [keys[:1], keys[1:]]
    for memory_obj in memory_objs:
        memory_obj.ref_count_down()


def test_chunks_after_a_remote_miss_are_released(async_loop):
    backend = _create_remote_backend(async_loop, write_back=True)
    connector = backend.connection
    local_cpu_backend = backend.local_cpu_backend
    keys = [dumb_cache_engine_key(i) for i in range(3)]
    connector.stored.update([keys[0], keys[2]])

    memory_objs = _prefetch(backend, async_loop, keys)

    assert len(memory_objs) == 1
    assert memory_objs[0].get_ref_count() == 2
    # The chunk after the gap is not handed out; the local CPU backend
    # holds its only reference.
    after_gap = local_cpu_backend.get_blocking(keys[2])
    assert after_gap is not None
    assert after_gap.get_ref_count() == 2
    after_gap.ref_count_down()
    memory_objs[0].ref_count_down()


def test_cancelled_prefetch_releases_local_chunks(async_loop):
    backend = _create_remote_backend(async_loop, write_back=True)
    connector = backend.connection
    keys = [dumb_cache_engine_key(i) for i in range(2)]
    connector.stored.update(keys)

    (local_chunk,) = _prefetch(backend, async_loop, keys[:1])
    local_chunk.ref_count_down()
    assert local_chunk.get_ref_count() == 1

    connector.hang = True

    async def prefetch_then_cancel():
        task = asyncio.ensure_future(backend.batched_get_non_blocking("lookup", keys))
        while len(connector.fetches) < 2:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run_coroutine_threadsafe(prefetch_then_cancel(), async_loop).result(
        timeout=10
    )
    assert local_chunk.get_ref_count() == 1


def test_prefetch_is_unchanged_without_the_option(async_loop):
    backend = _create_remote_backend(async_loop, write_back=False)
    connector = backend.connection
    local_cpu_backend = backend.local_cpu_backend
    keys = [dumb_cache_engine_key(i) for i in range(2)]
    connector.stored.update(keys)

    first = _prefetch(backend, async_loop, keys)
    second = _prefetch(backend, async_loop, keys)

    assert connector.fetches == [keys, keys]
    assert not any(local_cpu_backend.contains(key) for key in keys)
    for memory_obj in first + second:
        assert memory_obj.get_ref_count() == 1
        memory_obj.ref_count_down()
