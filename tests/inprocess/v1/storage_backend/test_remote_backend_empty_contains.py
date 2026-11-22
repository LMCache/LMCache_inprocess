# SPDX-License-Identifier: Apache-2.0
"""RemoteBackend.batched_async_contains with no keys."""

# Standard
from typing import List
import asyncio

# Third Party
import torch

# First Party
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.inprocess.v1.storage_backend.connector import RemoteConnector
from lmcache.inprocess.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.inprocess.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from tests.inprocess.utils import (
    close_asyncio_loop,
    dumb_cache_engine_key,
    init_asyncio_loop,
)


class CountingConnector(RemoteConnector):
    """Answers existence checks from a set and counts the round trips."""

    def __init__(self):
        self.stored = set()
        self.contains_calls: List[int] = []

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

    def support_batched_async_contains(self) -> bool:
        return True

    async def batched_async_contains(self, lookup_id, keys, pin=False) -> int:
        self.contains_calls.append(len(keys))
        num_hit = 0
        for key in keys:
            if key not in self.stored:
                break
            num_hit += 1
        return num_hit


def _create_remote_backend(loop):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=1.0,
        remote_url="mock://100",
        remote_serde="naive",
    )
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 16),
    )
    local_cpu_backend = LocalCPUBackend(
        config, metadata, memory_allocator=AdHocMemoryAllocator()
    )
    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop,
        local_cpu_backend=local_cpu_backend,
    )
    backend.connection = CountingConnector()
    return backend


def test_batched_async_contains_without_keys_makes_no_round_trip():
    async_loop, async_thread = init_asyncio_loop()
    try:
        backend = _create_remote_backend(async_loop)
        connector = backend.connection
        key = dumb_cache_engine_key(1)
        connector.stored.add(key)

        def contains(keys):
            return asyncio.run_coroutine_threadsafe(
                backend.batched_async_contains("lookup", keys), async_loop
            ).result(timeout=10)

        assert contains([]) == 0
        assert connector.contains_calls == []

        assert contains([key, dumb_cache_engine_key(2)]) == 1
        assert connector.contains_calls == [2]
    finally:
        close_asyncio_loop(async_loop, async_thread)
