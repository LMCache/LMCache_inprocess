# SPDX-License-Identifier: Apache-2.0
"""RemoteBackend writes with ``remote_put_dedup_ttl_secs``."""

# Standard
from typing import List
from unittest.mock import patch
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.inprocess.v1.storage_backend import remote_backend as remote_backend_module
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
TTL = 600


class WriteConnector(RemoteConnector):
    """Records every write. A write can be held back or made to fail."""

    def __init__(self):
        self.stored = set()
        self.writes: List[list] = []
        self.active = 0
        self.write_started = threading.Event()
        self.hold = None  # an asyncio.Event the writes wait for
        self.fail = False

    async def exists(self, key):
        return key in self.stored

    def exists_sync(self, key):
        return key in self.stored

    async def get(self, key):
        return None

    async def list(self) -> List[str]:
        return []

    async def close(self):
        pass

    async def _write(self, keys, memory_objs):
        self.writes.append(list(keys))
        self.write_started.set()
        self.active += 1
        try:
            if self.hold is not None:
                await self.hold.wait()
            if self.fail:
                raise RuntimeError("write failed")
            self.stored.update(keys)
        finally:
            self.active -= 1
            # The instrumented connector does this around a real connector.
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()

    async def put(self, key, memory_obj):
        await self._write([key], [memory_obj])

    def support_batched_put(self) -> bool:
        return True

    async def batched_put(self, keys, memory_objs):
        await self._write(keys, memory_objs)

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys) -> int:
        num_hit = 0
        for key in keys:
            if key not in self.stored:
                break
            num_hit += 1
        return num_hit

    def support_batched_async_contains(self) -> bool:
        return True

    async def batched_async_contains(self, lookup_id, keys, pin=False) -> int:
        return self.batched_contains(keys)


class Env:
    def __init__(self, loop, ttl):
        extra_config = {} if ttl is None else {"remote_put_dedup_ttl_secs": ttl}
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
        self.loop = loop
        self.local_cpu_backend = LocalCPUBackend(
            config, metadata, memory_allocator=PinMemoryAllocator(1024 * 1024 * 1024)
        )
        self.backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=loop,
            local_cpu_backend=self.local_cpu_backend,
        )
        self.connector = WriteConnector()
        self.backend.connection = self.connector
        self.keys = [dumb_cache_engine_key(i) for i in range(3)]
        self.memory_objs = [
            self.local_cpu_backend.allocate(CHUNK_SHAPE, CHUNK_DTYPE) for _ in self.keys
        ]

    def put(self, keys=None, wait=True):
        keys = self.keys if keys is None else keys
        memory_objs = [self.memory_objs[self.keys.index(key)] for key in keys]
        self.backend.batched_submit_put_task(keys, memory_objs)
        if wait:
            self.wait_idle()

    def wait_idle(self):
        """Wait until every submitted write and its callback have run."""

        async def drain():
            # Writes submitted before this coroutine have started by now.
            for _ in range(5000):
                if self.connector.active == 0:
                    break
                await asyncio.sleep(0.001)
            else:
                raise AssertionError("writes did not finish")
            # Let their completion callbacks run.
            for _ in range(5):
                await asyncio.sleep(0)

        asyncio.run_coroutine_threadsafe(drain(), self.loop).result(timeout=10)

    def on_loop(self, fn):
        async def call():
            return fn()

        return asyncio.run_coroutine_threadsafe(call(), self.loop).result(timeout=10)


@pytest.fixture
def async_loop():
    loop, thread = init_asyncio_loop()
    yield loop
    close_asyncio_loop(loop, thread)


def test_chunk_being_written_is_not_written_again(async_loop):
    env = Env(async_loop, TTL)
    env.connector.hold = env.on_loop(asyncio.Event)

    env.put(wait=False)
    assert env.connector.write_started.wait(timeout=5)
    env.put(wait=False)
    env.on_loop(env.connector.hold.set)
    env.wait_idle()

    assert env.connector.writes == [env.keys]
    assert [memory_obj.get_ref_count() for memory_obj in env.memory_objs] == [1, 1, 1]


def test_chunk_written_within_the_ttl_is_not_written_again(async_loop):
    env = Env(async_loop, TTL)
    now = remote_backend_module.time.monotonic()

    env.put()
    env.put()
    env.put(env.keys[:2])
    assert env.connector.writes == [env.keys]

    with patch.object(
        remote_backend_module.time, "monotonic", return_value=now + TTL + 60
    ):
        env.put()
    assert env.connector.writes == [env.keys, env.keys]


def test_only_new_chunks_of_a_batch_are_written(async_loop):
    env = Env(async_loop, TTL)

    env.put(env.keys[:2])
    env.put(env.keys)

    assert env.connector.writes == [env.keys[:2], env.keys[2:]]


def test_chunk_missing_in_the_remote_store_is_written_again(async_loop):
    env = Env(async_loop, TTL)
    env.put()

    # The remote store evicts the second chunk; the next lookup notices.
    env.connector.stored.discard(env.keys[1])
    num_hit = asyncio.run_coroutine_threadsafe(
        env.backend.batched_async_contains("lookup", env.keys), async_loop
    ).result(timeout=10)
    assert num_hit == 1
    env.put()

    assert env.connector.writes == [env.keys, [env.keys[1]]]


def test_sync_lookup_miss_also_allows_the_write(async_loop):
    env = Env(async_loop, TTL)
    env.put()

    env.connector.stored.discard(env.keys[0])
    assert env.backend.batched_contains(env.keys) == 0
    env.put()

    assert env.connector.writes == [env.keys, [env.keys[0]]]


def test_failed_write_is_retried(async_loop):
    env = Env(async_loop, TTL)

    env.connector.fail = True
    env.put()
    env.connector.fail = False
    env.put()

    assert env.connector.writes == [env.keys, env.keys]


def test_failed_serialization_does_not_block_later_writes(async_loop):
    env = Env(async_loop, TTL)

    with patch.object(
        env.backend.serializer, "serialize", side_effect=RuntimeError("serde")
    ):
        with pytest.raises(RuntimeError):
            env.put(wait=False)
    assert not env.backend.put_tasks
    env.put()

    assert env.connector.writes == [env.keys]
    assert [memory_obj.get_ref_count() for memory_obj in env.memory_objs] == [1, 1, 1]


def test_single_put_honours_the_ttl(async_loop):
    env = Env(async_loop, TTL)

    for _ in range(2):
        env.backend.submit_put_task(env.keys[0], env.memory_objs[0])
        env.wait_idle()

    assert env.connector.writes == [[env.keys[0]]]


def test_writes_are_unchanged_without_the_option(async_loop):
    env = Env(async_loop, None)

    env.put()
    env.put()

    assert env.connector.writes == [env.keys, env.keys]
