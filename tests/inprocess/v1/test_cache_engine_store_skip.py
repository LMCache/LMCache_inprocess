# SPDX-License-Identifier: Apache-2.0
"""LMCacheEngine.store with ``store_skip_chunks_in_local_cpu``."""

# Standard
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.inprocess.v1.cache_engine import LMCacheEngineBuilder
from lmcache.inprocess.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from tests.inprocess.utils import create_test_config, create_test_metadata

CHUNK = 256


class RecordingGPUConnector(MockGPUConnector):
    """Records which token ranges a store copies from the GPU."""

    def __init__(self, kv_shape):
        super().__init__(kv_shape)
        self.copied = []

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        self.copied.append(list(zip(starts, ends, strict=True)))


@pytest.fixture
def make_engine():
    instance_ids = []

    def _make(extra_config=None):
        instance_id = f"test_store_skip_{uuid.uuid4().hex[:8]}"
        metadata = create_test_metadata()
        engine = LMCacheEngineBuilder.get_or_create(
            instance_id=instance_id,
            config=create_test_config(
                instance_id=instance_id, extra_config=extra_config
            ),
            metadata=metadata,
            gpu_connector=RecordingGPUConnector(kv_shape=metadata.kv_shape),
            broadcast_fn=mock_up_broadcast_fn,
            broadcast_object_fn=mock_up_broadcast_object_fn,
        )
        engine.post_init()
        instance_ids.append(instance_id)
        return engine

    yield _make
    for instance_id in instance_ids:
        LMCacheEngineBuilder.destroy(instance_id)


def _tokens(num_tokens):
    return torch.arange(num_tokens)


def test_store_copies_held_chunks_again_by_default(make_engine):
    engine = make_engine()
    tokens = _tokens(3 * CHUNK)

    engine.store(tokens=tokens)
    engine.store(tokens=tokens)

    whole = [(0, CHUNK), (CHUNK, 2 * CHUNK), (2 * CHUNK, 3 * CHUNK)]
    assert engine.gpu_connector.copied == [whole, whole]


def test_store_skips_chunks_the_local_cpu_backend_holds(make_engine):
    engine = make_engine({"store_skip_chunks_in_local_cpu": True})
    tokens = _tokens(3 * CHUNK)

    engine.store(tokens=tokens[: 2 * CHUNK])
    engine.store(tokens=tokens)
    engine.store(tokens=tokens)

    assert engine.gpu_connector.copied == [
        [(0, CHUNK), (CHUNK, 2 * CHUNK)],
        [(2 * CHUNK, 3 * CHUNK)],
    ]
    assert engine.lookup(tokens) == 3 * CHUNK


def test_store_skip_does_not_apply_to_another_store_location(make_engine):
    engine = make_engine({"store_skip_chunks_in_local_cpu": True})
    tokens = _tokens(CHUNK)
    engine.store(tokens=tokens)

    engine.store_location = "RemoteBackend"
    engine.store(tokens=tokens)

    assert engine.gpu_connector.copied == [[(0, CHUNK)], [(0, CHUNK)]]
