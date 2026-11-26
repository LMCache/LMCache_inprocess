# SPDX-License-Identifier: Apache-2.0
"""``request_token_ids_as_buffer``: token ids as one int32 buffer.

For every request the scheduler side of the vLLM connector keeps the token ids
scheduled so far and sends the ones to load or save to every worker. As a
Python list that is one int object per token: copied twice when the request
arrives, pickled once per step and unpickled once per worker. With the option
they are one ``array('i')`` from the scheduler to the token database.

The tests compare everything derived from the ids (request metadata, chunk
hashes, cache keys) with the list path: it must be the same.
"""

# Standard
from array import array
from types import SimpleNamespace

# vLLM pickles the connector metadata from the scheduler to the workers: these
# tests measure and replay exactly that.
import pickle  # noqa: TID251
import random

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.inprocess.integration.vllm import vllm_v1_adapter as adapter
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.token_database import ChunkedTokenDatabase

# First Party (Tests)
from tests.inprocess.utils import dumb_metadata

BLOCK_SIZE = 16
CHUNK_SIZE = 256
NUM_TOKENS = 4000


def _token_ids(seed: int = 0, n: int = NUM_TOKENS) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, 150_000) for _ in range(n)]


def _config(as_buffer: bool) -> LMCacheEngineConfig:
    return LMCacheEngineConfig.from_legacy(
        chunk_size=CHUNK_SIZE,
        backend="cpu",
        extra_config={"request_token_ids_as_buffer": as_buffer},
    )


def _new_request(token_ids) -> SimpleNamespace:
    return SimpleNamespace(
        req_id="req-0",
        prompt_token_ids=token_ids,
        block_ids=([*range(100, 100 + -(-len(token_ids) // BLOCK_SIZE))],),
        sampling_params=None,
        mm_features=None,
    )


def _tracker(as_buffer: bool, token_ids, num_tokens: int) -> "adapter.RequestTracker":
    return adapter.RequestTracker.from_new_request(
        _config(as_buffer),
        _new_request(token_ids),  # type: ignore[arg-type]
        num_tokens,
        0,
        False,
    )


def test_token_ids_buffer_copies_lists_and_vllm_token_buffers() -> None:
    ids = _token_ids()

    from_list = adapter.token_ids_buffer(ids, 1000)
    assert type(from_list) is array and from_list.typecode == "i"
    assert from_list.tolist() == ids[:1000]
    assert adapter.token_ids_buffer(ids).tolist() == ids

    token_buffer = pytest.importorskip("vllm.v1.token_buffer").TokenBuffer(ids)
    from_vllm = adapter.token_ids_buffer(token_buffer, 1000)
    # A plain array, not vLLM's subclass (whose slices are lists), and a copy.
    assert type(from_vllm) is array and from_vllm.typecode == "i"
    assert from_vllm.tolist() == ids[:1000]
    from_vllm[0] += 1
    assert token_buffer[0] == ids[0]


@pytest.mark.parametrize("as_buffer", [False, True])
def test_new_request_tracker_holds_the_same_ids(as_buffer: bool) -> None:
    ids = _token_ids()
    tracker = _tracker(as_buffer, ids, 3000)

    assert isinstance(tracker.token_ids, array if as_buffer else list)
    assert list(tracker.token_ids) == ids[:3000]


def test_tracker_updates_keep_the_buffer() -> None:
    ids = _token_ids()
    tracker = _tracker(True, ids, 2048)

    # A later prefill chunk appends ...
    tracker.update(ids[2048:3000], ([900, 901],))
    assert isinstance(tracker.token_ids, array)
    assert tracker.token_ids.tolist() == ids[:3000]

    # ... and a preempted request restarts from the request's ids.
    tracker.update(
        ids[1024:1500],
        ([7, 8, 9],),
        preempted=True,
        lmcache_cached_tokens=1024,
        vllm_cached_tokens=0,
        all_token_ids=ids,
    )
    assert isinstance(tracker.token_ids, array)
    assert tracker.token_ids.tolist() == ids[:1500]
    assert tracker.allocated_block_ids == [7, 8, 9]


def test_request_metadata_is_the_same_and_pickles_smaller() -> None:
    ids = _token_ids()
    metas = {}
    for as_buffer in (False, True):
        tracker = _tracker(as_buffer, ids, NUM_TOKENS)
        metas[as_buffer] = adapter.ReqMeta.from_request_tracker(
            tracker, BLOCK_SIZE, CHUNK_SIZE, discard_partial_chunks=True
        )
    as_list, as_buffer_meta = metas[False], metas[True]
    assert as_list is not None and as_buffer_meta is not None

    assert isinstance(as_buffer_meta.token_ids, array)
    assert list(as_buffer_meta.token_ids) == list(as_list.token_ids)
    assert len(as_buffer_meta.token_ids) == NUM_TOKENS // CHUNK_SIZE * CHUNK_SIZE
    assert torch.equal(as_buffer_meta.slot_mapping, as_list.slot_mapping)
    assert as_buffer_meta.save_spec == as_list.save_spec
    assert as_buffer_meta.is_last_prefill == as_list.is_last_prefill

    restored = pickle.loads(pickle.dumps(as_buffer_meta))
    assert isinstance(restored.token_ids, array)
    assert restored.token_ids == as_buffer_meta.token_ids

    # The ids alone: 4 bytes per token instead of a pickled int per token.
    ids_as_list = len(pickle.dumps(list(as_list.token_ids)))
    ids_as_buffer = len(pickle.dumps(as_buffer_meta.token_ids))
    assert ids_as_buffer <= 4 * len(as_list.token_ids) + 200
    assert ids_as_buffer < ids_as_list


@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_chunk_hashes_and_keys_of_a_buffer_equal_those_of_a_list(
    save_unfull_chunk: bool,
) -> None:
    config = LMCacheEngineConfig.from_legacy(
        chunk_size=CHUNK_SIZE, backend="cpu", save_unfull_chunk=save_unfull_chunk
    )
    ids = _token_ids()
    reference_db = ChunkedTokenDatabase(config, dumb_metadata())
    reference = list(reference_db.process_tokens(tokens=ids, make_key=False))
    reference_keys = list(reference_db.process_tokens(tokens=ids))

    db = ChunkedTokenDatabase(config, dumb_metadata())
    first = list(db.process_tokens(tokens=array("i", ids), make_key=False))
    # The second pass takes every chunk hash from the memo.
    again = list(db.process_tokens(tokens=array("i", ids), make_key=False))
    keys = list(db.process_tokens(tokens=array("i", ids)))

    assert first == reference
    assert again == reference
    assert keys == reference_keys


def test_memoized_chunk_hashes_follow_the_prefix() -> None:
    """The memo is keyed by the previous hash too: the same chunk after a
    different prefix, and a different chunk after the same prefix, hash as the
    list path hashes them."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=CHUNK_SIZE, backend="cpu")
    shared, tail_a, tail_b = (
        _token_ids(1, 4 * CHUNK_SIZE),
        _token_ids(2, CHUNK_SIZE),
        _token_ids(3, CHUNK_SIZE),
    )
    other_prefix = _token_ids(4, 4 * CHUNK_SIZE)
    prompts = [shared + tail_a, shared + tail_b, other_prefix + tail_a]

    reference_db = ChunkedTokenDatabase(config, dumb_metadata())
    db = ChunkedTokenDatabase(config, dumb_metadata())
    hashes = []
    for prompt in prompts:
        reference = [
            h for _, _, h in reference_db.process_tokens(tokens=prompt, make_key=False)
        ]
        buffered = [
            h
            for _, _, h in db.process_tokens(tokens=array("i", prompt), make_key=False)
        ]
        assert buffered == reference
        hashes.append(buffered)

    # Shared prefix: same hashes for its chunks, different ones for the tails.
    assert hashes[0][:4] == hashes[1][:4]
    assert hashes[0][4] != hashes[1][4]
    # Same last chunk after another prefix: a different hash.
    assert hashes[0][4] != hashes[2][4]
