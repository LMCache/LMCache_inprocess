# SPDX-License-Identifier: Apache-2.0
"""``slot_mapping_from_block_ids``: send block ids, not the slot mapping.

The scheduler side of the vLLM connector builds, for every request with
something to load or save, a slot mapping with one int64 per token and sends it
to every worker. It is a function of the request's block ids and the block
size, so with the option the block ids travel instead and each worker builds
the mapping where it is used.

The tests check that the worker ends up with the mapping the scheduler would
have sent, and that the step that carries it is smaller.
"""

# Standard
from types import SimpleNamespace

# vLLM pickles the connector metadata from the scheduler to the workers: these
# tests measure and replay exactly that.
import pickle  # noqa: TID251

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.inprocess.integration.vllm import vllm_v1_adapter as adapter

BLOCK_SIZE = 16
CHUNK_SIZE = 256


def _tracker(num_tokens: int, num_blocks: int) -> "adapter.RequestTracker":
    return adapter.RequestTracker(
        req_id="req-0",
        prompt_len=num_tokens,
        token_ids=list(range(num_tokens)),
        # Not contiguous, as blocks come out of vLLM's pool.
        allocated_block_ids=[1000 + 7 * i % 4099 for i in range(num_blocks)],
    )


def _metas(num_tokens: int, num_blocks: int, **kwargs):
    sent = adapter.ReqMeta.from_request_tracker(
        _tracker(num_tokens, num_blocks), BLOCK_SIZE, CHUNK_SIZE, **kwargs
    )
    from_blocks = adapter.ReqMeta.from_request_tracker(
        _tracker(num_tokens, num_blocks),
        BLOCK_SIZE,
        CHUNK_SIZE,
        slot_mapping_from_block_ids=True,
        **kwargs,
    )
    assert sent is not None and from_blocks is not None
    return sent, from_blocks


@pytest.mark.parametrize(
    ("num_tokens", "num_blocks", "discard_partial_chunks"),
    [
        # Whole chunks only: 3,840 of 4,000 tokens, fewer blocks than allocated.
        (4000, 250, True),
        # Every token, the last block partly used.
        (4000, 250, False),
        # More blocks allocated than the prompt needs yet.
        (1024, 300, True),
    ],
)
def test_worker_rebuilds_the_mapping_the_scheduler_would_send(
    num_tokens: int, num_blocks: int, discard_partial_chunks: bool
) -> None:
    sent, from_blocks = _metas(
        num_tokens, num_blocks, discard_partial_chunks=discard_partial_chunks
    )
    assert sent.slot_mapping is not None and sent.block_ids is None
    assert from_blocks.slot_mapping is None
    assert from_blocks.token_ids == sent.token_ids
    assert from_blocks.save_spec == sent.save_spec

    # What a worker receives.
    received = pickle.loads(pickle.dumps(from_blocks))
    rebuilt = adapter.request_slot_mapping(received)

    assert rebuilt.dtype == torch.long
    assert torch.equal(rebuilt, sent.slot_mapping)
    # Built once per request and worker.
    assert adapter.request_slot_mapping(received) is rebuilt
    # A mapping that was sent is used as it is.
    assert adapter.request_slot_mapping(sent) is sent.slot_mapping


def test_block_ids_are_smaller_than_the_slot_mapping() -> None:
    sent, from_blocks = _metas(64_000, 4000, discard_partial_chunks=True)

    def without_tokens(meta) -> int:
        meta.token_ids = []
        return len(pickle.dumps(meta))

    # 8 bytes per token against one block id per 'BLOCK_SIZE' tokens.
    assert without_tokens(sent) > 8 * 63_000
    assert without_tokens(from_blocks) < 8 * 63_000 // BLOCK_SIZE


def test_tokens_beyond_the_allocated_blocks_keep_the_sent_mapping() -> None:
    """A desynced request (more tokens than its blocks can hold) is reported
    by the save path from the length of the mapping it was sent: that mapping
    still travels, option or not."""
    sent, from_blocks = _metas(4000, 100, discard_partial_chunks=False)

    assert from_blocks.slot_mapping is not None
    assert torch.equal(from_blocks.slot_mapping, sent.slot_mapping)
    assert len(from_blocks.slot_mapping) == 100 * BLOCK_SIZE < len(sent.token_ids)


def test_wait_for_save_stores_with_the_rebuilt_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End of the path: the worker's save hands the cache engine the same
    mapping whether it was sent or rebuilt."""
    monkeypatch.setattr(
        adapter, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    stored = []

    class _Engine:
        def lookup_unpin(self, req_id: str) -> None:
            pass

        def store(self, token_ids, **kwargs) -> None:
            stored.append(kwargs["slot_mapping"].clone())

    for meta in _metas(4000, 250, discard_partial_chunks=True):
        received = pickle.loads(pickle.dumps(meta))
        connector = adapter.LMCacheConnectorV1Impl.__new__(
            adapter.LMCacheConnectorV1Impl
        )
        connector._parent = SimpleNamespace(
            _get_connector_metadata=lambda received=received: (
                adapter.LMCacheConnectorMetadata(requests=[received])
            )
        )
        connector._manager = SimpleNamespace(  # type: ignore[assignment]
            lmcache_engine=_Engine()
        )
        connector.kv_role = "kv_both"
        connector.use_layerwise = False
        connector.enable_blending = False
        connector.device = "cpu"
        connector._lmcache_chunk_size = CHUNK_SIZE
        connector.kv_caches = {"layer0": torch.zeros(1)}
        connector.config = SimpleNamespace(  # type: ignore[assignment]
            pd_bidirectional=False
        )
        connector.wait_for_save()

    assert len(stored) == 2
    assert torch.equal(stored[0], stored[1])
    assert len(stored[0]) == 3840
