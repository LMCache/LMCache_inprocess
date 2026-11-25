# SPDX-License-Identifier: Apache-2.0
"""``build_connector_meta`` with vLLM's ``CachedRequestData`` as of v0.34.

vllm-project/vllm#63373 removed ``CachedRequestData.resumed_req_ids``: a
request resumed after preemption is re-sent in full through
``scheduled_new_reqs`` instead. The adapter used that field (or the older
``resumed_from_preemption``) to tell whether a cached request was resumed and
raised ``AttributeError`` when neither existed, which killed the engine core on
the first step that scheduled a running request again.

These tests feed ``build_connector_meta`` a running request in each of the
three shapes and check that the tracker advances, and that the two older
shapes still take the resumed path.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    LoadSpec,
    RequestTracker,
)

BLOCK_SIZE = 4
CHUNK_SIZE = 8
REQ_ID = "req-0"


def _make_connector(
    prompt: list[int], num_computed: int
) -> tuple[LMCacheConnectorV1Impl, RequestTracker]:
    """A scheduler-side connector that has already scheduled the first
    ``num_computed`` prompt tokens of one request."""
    tracker = RequestTracker(
        req_id=REQ_ID,
        prompt_len=len(prompt),
        token_ids=prompt[:num_computed],
        allocated_block_ids=list(range(num_computed // BLOCK_SIZE)),
        skip_save=True,
    )
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    # ``lookup_client`` is a read-only property backed by ``self._manager``.
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lookup_client=object()
    )
    connector.kv_role = "kv_both"
    connector.force_skip_save = False
    connector._request_trackers = {REQ_ID: tracker}
    connector._unfinished_requests = {  # type: ignore[assignment]
        REQ_ID: SimpleNamespace(num_computed_tokens=num_computed, all_token_ids=prompt)
    }
    connector.load_specs = {}
    connector._block_size = BLOCK_SIZE
    connector._lmcache_chunk_size = CHUNK_SIZE
    connector._discard_partial_chunks = True
    connector.config = SimpleNamespace(  # type: ignore[assignment]
        save_decode_cache=False, priority_limit=None
    )
    return connector, tracker


def _scheduler_output(cached_reqs: SimpleNamespace, num_new_tokens: int):
    return SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens={REQ_ID: num_new_tokens},
    )


def test_cached_request_without_resumed_fields_is_not_resumed() -> None:
    """vLLM >= 0.34: no ``resumed_req_ids``, no ``resumed_from_preemption``."""
    prompt = list(range(100, 124))
    connector, tracker = _make_connector(prompt, num_computed=8)
    cached_reqs = SimpleNamespace(
        req_ids=[REQ_ID],
        new_token_ids=[[]],
        new_block_ids=[([2, 3],)],
        num_computed_tokens=[8],
        num_output_tokens=[0],
    )
    assert not hasattr(cached_reqs, "resumed_req_ids")
    assert not hasattr(cached_reqs, "resumed_from_preemption")

    connector.build_connector_meta(_scheduler_output(cached_reqs, 8))

    assert tracker.token_ids == prompt[:16]
    assert tracker.allocated_block_ids == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "resumed_fields",
    [
        {"resumed_req_ids": {REQ_ID}},
        {"resumed_from_preemption": [True]},
    ],
)
def test_older_cached_request_shapes_still_take_the_resumed_path(
    resumed_fields: dict,
) -> None:
    """vLLM < 0.34 reports a resumed request through the cached requests: the
    tracker restarts from the request's tokens and its new blocks."""
    prompt = list(range(100, 124))
    connector, tracker = _make_connector(prompt, num_computed=16)
    # Preemption reset the request; LMCache holds its first chunk.
    connector._unfinished_requests[REQ_ID].num_computed_tokens = 8
    connector.load_specs[REQ_ID] = LoadSpec(
        vllm_cached_tokens=0, lmcache_cached_tokens=8, can_load=True
    )
    cached_reqs = SimpleNamespace(
        req_ids=[REQ_ID],
        new_token_ids=[[]],
        new_block_ids=[([7, 8, 9, 10],)],
        num_computed_tokens=[8],
        num_output_tokens=[0],
        **resumed_fields,
    )

    connector.build_connector_meta(_scheduler_output(cached_reqs, 8))

    assert tracker.allocated_block_ids == [7, 8, 9, 10]
    assert tracker.token_ids == prompt[:16]
