# SPDX-License-Identifier: Apache-2.0
"""``skip_lookup_local_tail_tokens``: no lookup for a prompt vLLM already holds.

Before it schedules a request, vLLM asks the connector how many tokens beyond
its own prefix-cache hit could be loaded. The connector answers with a lookup,
and with async loading the scheduler waits for it. When vLLM's prefix cache
already holds all of the prompt but a short tail, the lookup can add at most
that tail.

With the option set, such a request is not looked up: the connector reports no
external tokens and records the whole chunks of the locally cached prefix as
already stored, so that the save path stores only what follows them.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    ReqMeta,
    RequestTracker,
)

BLOCK_SIZE = 16
CHUNK_SIZE = 256
REQ_ID = "req-0"


class _LookupClient:
    """Counts lookups and answers them with ``hit_tokens``."""

    def __init__(self, hit_tokens: int = 0) -> None:
        self.hit_tokens = hit_tokens
        self.lookups: list[str] = []

    def lookup_cache(self, lookup_id: str) -> int:
        return -1

    def lookup(self, token_ids, lookup_id: str, request_configs=None) -> int:
        self.lookups.append(lookup_id)
        return self.hit_tokens


def _make_connector(
    skip_tail_tokens: int, hit_tokens: int = 0
) -> tuple[LMCacheConnectorV1Impl, _LookupClient]:
    client = _LookupClient(hit_tokens)
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    # ``lookup_client`` is a read-only property backed by ``self._manager``.
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lookup_client=client
    )
    connector.kv_role = "kv_both"
    connector.skip_last_n_tokens = 0
    connector._skip_lookup_local_tail_tokens = skip_tail_tokens
    connector._lmcache_chunk_size = CHUNK_SIZE
    connector._max_tokens_per_load = 0
    connector._requests_priority = {}
    connector.load_specs = {}
    connector.config = SimpleNamespace(  # type: ignore[assignment]
        min_retrieve_tokens=0
    )
    return connector, client


def _request(num_tokens: int, priority: int = 0) -> SimpleNamespace:
    token_ids = list(range(num_tokens))
    return SimpleNamespace(
        request_id=REQ_ID,
        num_tokens=num_tokens,
        all_token_ids=token_ids,
        prompt_token_ids=token_ids,
        sampling_params=None,
        priority=priority,
        mm_features=None,
    )


def test_short_tail_after_a_local_hit_is_not_looked_up() -> None:
    connector, client = _make_connector(skip_tail_tokens=512)
    request = _request(num_tokens=1500, priority=3)

    # vLLM's prefix cache holds 1,200 tokens: 300 are left to compute.
    assert connector.get_num_new_matched_tokens(request, 1200) == 0

    assert client.lookups == []
    load_spec = connector.load_specs[REQ_ID]
    assert load_spec.vllm_cached_tokens == 1200
    # Four whole chunks lie inside the locally cached prefix.
    assert load_spec.lmcache_cached_tokens == 1024
    assert load_spec.lmcache_cached_tokens % CHUNK_SIZE == 0
    assert load_spec.can_load is False
    # The priority a lookup would have recorded is still there for the save.
    assert connector._requests_priority[REQ_ID] == 3


@pytest.mark.parametrize(
    ("skip_tail_tokens", "num_tokens", "num_computed_tokens"),
    [
        # Option off (the default): every request is looked up.
        (0, 1500, 1200),
        # Nothing in vLLM's prefix cache: LMCache may hold the prompt.
        (512, 300, 0),
        # The tail is longer than the option allows.
        (512, 4000, 1200),
    ],
)
def test_other_requests_are_looked_up(
    skip_tail_tokens: int, num_tokens: int, num_computed_tokens: int
) -> None:
    hit_tokens = 2048 if num_tokens > 2048 else 0
    connector, client = _make_connector(skip_tail_tokens, hit_tokens=hit_tokens)
    request = _request(num_tokens)

    result = connector.get_num_new_matched_tokens(request, num_computed_tokens)

    assert client.lookups == [REQ_ID]
    assert result == max(hit_tokens - num_computed_tokens, 0)
    assert connector.load_specs[REQ_ID].lmcache_cached_tokens == hit_tokens


def test_save_after_a_skipped_lookup_starts_at_the_recorded_chunk_boundary() -> None:
    """The save that follows stores from the chunk boundary the skipped lookup
    recorded: the chunks inside the locally cached prefix are not stored again,
    the chunk the tail completes is."""
    connector, _ = _make_connector(skip_tail_tokens=512)
    num_tokens, num_computed_tokens = 1500, 1200
    request = _request(num_tokens)
    assert connector.get_num_new_matched_tokens(request, num_computed_tokens) == 0
    load_spec = connector.load_specs[REQ_ID]

    tracker = RequestTracker(
        req_id=REQ_ID,
        prompt_len=num_tokens,
        token_ids=list(range(num_tokens)),
        allocated_block_ids=list(range(-(-num_tokens // BLOCK_SIZE))),
        num_saved_tokens=load_spec.lmcache_cached_tokens,
        num_lmcache_cached_tokens=load_spec.lmcache_cached_tokens,
    )
    req_meta = ReqMeta.from_request_tracker(
        tracker,
        BLOCK_SIZE,
        CHUNK_SIZE,
        load_spec=load_spec,
        discard_partial_chunks=True,
    )

    assert req_meta is not None
    assert req_meta.save_spec is not None
    assert req_meta.save_spec.can_save is True
    assert req_meta.save_spec.skip_leading_tokens == 1024
    # Chunk 5 (tokens 1024..1279) is complete; the partial sixth is left out.
    assert len(req_meta.token_ids) == 1280
    assert tracker.num_saved_tokens == 1280
